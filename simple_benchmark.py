# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional

import httpx


@dataclass
class RequestItem:
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None


@dataclass
class RequestMetrics:
    ok: bool
    error: Optional[str]
    start: float
    first_byte: float
    end: float
    ttft: float
    itl: float
    completion_tokens: int
    prompt_tokens: int


def _role_from_sharegpt(value: str) -> Optional[str]:
    mapping = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }
    return mapping.get(value.lower()) if isinstance(value, str) else None


def _build_messages(entry: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    conversations = entry.get("conversations") or entry.get("conversation")
    if not conversations or not isinstance(conversations, list):
        return None
    messages: List[Dict[str, str]] = []
    for msg in conversations:
        role = _role_from_sharegpt(msg.get("from") or msg.get("role"))
        content = msg.get("value") or msg.get("content")
        if not role or content is None:
            continue
        messages.append({"role": role, "content": content})
    if not messages:
        return None
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()
    if not messages or messages[-1]["role"] != "user":
        return None
    return messages


def _messages_to_prompt(messages: Iterable[Dict[str, str]]) -> str:
    lines = []
    for msg in messages:
        lines.append(f'{msg["role"]}: {msg["content"]}')
    lines.append("assistant:")
    return "\n".join(lines)


def _estimate_messages_tokens(messages: Iterable[Dict[str, str]]) -> int:
    return _estimate_tokens(_messages_to_prompt(messages))


def _estimate_item_tokens(item: RequestItem, endpoint: str) -> int:
    if endpoint == "chat":
        return _estimate_messages_tokens(item.messages or [])
    return _estimate_tokens(item.prompt or "")


def load_requests(
    dataset_path: str,
    num_prompts: int,
    endpoint: str,
    seed: int,
    shuffle: bool,
    input_tokens: Optional[int],
) -> List[RequestItem]:
    rng = random.Random(seed)
    reservoir: List[RequestItem] = []
    total = 0

    with open(dataset_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = _build_messages(entry)
            if not messages:
                continue
            if endpoint == "chat":
                item = RequestItem(messages=messages)
            else:
                item = RequestItem(prompt=_messages_to_prompt(messages))
            if input_tokens and _estimate_item_tokens(item, endpoint) > input_tokens:
                total += 1
                continue

            if len(reservoir) < num_prompts:
                reservoir.append(item)
            elif shuffle:
                idx = rng.randint(0, total)
                if idx < num_prompts:
                    reservoir[idx] = item
            total += 1
            if not shuffle and len(reservoir) >= num_prompts:
                break

    if len(reservoir) < num_prompts:
        raise ValueError(f"Only found {len(reservoir)} prompts in dataset.")
    return reservoir


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped.split()))


async def run_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
) -> RequestMetrics:
    start = time.perf_counter()
    first_byte = 0.0
    end = start
    completion_tokens = 0
    prompt_tokens = 0
    try:
        async with client.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            chunks: List[bytes] = []
            async for chunk in resp.aiter_bytes():
                if chunk and first_byte == 0.0:
                    first_byte = time.perf_counter()
                chunks.append(chunk)
            end = time.perf_counter()
        if first_byte == 0.0:
            first_byte = end
        data = json.loads(b"".join(chunks).decode("utf-8"))
        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        if completion_tokens == 0:
            choice = data.get("choices", [{}])[0]
            text = choice.get("text") or choice.get("message", {}).get("content", "")
            completion_tokens = _estimate_tokens(text)
    except Exception as exc:
        end = time.perf_counter()
        if first_byte == 0.0:
            first_byte = end
        return RequestMetrics(
            ok=False,
            error=str(exc),
            start=start,
            first_byte=first_byte,
            end=end,
            ttft=first_byte - start,
            itl=0.0,
            completion_tokens=0,
            prompt_tokens=0,
        )

    decode_time = max(0.0, end - first_byte)
    ttft = max(0.0, first_byte - start)
    itl = decode_time / max(completion_tokens - 1, 1)
    return RequestMetrics(
        ok=True,
        error=None,
        start=start,
        first_byte=first_byte,
        end=end,
        ttft=ttft,
        itl=itl,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
    )


async def run_benchmark(args: argparse.Namespace) -> None:
    requests = load_requests(
        args.dataset,
        args.num_prompts,
        args.endpoint,
        args.seed,
        args.shuffle,
        args.input_tokens,
    )

    path = "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions"
    base_url = args.base_url.rstrip("/")
    url = f"{base_url}{path}"
    timeout = httpx.Timeout(args.timeout)

    sem = asyncio.Semaphore(args.concurrency)
    results: List[RequestMetrics] = []

    async with httpx.AsyncClient(timeout=timeout) as client:

        async def _task(item: RequestItem) -> None:
            async with sem:
                payload: Dict[str, Any] = {"max_tokens": args.max_output_tokens, "stream": False}
                if args.model:
                    payload["model"] = args.model
                if args.endpoint == "chat":
                    payload["messages"] = item.messages or []
                else:
                    payload["prompt"] = item.prompt or ""
                metrics = await run_request(client, url, payload)
                results.append(metrics)

        tasks = [asyncio.create_task(_task(item)) for item in requests]
        await asyncio.gather(*tasks)

    ok_results = [r for r in results if r.ok]
    if not ok_results:
        raise RuntimeError("All requests failed.")

    wall_start = min(r.start for r in ok_results)
    wall_end = max(r.end for r in ok_results)
    wall_time = max(1e-9, wall_end - wall_start)
    total_completion = sum(r.completion_tokens for r in ok_results)
    total_prompt = sum(r.prompt_tokens for r in ok_results)
    mean_ttft = mean(r.ttft for r in ok_results)
    median_ttft = median(r.ttft for r in ok_results)
    mean_itl = mean(r.itl for r in ok_results)
    output_throughput = total_completion / wall_time

    print("Benchmark Summary")
    print(f"  Requests: {len(ok_results)} ok / {len(results)} total")
    print(f"  Mean TTFT (s): {mean_ttft:.4f}")
    print(f"  Median TTFT (s): {median_ttft:.4f}")
    print(f"  Mean ITL (s): {mean_itl:.4f}")
    print(f"  Output Throughput (tokens/s): {output_throughput:.2f}")
    print(f"  Total Tokens: prompt={total_prompt}, completion={total_completion}")

    errors = [r for r in results if not r.ok]
    if errors:
        print("Errors:")
        for err in errors[:5]:
            print(f"  - {err.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple benchmark for chat/completions endpoints.")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--endpoint", type=str, choices=["chat", "completion"], default="chat")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--input-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
