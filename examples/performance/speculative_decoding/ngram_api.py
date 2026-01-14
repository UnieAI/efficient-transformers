# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from prompt_lookup import find_candidate_pred_tokens


def parse_device_group(value: str) -> List[int]:
    if not value:
        return [0]
    return [int(x) for x in value.split(",") if x.strip()]


def get_padded_input_len(input_len: int, prefill_seq_len: int, ctx_len: int) -> int:
    num_chunks = -(input_len // -prefill_seq_len)
    input_len_padded = num_chunks * prefill_seq_len
    if input_len_padded > ctx_len:
        raise ValueError("input_len rounded to prefill_seq_len multiple must be <= ctx_len")
    return input_len_padded


def prepare_prefill_inputs(
    tokenizer: AutoTokenizer, prompt: str, prefill_seq_len: int, ctx_len: int
) -> Tuple[Dict[str, np.ndarray], List[int], int]:
    enc = tokenizer(prompt, return_tensors="np", padding=False)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = np.ones_like(input_ids)
    input_len = input_ids.shape[1]
    max_prefill_len = max(1, (ctx_len // prefill_seq_len) * prefill_seq_len)
    if input_len > max_prefill_len:
        input_ids = input_ids[:, -max_prefill_len:]
        attention_mask = attention_mask[:, -max_prefill_len:]
        input_len = max_prefill_len
    input_len_padded = get_padded_input_len(input_len, prefill_seq_len, ctx_len)
    if input_len_padded > input_len:
        pad_width = input_len_padded - input_len
        pad_ids = np.full((1, pad_width), tokenizer.pad_token_id, dtype=input_ids.dtype)
        pad_mask = np.zeros((1, pad_width), dtype=attention_mask.dtype)
        input_ids = np.concatenate([input_ids, pad_ids], axis=1)
        attention_mask = np.concatenate([attention_mask, pad_mask], axis=1)
    position_ids = np.where(attention_mask, np.arange(input_len_padded), -1)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "num_logits_to_keep": np.zeros((1, 1), dtype=np.int64),
    }
    prompt_ids = input_ids[0, :input_len].tolist()
    return inputs, prompt_ids, input_len


def filter_inputs(session: QAICInferenceSession, inputs: dict) -> dict:
    return {name: value for name, value in inputs.items() if name in session.input_names}


def run_prefill_on_target(session: QAICInferenceSession, inputs: dict, prefill_seq_len: int) -> dict:
    input_len = inputs["input_ids"].shape[1]
    num_chunks = input_len // prefill_seq_len
    cache_index = np.array([[0]], np.int64)
    outputs = None
    has_attention_mask = "attention_mask" in inputs
    for _ in range(num_chunks):
        chunk_inputs = inputs.copy()
        start = cache_index[0, 0]
        end = start + prefill_seq_len
        chunk_inputs["input_ids"] = inputs["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, start:end]
        if has_attention_mask:
            chunk_inputs["attention_mask"] = inputs["attention_mask"][:, start:end]
        outputs = session.run(filter_inputs(session, chunk_inputs))
        cache_index += prefill_seq_len
    return outputs


def get_binding_dtype(session: QAICInferenceSession, name: str) -> np.dtype:
    for binding in session.bindings:
        if binding.name == name:
            return session.aic_to_np_dtype_mapping[binding.type]
    return np.float32


def split_extra_outputs(session: QAICInferenceSession, keep_outputs: List[str]) -> List[str]:
    keep_set = set(keep_outputs)
    extra_outputs = []
    for name in session.output_names:
        if name in keep_set or name.endswith("_RetainedState"):
            continue
        binding = session.bindings[session.binding_index_map[name]]
        if binding.is_partial_buf_allowed:
            session.skip_buffers([name])
        else:
            extra_outputs.append(name)
    return extra_outputs


def build_output_buffers(
    session: QAICInferenceSession,
    vocab_size: int,
    hidden_size: int,
    num_logits_to_keep: int,
    extra_output_names: List[str],
) -> dict:
    buffers = {}
    logits_dtype = get_binding_dtype(session, "logits")
    buffers["logits"] = np.zeros((1, num_logits_to_keep, vocab_size), dtype=logits_dtype)
    for name in extra_output_names:
        dtype = get_binding_dtype(session, name)
        buffers[name] = np.zeros((1, num_logits_to_keep, hidden_size), dtype=dtype)
    return buffers


def format_chat_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, Any]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("assistant:")
    return "\n".join(lines)


def build_stop_token_ids(tokenizer: AutoTokenizer) -> set:
    stop_ids = set()
    for token_id in tokenizer.all_special_ids:
        stop_ids.add(token_id)
    for token_id in (tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id):
        if token_id is not None and token_id in stop_ids:
            stop_ids.remove(token_id)
    vocab = tokenizer.get_vocab()
    for token in ("<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|endoftext|>"):
        if token in vocab:
            stop_ids.add(vocab[token])
    return stop_ids


def trim_on_stop(tokens: List[int], stop_ids: set) -> tuple[List[int], bool]:
    for idx, token_id in enumerate(tokens):
        if token_id in stop_ids:
            return tokens[: idx + 1], True
    return tokens, False


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


class NgramEngine:
    def __init__(
        self,
        target_model_name: str,
        num_speculative_tokens: int,
        prefill_seq_len: int,
        ctx_len: int,
        device_group: List[int],
        target_num_cores: int,
        max_tokens: int,
        max_ngram_size: int,
        lookahead_tokens: int,
        qaic_debug: bool,
    ) -> None:
        if lookahead_tokens and lookahead_tokens < num_speculative_tokens:
            raise ValueError("lookahead_tokens must be >= num_speculative_tokens when enabled.")
        self.target_model_name = target_model_name
        self.num_speculative_tokens = num_speculative_tokens
        self.prefill_seq_len = prefill_seq_len
        self.ctx_len = ctx_len
        self.device_group = device_group
        self.target_num_cores = target_num_cores
        self.max_tokens = max_tokens
        self.max_ngram_size = max_ngram_size
        self.lookahead_tokens = lookahead_tokens
        self.qaic_debug = qaic_debug
        self._lock = Lock()

        self._setup()

    def _setup(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.stop_token_ids = build_stop_token_ids(self.tokenizer)

        qaic_config = {"speculative_model_type": "target"}
        target_model = AutoModelForCausalLM.from_pretrained(self.target_model_name, qaic_config=qaic_config)

        core_candidates = [self.target_num_cores, 8, 6, 4, 2, 1]
        core_candidates = [c for i, c in enumerate(core_candidates) if c > 0 and c <= self.target_num_cores]
        core_candidates = [c for i, c in enumerate(core_candidates) if c not in core_candidates[:i]]

        self.target_session = None
        for num_cores in core_candidates:
            target_qpc = target_model.compile(
                num_devices=len(self.device_group),
                prefill_seq_len=self.prefill_seq_len,
                ctx_len=self.ctx_len,
                num_speculative_tokens=self.num_speculative_tokens,
                aic_enable_depth_first=True,
                num_cores=num_cores,
            )
            try:
                self.target_session = QAICInferenceSession(
                    target_qpc,
                    device_ids=self.device_group,
                    enable_debug_logs=self.qaic_debug,
                )
                break
            except RuntimeError as exc:
                msg = str(exc)
                retryable = any(
                    token in msg
                    for token in (
                        "Couldn't find a suitable device",
                        "Not enough NSPs",
                        "Failed to allocate NSP resources",
                        "Failed to Load program",
                    )
                )
                if not retryable or num_cores == core_candidates[-1]:
                    raise
                continue

        if self.target_session is None:
            raise RuntimeError("Unable to create target session with available core configurations.")

        self.target_session.skip_buffers(set([x for x in self.target_session.input_names if x.startswith("past_")]))
        self.target_session.skip_buffers(
            set([x for x in self.target_session.output_names if x.endswith("_RetainedState")])
        )
        self.extra_target_outputs = split_extra_outputs(self.target_session, ["logits"])
        model_config = getattr(target_model, "config", None) or getattr(target_model, "model", None).config
        self.vocab_size = getattr(model_config, "vocab_size", len(self.tokenizer))
        self.hidden_size = model_config.hidden_size

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> GenerateResult:
        max_tokens = max_tokens or self.max_tokens
        enable_lookahead = self.lookahead_tokens > 0
        lookahead_len = max(self.num_speculative_tokens, self.lookahead_tokens) if enable_lookahead else 0
        finish_reason = "length"

        with self._lock:
            inputs, prompt_ids, input_len = prepare_prefill_inputs(
                self.tokenizer, prompt, self.prefill_seq_len, self.ctx_len
            )
            max_tokens = min(max_tokens, max(0, self.ctx_len - input_len))
            if max_tokens <= 0:
                return GenerateResult(
                    text="",
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=0,
                    finish_reason="length",
                )

            prefill_buffers = build_output_buffers(
                self.target_session,
                self.vocab_size,
                self.hidden_size,
                num_logits_to_keep=1,
                extra_output_names=self.extra_target_outputs,
            )
            self.target_session.set_buffers(prefill_buffers)

            target_outs = run_prefill_on_target(self.target_session, inputs, prefill_seq_len=self.prefill_seq_len)
            if target_outs is None:
                raise RuntimeError("Prefill produced no outputs. Check prefill_seq_len and input padding.")
            target_logits = target_outs["logits"]
            current_token = target_logits.argmax(2).astype(np.int64)
            generated_ids = [current_token.item()]
            context_ids = prompt_ids + generated_ids
            position_cursor = len(prompt_ids) + 1

            verify_len = self.num_speculative_tokens + 1
            verify_buffers = build_output_buffers(
                self.target_session,
                self.vocab_size,
                self.hidden_size,
                num_logits_to_keep=verify_len,
                extra_output_names=self.extra_target_outputs,
            )
            self.target_session.set_buffers(verify_buffers)

            lookahead_buffer: List[int] = []
            while len(generated_ids) < max_tokens:
                verify_seed_token = current_token.copy()
                context_len = len(context_ids)
                effective_ngram = min(self.max_ngram_size, max(1, context_len))
                if enable_lookahead and len(lookahead_buffer) >= self.num_speculative_tokens:
                    draft_tokens = lookahead_buffer[: self.num_speculative_tokens]
                else:
                    spec_tokens, has_empty_tokens = find_candidate_pred_tokens(
                        np.array([context_ids], dtype=np.int64),
                        fill_tok=-1,
                        max_ngram_size=effective_ngram,
                        num_pred_tokens=lookahead_len if enable_lookahead else self.num_speculative_tokens,
                    )
                    if has_empty_tokens:
                        lookahead_buffer = []
                        draft_tokens = []
                    else:
                        spec_list = spec_tokens.tolist()
                        lookahead_buffer = spec_list if enable_lookahead else []
                        draft_tokens = spec_list[: self.num_speculative_tokens] if enable_lookahead else spec_list

                verify_inputs = {
                    "input_ids": np.zeros((1, verify_len), dtype=np.int64),
                    "position_ids": np.zeros((1, verify_len), dtype=np.int64),
                    "num_logits_to_keep": np.zeros((verify_len, 1), dtype=np.int64),
                }
                verify_inputs["input_ids"][0, 0] = verify_seed_token.item()
                valid_draft = np.array(draft_tokens, dtype=np.int64)
                max_draft = verify_len - 1
                verify_inputs["input_ids"][0, 1 : 1 + len(valid_draft)] = valid_draft[:max_draft]
                if len(valid_draft) < max_draft:
                    verify_inputs["input_ids"][0, 1 + len(valid_draft) :] = self.tokenizer.pad_token_id
                verify_inputs["position_ids"][0] = -1
                start_pos = position_cursor - 1
                verify_inputs["position_ids"][0, : 1 + len(valid_draft)] = np.arange(
                    start_pos, start_pos + 1 + len(valid_draft), dtype=np.int64
                )
                if "attention_mask" in self.target_session.input_names:
                    verify_inputs["attention_mask"] = np.zeros((1, verify_len), dtype=np.int64)
                    verify_inputs["attention_mask"][0, : 1 + len(valid_draft)] = 1

                verify_outs = self.target_session.run(filter_inputs(self.target_session, verify_inputs))
                target_logits = verify_outs["logits"]
                target_tokens = target_logits.argmax(-1)[0]
                draft_array = np.array(draft_tokens, dtype=np.int64)
                matches = draft_array == target_tokens[: len(draft_array)]
                accepted_draft = int(np.cumprod(matches).sum()) if matches.size else 0
                full_match = bool(draft_tokens) and accepted_draft == len(draft_tokens)
                bonus_stop = False
                if enable_lookahead and full_match and len(target_tokens) > len(draft_tokens):
                    bonus_token = target_tokens[len(draft_tokens)]
                    if bonus_token in self.stop_token_ids:
                        bonus_stop = True
                if enable_lookahead and full_match:
                    accepted_count = len(draft_tokens)
                else:
                    accepted_count = min(accepted_draft + 1, 1 + len(draft_tokens))
                remaining_quota = max_tokens - len(generated_ids)
                accepted_count = min(accepted_count, remaining_quota)
                verified_tokens = target_tokens[:accepted_count].tolist()
                verified_tokens, stop_on_eos = trim_on_stop(verified_tokens, self.stop_token_ids)
                stop_on_eos = stop_on_eos or bonus_stop

                if not verified_tokens:
                    break
                generated_ids.extend(verified_tokens)
                context_ids.extend(verified_tokens)
                position_cursor += len(verified_tokens)
                current_token = np.array([[verified_tokens[-1]]], dtype=np.int64)
                if enable_lookahead:
                    if full_match:
                        lookahead_buffer = lookahead_buffer[accepted_count:]
                    else:
                        lookahead_buffer = []
                if stop_on_eos:
                    finish_reason = "stop"
                    break

        assistant_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerateResult(
            text=assistant_text,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(generated_ids),
            finish_reason=finish_reason,
        )


class CompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = Field(default=None, ge=1, alias="max-tokens")
    stream: Optional[bool] = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1, alias="max-tokens")
    stream: Optional[bool] = False


def create_app(engine: NgramEngine) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.target_model_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "local",
                }
            ],
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest) -> Dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream not supported")
        prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
        choices = []
        total_prompt = 0
        total_completion = 0
        for idx, prompt in enumerate(prompts):
            result = engine.generate(prompt, max_tokens=req.max_tokens)
            choices.append(
                {
                    "text": result.text,
                    "index": idx,
                    "finish_reason": result.finish_reason,
                    "logprobs": None,
                }
            )
            total_prompt += result.prompt_tokens
            total_completion += result.completion_tokens
        return {
            "id": f"cmpl-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.target_model_name,
            "choices": choices,
            "usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
            },
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream not supported")
        messages = [msg.model_dump() for msg in req.messages]
        prompt = format_chat_prompt(engine.tokenizer, messages)
        result = engine.generate(prompt, max_tokens=req.max_tokens)
        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.target_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.prompt_tokens + result.completion_tokens,
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--device-group", type=str, default="0")
    parser.add_argument("--target-num-cores", type=int, default=14)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-ngram-size", type=int, default=3)
    parser.add_argument("--lookahead-tokens", type=int, default=0)
    parser.add_argument("--enable-lookahead", action="store_true")
    parser.add_argument("--qaic-debug", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    enable_lookahead = args.enable_lookahead or args.lookahead_tokens > 0
    lookahead_tokens = args.lookahead_tokens
    if enable_lookahead and lookahead_tokens <= 0:
        lookahead_tokens = args.num_speculative_tokens * 2
    if not enable_lookahead:
        lookahead_tokens = 0

    engine = NgramEngine(
        target_model_name=args.target_model_name,
        num_speculative_tokens=args.num_speculative_tokens,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        device_group=parse_device_group(args.device_group),
        target_num_cores=args.target_num_cores,
        max_tokens=args.max_tokens,
        max_ngram_size=args.max_ngram_size,
        lookahead_tokens=lookahead_tokens,
        qaic_debug=args.qaic_debug,
    )
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
