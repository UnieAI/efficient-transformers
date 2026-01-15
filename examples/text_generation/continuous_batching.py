# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import Dict, List, Optional

import numpy as np

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM


class ContinuousBatchingEngine:
    def __init__(
        self,
        model_name: str,
        prefill_seq_len: int,
        ctx_len: int,
        full_batch_size: int,
        generation_len: int,
        num_cores: int,
        device_group: Optional[List[int]] = None,
    ):
        self.model_name = model_name
        self.prefill_seq_len = prefill_seq_len
        self.ctx_len = ctx_len
        self.full_batch_size = full_batch_size
        self.default_generation_len = generation_len
        self.num_cores = num_cores
        self.device_group = device_group

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = QEFFAutoModelForCausalLM.from_pretrained(model_name, continuous_batching=True)
        self.qpc_path = self.model.compile(
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            full_batch_size=full_batch_size,
            num_cores=num_cores,
            num_devices=(1 if device_group is None else len(device_group)),
        )

    def generate_batch(
        self,
        prompts: List[str],
        generation_len: Optional[int] = None,
        max_tokens_per_prompt: Optional[List[int]] = None,
        prompt_token_counts: Optional[List[int]] = None,
    ) -> List[Dict[str, object]]:
        if not prompts:
            return []

        if generation_len is None:
            generation_len = self.default_generation_len

        if max_tokens_per_prompt is None:
            max_tokens_per_prompt = [generation_len] * len(prompts)

        exec_info = self.model.generate(
            tokenizer=self.tokenizer,
            prompts=prompts,
            device_id=self.device_group,
            generation_len=generation_len,
        )

        batch_ids = self._normalize_generated_ids(exec_info.generated_ids, len(prompts))
        results = []
        for idx, prompt in enumerate(prompts):
            prompt_tokens = (
                prompt_token_counts[idx]
                if prompt_token_counts is not None
                else len(self.tokenizer.encode(prompt))
            )
            token_ids = self._trim_token_ids(batch_ids[idx], max_tokens_per_prompt[idx])
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            results.append(
                {
                    "text": text,
                    "token_ids": token_ids,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(token_ids),
                }
            )
        return results

    def _normalize_generated_ids(
        self, generated_ids: object, batch_size: int
    ) -> List[np.ndarray]:
        if isinstance(generated_ids, np.ndarray):
            if generated_ids.ndim == 1:
                return [generated_ids]
            return [generated_ids[i] for i in range(min(batch_size, generated_ids.shape[0]))]
        if isinstance(generated_ids, list):
            if (
                len(generated_ids) == 1
                and isinstance(generated_ids[0], np.ndarray)
                and generated_ids[0].ndim == 2
            ):
                return [
                    generated_ids[0][i]
                    for i in range(min(batch_size, generated_ids[0].shape[0]))
                ]
            return [np.asarray(generated_ids[i]) for i in range(min(batch_size, len(generated_ids)))]
        return [np.asarray(generated_ids)]

    def _trim_token_ids(self, token_ids: np.ndarray, max_tokens: int) -> List[int]:
        trimmed: List[int] = []
        for token_id in token_ids.tolist():
            if token_id < 0 or token_id == self.tokenizer.pad_token_id:
                break
            if token_id == self.tokenizer.eos_token_id:
                break
            trimmed.append(int(token_id))
            if len(trimmed) >= max_tokens:
                break
        return trimmed


def main():
    parser = argparse.ArgumentParser(description="Continuous batching inference")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2-1.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument(
        "--prompts",
        type=str,
        default="Hello! How can I help?|Hi there! What’s up?|Hey! Need assistance?|Welcome! How can I support you today?",
        help="Pipe-separated prompts for batch processing",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=128, help="Prefill sequence length")
    parser.add_argument("--ctx-len", type=int, default=512, help="Context length")
    parser.add_argument("--full-batch-size", type=int, default=4, help="Full batch size for continuous batching")
    parser.add_argument("--generation-len", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--num-cores", type=int, default=16, help="Number of cores")
    parser.add_argument(
        "--device-group",
        type=lambda device_ids: [int(x) for x in device_ids.strip("[]").split(",")],
        default=None,
        help="Device IDs (comma-separated) e.g. [0,1]",
    )
    args = parser.parse_args()

    # Parse prompts
    prompt_list = args.prompts.split("|")
    print(f"Processing {len(prompt_list)} prompts with continuous batching")

    engine = ContinuousBatchingEngine(
        model_name=args.model_name,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        full_batch_size=args.full_batch_size,
        generation_len=args.generation_len,
        num_cores=args.num_cores,
        device_group=args.device_group,
    )
    print(f"Model compiled to: {engine.qpc_path}")

    results = engine.generate_batch(prompt_list, generation_len=args.generation_len)

    # Display results
    print("\n" + "=" * 80)
    for i, (prompt, result) in enumerate(zip(prompt_list, results)):
        print(f"\nPrompt {i + 1}: {prompt}")
        print(f"Generated: {result['text']}")
        print("-" * 80)


if __name__ == "__main__":
    main()
