# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import List

import numpy as np
import torch

from QEfficient.generation.cloud_infer import QAICInferenceSession
from eagle_utils import (
    EAGLE_INPUT_HIDDEN,
    EAGLE_INPUT_IDS,
    EAGLE_INPUT_PAST_KEY,
    EAGLE_INPUT_PAST_VALUE,
    EAGLE_LOOP_OUTPUT_HIDDEN,
    EAGLE_LOOP_OUTPUT_TOKENS,
    EAGLE_OUTPUT_PRESENT_KEY,
    EAGLE_OUTPUT_PRESENT_VALUE,
    compile_eagle_loop,
    compile_eagle_qpc,
    load_eagle_head,
)


def parse_device_group(value: str) -> List[int]:
    if not value:
        return [0]
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="QAIC smoke check for Eagle loop I/O shapes.")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--past-len", type=int, default=1)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="eagle_loop_qpc")
    parser.add_argument("--num-cores", type=int, default=2)
    parser.add_argument("--device-group", type=str, default="0")
    args = parser.parse_args()

    if args.hidden_size % args.num_heads != 0:
        raise ValueError("hidden-size must be divisible by num-heads.")
    if args.num_steps < 1:
        raise ValueError("num-steps must be >= 1")

    device_group = parse_device_group(args.device_group)
    head_dim = args.hidden_size // args.num_heads
    total_len = args.past_len + args.num_steps

    eagle_model = load_eagle_head(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        weights_path=args.weights,
    ).to(torch.float16)

    onnx_path = compile_eagle_loop(
        eagle_model,
        num_steps=args.num_steps,
        output_dir=args.output_dir,
        batch_size=1,
        dtype=torch.float16,
        use_cache=True,
        past_len=args.past_len,
    )
    try:
        qpc_path = compile_eagle_qpc(
            onnx_path,
            output_dir=args.output_dir,
            batch_size=1,
            seq_len=1,
            past_len=args.past_len,
            total_len=total_len,
            num_cores=args.num_cores,
            device_group=device_group,
        )
    except FileNotFoundError as exc:
        print(f"SKIP: qaic-exec not available ({exc})")
        return

    try:
        session = QAICInferenceSession(qpc_path, device_ids=device_group)
    except ImportError as exc:
        print(f"SKIP: QAIC runtime not available ({exc})")
        return

    inputs = {
        EAGLE_INPUT_IDS: np.random.randint(0, args.vocab_size, size=(1, 1), dtype=np.int64),
        EAGLE_INPUT_HIDDEN: np.random.randn(1, 1, args.hidden_size).astype(np.float16),
        EAGLE_INPUT_PAST_KEY: np.zeros((1, args.num_heads, args.past_len, head_dim), dtype=np.float16),
        EAGLE_INPUT_PAST_VALUE: np.zeros((1, args.num_heads, args.past_len, head_dim), dtype=np.float16),
    }

    outputs = session.run(inputs)
    required_outputs = {
        EAGLE_LOOP_OUTPUT_TOKENS,
        EAGLE_LOOP_OUTPUT_HIDDEN,
        EAGLE_OUTPUT_PRESENT_KEY,
        EAGLE_OUTPUT_PRESENT_VALUE,
    }
    missing = [name for name in required_outputs if name not in outputs]
    if missing:
        raise KeyError(f"Missing outputs from QAIC session: {missing}")
    tokens = outputs[EAGLE_LOOP_OUTPUT_TOKENS]
    hidden = outputs[EAGLE_LOOP_OUTPUT_HIDDEN]
    present_key = outputs[EAGLE_OUTPUT_PRESENT_KEY]
    present_value = outputs[EAGLE_OUTPUT_PRESENT_VALUE]

    expected_tokens = (1, args.num_steps)
    expected_hidden = (1, args.num_steps, args.hidden_size)
    expected_present = (1, args.num_heads, total_len, head_dim)

    if tokens.shape != expected_tokens:
        raise AssertionError(f"tokens shape {tokens.shape} != {expected_tokens}")
    if hidden.shape != expected_hidden:
        raise AssertionError(f"hidden shape {hidden.shape} != {expected_hidden}")
    if present_key.shape != expected_present:
        raise AssertionError(f"present_key shape {present_key.shape} != {expected_present}")
    if present_value.shape != expected_present:
        raise AssertionError(f"present_value shape {present_value.shape} != {expected_present}")

    print("PASS: QAIC Eagle loop I/O shapes OK.")
    print(f" tokens: {tokens.shape} dtype={tokens.dtype}")
    print(f" hidden: {hidden.shape} dtype={hidden.dtype}")
    print(f" present_key: {present_key.shape} dtype={present_key.dtype}")
    print(f" present_value: {present_value.shape} dtype={present_value.dtype}")


if __name__ == "__main__":
    main()
