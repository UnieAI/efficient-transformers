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
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from eagle_utils import (
    EAGLE_INPUT_HIDDEN,
    EAGLE_INPUT_IDS,
    EAGLE_INPUT_PAST_KEY,
    EAGLE_INPUT_PAST_VALUE,
    EAGLE_LOOP_OUTPUT_HIDDEN,
    EAGLE_LOOP_OUTPUT_TOKENS,
    EAGLE_OUTPUT_HIDDEN,
    EAGLE_OUTPUT_LOGITS,
    EAGLE_OUTPUT_PRESENT_KEY,
    EAGLE_OUTPUT_PRESENT_VALUE,
    EagleFeatureCache,
    compile_eagle_head,
    compile_eagle_loop,
    compile_eagle_qpc,
    load_eagle_head,
    normalize_hidden_states,
    normalize_token_ids,
    select_last_hidden_states,
)


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


def run_prefill_on_target(session: QAICInferenceSession, inputs: dict, prefill_seq_len: int) -> dict:
    input_len = inputs["input_ids"].shape[1]
    num_chunks = input_len // prefill_seq_len
    cache_index = np.array([[0]], np.int64)
    outputs = None
    for _ in range(num_chunks):
        chunk_inputs = inputs.copy()
        start = cache_index[0, 0]
        end = start + prefill_seq_len
        chunk_inputs["input_ids"] = inputs["input_ids"][:, start:end]
        chunk_inputs["position_ids"] = inputs["position_ids"][:, start:end]
        chunk_inputs["attention_mask"] = inputs["attention_mask"][:, start:end]
        outputs = session.run(chunk_inputs)
        cache_index += prefill_seq_len
    return outputs


def get_binding_dtype(session: QAICInferenceSession, name: str) -> np.dtype:
    for binding in session.bindings:
        if binding.name == name:
            return session.aic_to_np_dtype_mapping[binding.type]
    return np.float32


def normalize_qaic_inputs(session: QAICInferenceSession, inputs: dict) -> dict:
    normalized = {}
    for name, value in inputs.items():
        dtype = get_binding_dtype(session, name)
        if value.dtype != dtype:
            value = value.astype(dtype, copy=False)
        normalized[name] = np.ascontiguousarray(value)
    return normalized


def _get_session_io_names(session):
    if hasattr(session, "get_inputs") and hasattr(session, "get_outputs"):
        input_names = {inp.name for inp in session.get_inputs()}
        output_names = [out.name for out in session.get_outputs()]
    else:
        input_names = set(session.input_names)
        output_names = list(session.output_names)
    return input_names, output_names


def resolve_eagle_names(session):
    input_names, output_names = _get_session_io_names(session)
    input_ids_name = EAGLE_INPUT_IDS if EAGLE_INPUT_IDS in input_names else "input_ids"
    hidden_input = EAGLE_INPUT_HIDDEN if EAGLE_INPUT_HIDDEN in input_names else "history_features"
    if hidden_input not in input_names:
        raise ValueError(f"Missing Eagle hidden input. Available inputs: {sorted(input_names)}")
    logits_output = EAGLE_OUTPUT_LOGITS if EAGLE_OUTPUT_LOGITS in output_names else output_names[0]
    hidden_output = EAGLE_OUTPUT_HIDDEN if EAGLE_OUTPUT_HIDDEN in output_names else "new_features"
    if hidden_output not in output_names:
        raise ValueError(f"Missing Eagle hidden output. Available outputs: {sorted(output_names)}")
    past_key = EAGLE_INPUT_PAST_KEY if EAGLE_INPUT_PAST_KEY in input_names else None
    past_value = EAGLE_INPUT_PAST_VALUE if EAGLE_INPUT_PAST_VALUE in input_names else None
    present_key = EAGLE_OUTPUT_PRESENT_KEY if EAGLE_OUTPUT_PRESENT_KEY in output_names else None
    present_value = EAGLE_OUTPUT_PRESENT_VALUE if EAGLE_OUTPUT_PRESENT_VALUE in output_names else None
    return input_ids_name, hidden_input, logits_output, hidden_output, past_key, past_value, present_key, present_value


def resolve_eagle_loop_names(session):
    _, output_names = _get_session_io_names(session)
    token_output = EAGLE_LOOP_OUTPUT_TOKENS if EAGLE_LOOP_OUTPUT_TOKENS in output_names else output_names[0]
    hidden_output = EAGLE_LOOP_OUTPUT_HIDDEN if EAGLE_LOOP_OUTPUT_HIDDEN in output_names else None
    if hidden_output is None or hidden_output not in output_names:
        raise ValueError(f"Missing Eagle loop hidden output. Available outputs: {sorted(output_names)}")
    present_key = EAGLE_OUTPUT_PRESENT_KEY if EAGLE_OUTPUT_PRESENT_KEY in output_names else None
    present_value = EAGLE_OUTPUT_PRESENT_VALUE if EAGLE_OUTPUT_PRESENT_VALUE in output_names else None
    return token_output, hidden_output, present_key, present_value


def eagle_spec_decode_inference(
    prompts,
    target_model_name,
    eagle_weights_path,
    num_speculative_tokens=3,
    prefill_seq_len=128,
    ctx_len=512,
    device_group=None,
    target_device_group=None,
    eagle_device_group=None,
    eagle_window=1,
    eagle_dtype="fp32",
    eagle_use_cache=False,
    eagle_cache_max_len=None,
    eagle_on_device_loop=False,
):
    device_group = device_group or [0]
    target_device_group = target_device_group or device_group
    eagle_device_group = eagle_device_group or device_group
    print(f"Loading Target Model: {target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    qaic_config = {"speculative_model_type": "target", "return_hidden_states": True}
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, qaic_config=qaic_config)

    print("Compiling Target Model...")
    target_qpc = target_model.compile(
        num_devices=len(target_device_group),
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        num_speculative_tokens=num_speculative_tokens,
        aic_enable_depth_first=True,
    )
    target_session = QAICInferenceSession(target_qpc, device_ids=target_device_group)
    target_session.skip_buffers(set([x for x in target_session.input_names if x.startswith("past_")]))
    target_session.skip_buffers(set([x for x in target_session.output_names if x.endswith("_RetainedState")]))

    if "hidden_states" not in target_session.output_names:
        raise ValueError(
            "hidden_states output not found. Re-export the target with "
            'qaic_config={"return_hidden_states": True}.'
        )

    print("Setting up Eagle Head...")
    model_config = getattr(target_model, "config", None) or getattr(target_model, "model", None).config
    num_attention_heads = getattr(model_config, "num_attention_heads", 4)
    eagle_model = load_eagle_head(
        vocab_size=len(tokenizer),
        hidden_size=model_config.hidden_size,
        num_attention_heads=num_attention_heads,
        weights_path=eagle_weights_path,
    )
    eagle_torch_dtype = torch.float16 if eagle_dtype == "fp16" else torch.float32
    eagle_np_dtype = np.float16 if eagle_dtype == "fp16" else np.float32
    use_cache = bool(eagle_use_cache)
    loop_steps = num_speculative_tokens
    if eagle_on_device_loop and use_cache and num_speculative_tokens > 1:
        loop_steps = num_speculative_tokens - 1
    if eagle_on_device_loop and not use_cache:
        print("On-device loop requires --eagle-use-cache; falling back to host loop.")
    use_device_loop = eagle_on_device_loop and use_cache and loop_steps > 0
    eagle_head_compile_dir = "eagle_head_qpc"
    eagle_loop_compile_dir = "eagle_loop_qpc"
    eagle_head_num_cores = 2
    eagle_loop_num_cores = 2

    eagle_session = None
    eagle_loop_session = None
    if use_device_loop:
        eagle_loop_onnx_path = compile_eagle_loop(
            eagle_model,
            num_steps=loop_steps,
            batch_size=1,
            dtype=eagle_torch_dtype,
            use_cache=use_cache,
            past_len=1,
        )
        eagle_loop_qpc_path = compile_eagle_qpc(
            eagle_loop_onnx_path,
            output_dir=eagle_loop_compile_dir,
            batch_size=1,
            seq_len=1,
            past_len=1,
            total_len=loop_steps + 1,
            num_cores=eagle_loop_num_cores,
            device_group=eagle_device_group,
        )
        eagle_loop_session = QAICInferenceSession(eagle_loop_qpc_path, device_ids=eagle_device_group)
        loop_token_output, loop_hidden_output, loop_present_key, loop_present_value = resolve_eagle_loop_names(
            eagle_loop_session
        )
        loop_output_names = eagle_loop_session.output_names

    use_step_session = not use_device_loop
    if use_cache and num_speculative_tokens <= 1:
        use_step_session = False
    if use_step_session:
        eagle_onnx_path = compile_eagle_head(
            eagle_model,
            batch_size=1,
            seq_len=1,
            dtype=eagle_torch_dtype,
            use_cache=use_cache,
            past_len=1,
        )
        if use_cache:
            eagle_custom_io = {
                EAGLE_INPUT_HIDDEN: "float16",
                EAGLE_INPUT_PAST_KEY: "float16",
                EAGLE_INPUT_PAST_VALUE: "float16",
                EAGLE_OUTPUT_HIDDEN: "float16",
                EAGLE_OUTPUT_LOGITS: "float16",
                EAGLE_OUTPUT_PRESENT_KEY: "float16",
                EAGLE_OUTPUT_PRESENT_VALUE: "float16",
            }
        else:
            eagle_custom_io = {
                EAGLE_INPUT_HIDDEN: "float16",
                EAGLE_OUTPUT_HIDDEN: "float16",
                EAGLE_OUTPUT_LOGITS: "float16",
            }

        eagle_qpc_path = compile_eagle_qpc(
            eagle_onnx_path,
            output_dir=eagle_head_compile_dir,
            batch_size=1,
            seq_len=1,
            past_len=1 if use_cache else None,
            total_len=1 if use_cache else None,
            num_cores=eagle_head_num_cores,
            device_group=eagle_device_group,
            custom_io=eagle_custom_io,
        )
        eagle_session = QAICInferenceSession(eagle_qpc_path, device_ids=eagle_device_group)
        (
            eagle_input_ids,
            eagle_hidden_input,
            eagle_logits_output,
            eagle_hidden_output,
            eagle_past_key,
            eagle_past_value,
            eagle_present_key,
            eagle_present_value,
        ) = resolve_eagle_names(eagle_session)
        if use_cache:
            if not all([eagle_past_key, eagle_past_value, eagle_present_key, eagle_present_value]):
                raise ValueError("Eagle cache inputs/outputs not found. Re-export with use_cache=True.")
        eagle_output_names = eagle_session.output_names

    print(f"Starting Inference on {len(prompts)} prompts...")
    vocab_size = len(tokenizer)
    hidden_dtype = get_binding_dtype(target_session, "hidden_states")
    logits_dtype = get_binding_dtype(target_session, "logits")
    target_session.set_buffers(
        {
            "logits": np.zeros((1, 1, vocab_size), dtype=logits_dtype),
            "hidden_states": np.zeros((1, 1, model_config.hidden_size), dtype=hidden_dtype),
        }
    )

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        input_len = tokenizer(prompt, return_tensors="np", padding=True).input_ids.shape[1]
        input_len_padded = get_padded_input_len(input_len, prefill_seq_len, ctx_len)
        inputs = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=input_len_padded)
        position_ids = np.where(inputs["attention_mask"], np.arange(input_len_padded), -1)
        inputs["position_ids"] = position_ids
        inputs["num_logits_to_keep"] = np.zeros((1, 1), dtype=np.int64)

        target_outs = run_prefill_on_target(target_session, inputs, prefill_seq_len=prefill_seq_len)
        if target_outs is None:
            raise RuntimeError("Prefill produced no outputs. Check prefill_seq_len and input padding.")
        target_logits = target_outs["logits"]
        target_hidden = target_outs["hidden_states"]

        current_token = target_logits.argmax(2).astype(np.int64)
        current_feature = select_last_hidden_states(target_hidden)

        cache = EagleFeatureCache(max_len=eagle_window)
        cache.append(current_token, current_feature, dtype=eagle_np_dtype)
        past_key = None
        past_value = None
        draft_tokens = []
        remaining_steps = num_speculative_tokens
        if use_cache and remaining_steps > 0:
            empty_past = np.zeros(
                (1, eagle_model.num_heads, 1, eagle_model.head_dim), dtype=eagle_np_dtype
            )
            if use_device_loop:
                past_key = empty_past
                past_value = empty_past
            else:
                if eagle_session is None:
                    raise RuntimeError("Eagle QPC session is not available for host-side cache warmup.")
                step_tokens = normalize_token_ids(current_token)
                step_features = normalize_hidden_states(current_feature, dtype=eagle_np_dtype)
                eagle_inputs = {
                    eagle_input_ids: step_tokens,
                    eagle_hidden_input: step_features,
                    eagle_past_key: empty_past,
                    eagle_past_value: empty_past,
                }
                eagle_outs = eagle_session.run(None, eagle_inputs)
                logits = eagle_outs[eagle_logits_index]
                next_hidden = eagle_outs[eagle_hidden_index]
                next_token = np.argmax(logits[:, -1:, :], axis=-1).astype(np.int64)
                next_feature = next_hidden[:, -1:, :]
                draft_tokens = [next_token.item()]
                past_key = eagle_outs[present_key_index]
                past_value = eagle_outs[present_value_index]
                if eagle_cache_max_len:
                    past_key = past_key[:, :, -eagle_cache_max_len:, :]
                    past_value = past_value[:, :, -eagle_cache_max_len:, :]
                current_token = next_token
                current_feature = next_feature
                cache.append(next_token, next_feature, dtype=eagle_np_dtype)
                remaining_steps -= 1

        if remaining_steps > 0 and use_device_loop:
            step_tokens = normalize_token_ids(current_token)
            step_features = normalize_hidden_states(current_feature, dtype=eagle_np_dtype)
            loop_inputs = {EAGLE_INPUT_IDS: step_tokens, EAGLE_INPUT_HIDDEN: step_features}
            if use_cache:
                loop_inputs[EAGLE_INPUT_PAST_KEY] = past_key
                loop_inputs[EAGLE_INPUT_PAST_VALUE] = past_value
            loop_outs = eagle_loop_session.run(normalize_qaic_inputs(eagle_loop_session, loop_inputs))
            loop_tokens = loop_outs[loop_token_output]
            loop_hidden = loop_outs[loop_hidden_output]
            if loop_tokens.ndim == 1:
                loop_tokens = loop_tokens.reshape(1, -1)
            draft_tokens.extend(loop_tokens[0].tolist())
            current_token = loop_tokens[:, -1:].astype(np.int64)
            current_feature = loop_hidden[:, -1:, :]
            if use_cache and loop_present_key and loop_present_value:
                past_key = loop_outs[loop_present_key]
                past_value = loop_outs[loop_present_value]
                if eagle_cache_max_len:
                    past_key = past_key[:, :, -eagle_cache_max_len:, :]
                    past_value = past_value[:, :, -eagle_cache_max_len:, :]
        elif remaining_steps > 0:
            if eagle_session is None:
                raise RuntimeError("Eagle ONNX session is not available for host-side loop.")
            step_offset = num_speculative_tokens - remaining_steps
            for i in range(remaining_steps):
                step_idx = step_offset + i + 1
                print(f"  [Draft Step {step_idx}/{num_speculative_tokens}] Eagle Forward...")
                if use_cache:
                    step_tokens = normalize_token_ids(current_token)
                    step_features = normalize_hidden_states(current_feature, dtype=eagle_np_dtype)
                    eagle_inputs = {
                        eagle_input_ids: step_tokens,
                        eagle_hidden_input: step_features,
                        eagle_past_key: past_key,
                        eagle_past_value: past_value,
                    }
                else:
                    window_tokens, window_features = cache.window()
                    window_tokens = normalize_token_ids(window_tokens)
                    window_features = normalize_hidden_states(window_features, dtype=eagle_np_dtype)
                    eagle_inputs = {eagle_input_ids: window_tokens, eagle_hidden_input: window_features}
                eagle_outs = eagle_session.run(normalize_qaic_inputs(eagle_session, eagle_inputs))
                logits = eagle_outs[eagle_logits_output]
                next_hidden = eagle_outs[eagle_hidden_output]
                if use_cache:
                    past_key = eagle_outs[eagle_present_key]
                    past_value = eagle_outs[eagle_present_value]
                    if eagle_cache_max_len:
                        past_key = past_key[:, :, -eagle_cache_max_len:, :]
                        past_value = past_value[:, :, -eagle_cache_max_len:, :]
                next_token = np.argmax(logits[:, -1:, :], axis=-1).astype(np.int64)
                next_feature = next_hidden[:, -1:, :]
                draft_tokens.append(next_token.item())
                cache.append(next_token, next_feature, dtype=eagle_np_dtype)
                current_token = next_token
                current_feature = next_feature

        print(f" Drafted Tokens: {draft_tokens}")
        print(" [Verify] Target verification loop not implemented yet.")

    return "Done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompts", type=str, nargs="+", default=["Hello, my name is"])
    parser.add_argument("--eagle-weights", type=str, default=None)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--device-group", type=str, default="0")
    parser.add_argument("--target-device-group", type=str, default=None)
    parser.add_argument("--eagle-device-group", type=str, default=None)
    parser.add_argument("--eagle-window", type=int, default=1)
    parser.add_argument("--eagle-dtype", type=str, choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--eagle-use-cache", action="store_true")
    parser.add_argument("--eagle-cache-max-len", type=int, default=None)
    parser.add_argument("--eagle-on-device-loop", action="store_true")
    args = parser.parse_args()

    eagle_spec_decode_inference(
        args.prompts,
        args.target_model_name,
        args.eagle_weights,
        num_speculative_tokens=args.num_speculative_tokens,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        device_group=parse_device_group(args.device_group),
        target_device_group=parse_device_group(args.target_device_group)
        if args.target_device_group
        else None,
        eagle_device_group=parse_device_group(args.eagle_device_group) if args.eagle_device_group else None,
        eagle_window=args.eagle_window,
        eagle_dtype=args.eagle_dtype,
        eagle_use_cache=args.eagle_use_cache,
        eagle_cache_max_len=args.eagle_cache_max_len,
        eagle_on_device_loop=args.eagle_on_device_loop,
    )
