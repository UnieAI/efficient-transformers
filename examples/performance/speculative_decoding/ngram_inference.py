# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
from typing import List

import numpy as np
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


def ngram_spec_decode_inference(
    prompts,
    target_model_name,
    num_speculative_tokens=3,
    prefill_seq_len=128,
    ctx_len=512,
    device_group=None,
    target_num_cores=14,
    max_tokens=64,
    max_ngram_size=3,
    lookahead_tokens=0,
    qaic_debug=False,
):
    if lookahead_tokens and lookahead_tokens < num_speculative_tokens:
        raise ValueError("lookahead_tokens must be >= num_speculative_tokens when enabled.")
    device_group = device_group or [0]
    print(f"Loading Target Model: {target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    eos_token_id = tokenizer.eos_token_id

    qaic_config = {"speculative_model_type": "target"}
    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, qaic_config=qaic_config)

    core_candidates = [target_num_cores, 8, 6, 4, 2, 1]
    core_candidates = [c for i, c in enumerate(core_candidates) if c > 0 and c <= target_num_cores]
    core_candidates = [c for i, c in enumerate(core_candidates) if c not in core_candidates[:i]]

    target_session = None
    target_qpc = None
    selected_target_cores = None
    for num_cores in core_candidates:
        print("Compiling Target Model...")
        target_qpc = target_model.compile(
            num_devices=len(device_group),
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            num_speculative_tokens=num_speculative_tokens,
            aic_enable_depth_first=True,
            num_cores=num_cores,
        )
        print("Creating target session...")
        try:
            target_session = QAICInferenceSession(
                target_qpc,
                device_ids=device_group,
                enable_debug_logs=qaic_debug,
            )
            selected_target_cores = num_cores
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
            print(f"Target session failed with num_cores={num_cores}: {exc}")
            if not retryable or num_cores == core_candidates[-1]:
                raise
            print("Retrying with fewer target cores...")
            continue

    if target_session is None or target_qpc is None or selected_target_cores is None:
        raise RuntimeError("Unable to create target session with available core configurations.")
    print("Configuring target session buffers...")
    target_session.skip_buffers(set([x for x in target_session.input_names if x.startswith("past_")]))
    target_session.skip_buffers(set([x for x in target_session.output_names if x.endswith("_RetainedState")]))
    extra_target_outputs = split_extra_outputs(target_session, ["logits"])
    print("Target session ready.")

    model_config = getattr(target_model, "config", None) or getattr(target_model, "model", None).config
    vocab_size = getattr(model_config, "vocab_size", len(tokenizer))
    hidden_size = model_config.hidden_size

    prefill_buffers = build_output_buffers(
        target_session,
        vocab_size,
        hidden_size,
        num_logits_to_keep=1,
        extra_output_names=extra_target_outputs,
    )
    target_session.set_buffers(prefill_buffers)

    enable_lookahead = lookahead_tokens > 0
    lookahead_len = max(num_speculative_tokens, lookahead_tokens) if enable_lookahead else num_speculative_tokens

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
        current_token = target_logits.argmax(2).astype(np.int64)
        generated_ids = [current_token.item()]
        prompt_ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist()
        context_ids = prompt_ids + generated_ids
        position_cursor = len(prompt_ids) + 1
        total_draft_tokens = 0
        accepted_draft_tokens = 0
        verified_token_count = 0

        verify_len = num_speculative_tokens + 1
        verify_buffers = build_output_buffers(
            target_session,
            vocab_size,
            hidden_size,
            num_logits_to_keep=verify_len,
            extra_output_names=extra_target_outputs,
        )
        target_session.set_buffers(verify_buffers)

        lookahead_buffer: List[int] = []
        next_report_at = 50
        while len(generated_ids) < max_tokens:
            verify_seed_token = current_token.copy()
            context_len = len(context_ids)
            effective_ngram = min(max_ngram_size, max(1, context_len))
            if enable_lookahead and len(lookahead_buffer) >= num_speculative_tokens:
                draft_tokens = lookahead_buffer[:num_speculative_tokens]
            else:
                spec_tokens, has_empty_tokens = find_candidate_pred_tokens(
                    np.array([context_ids], dtype=np.int64),
                    fill_tok=-1,
                    max_ngram_size=effective_ngram,
                    num_pred_tokens=lookahead_len if enable_lookahead else num_speculative_tokens,
                )
                if has_empty_tokens:
                    lookahead_buffer = []
                    draft_tokens = []
                else:
                    spec_list = spec_tokens.tolist()
                    lookahead_buffer = spec_list if enable_lookahead else []
                    draft_tokens = spec_list[:num_speculative_tokens] if enable_lookahead else spec_list
            total_draft_tokens += len(draft_tokens)
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
                verify_inputs["input_ids"][0, 1 + len(valid_draft) :] = tokenizer.pad_token_id
            verify_inputs["position_ids"][0] = -1
            start_pos = position_cursor - 1
            verify_inputs["position_ids"][0, : 1 + len(valid_draft)] = np.arange(
                start_pos, start_pos + 1 + len(valid_draft), dtype=np.int64
            )
            if "attention_mask" in target_session.input_names:
                verify_inputs["attention_mask"] = np.zeros((1, verify_len), dtype=np.int64)
                verify_inputs["attention_mask"][0, : 1 + len(valid_draft)] = 1
            verify_outs = target_session.run(filter_inputs(target_session, verify_inputs))
            target_logits = verify_outs["logits"]
            target_tokens = target_logits.argmax(-1)[0]
            draft_array = np.array(draft_tokens, dtype=np.int64)
            matches = draft_array == target_tokens[: len(draft_array)]
            accepted_draft = int(np.cumprod(matches).sum()) if matches.size else 0
            full_match = bool(draft_tokens) and accepted_draft == len(draft_tokens)
            # Lookahead decoding: skip the bonus token on full match to keep the draft buffer aligned.
            if enable_lookahead and full_match:
                accepted_count = len(draft_tokens)
            else:
                accepted_count = min(accepted_draft + 1, 1 + len(draft_tokens))
            remaining_quota = max_tokens - len(generated_ids)
            accepted_count = min(accepted_count, remaining_quota)
            verified_tokens = target_tokens[:accepted_count].tolist()
            stop_on_eos = False
            if eos_token_id is not None and eos_token_id in verified_tokens:
                eos_index = verified_tokens.index(eos_token_id) + 1
                verified_tokens = verified_tokens[:eos_index]
                stop_on_eos = True
            accepted_draft_tokens += accepted_draft
            verified_token_count += len(verified_tokens)

            if not verified_tokens:
                break
            generated_ids.extend(verified_tokens)
            context_ids.extend(verified_tokens)
            position_cursor += len(verified_tokens)
            current_token = np.array([[verified_tokens[-1]]], dtype=np.int64)
            while len(generated_ids) >= next_report_at:
                if total_draft_tokens:
                    acceptance_rate = accepted_draft_tokens / total_draft_tokens
                    print(
                        f" Progress: generated={len(generated_ids)}, "
                        f"draft_tokens={total_draft_tokens}, "
                        f"accepted_draft_tokens={accepted_draft_tokens}, "
                        f"acceptance_rate={acceptance_rate:.2%}"
                    )
                else:
                    print(
                        f" Progress: generated={len(generated_ids)}, "
                        "draft_tokens=0, accepted_draft_tokens=0, acceptance_rate=0.00%"
                    )
                next_report_at += 50
            if enable_lookahead:
                if full_match:
                    lookahead_buffer = lookahead_buffer[accepted_count:]
                else:
                    lookahead_buffer = []
            if stop_on_eos:
                break
            if len(generated_ids) >= max_tokens:
                break

        full_output_ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist() + generated_ids
        full_text = tokenizer.decode(full_output_ids, skip_special_tokens=False)
        print(f" Full Text: {full_text!r}")
        print(
            " Total output tokens: "
            f"{len(full_output_ids)} (prompt={len(prompt_ids)}, generated={len(generated_ids)})"
        )
        if total_draft_tokens:
            acceptance_rate = accepted_draft_tokens / total_draft_tokens
            print(
                " Summary: "
                f"verified_tokens={verified_token_count}, "
                f"draft_tokens={total_draft_tokens}, "
                f"accepted_draft_tokens={accepted_draft_tokens}, "
                f"acceptance_rate={acceptance_rate:.2%}"
            )
        else:
            print(f" Summary: verified_tokens={verified_token_count}, draft_tokens=0")

    return "Done"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompt", "--prompts", dest="prompts", type=str, nargs="+", default=["Hello, my name is"])
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--prefill-seq-len", type=int, default=128)
    parser.add_argument("--ctx-len", type=int, default=512)
    parser.add_argument("--device-group", type=str, default="0")
    parser.add_argument("--target-num-cores", type=int, default=14)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-ngram-size", type=int, default=3)
    parser.add_argument(
        "--lookahead-tokens",
        type=int,
        default=0,
        help="Enable lookahead decoding by caching n-gram tokens (>= num-speculative-tokens)",
    )
    parser.add_argument("--qaic-debug", action="store_true")
    args = parser.parse_args()

    ngram_spec_decode_inference(
        args.prompts,
        args.target_model_name,
        num_speculative_tokens=args.num_speculative_tokens,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        device_group=parse_device_group(args.device_group),
        target_num_cores=args.target_num_cores,
        max_tokens=args.max_tokens,
        max_ngram_size=args.max_ngram_size,
        lookahead_tokens=args.lookahead_tokens,
        qaic_debug=args.qaic_debug,
    )
