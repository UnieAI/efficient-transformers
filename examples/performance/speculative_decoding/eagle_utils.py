# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from QEfficient.compile.compile_helper import compile_kv_model_on_cloud_ai_100
from QEfficient.transformers.spd.eagle import EagleConfig, EagleDraftLoop, EagleHead

EAGLE_INPUT_IDS = "input_ids"
EAGLE_INPUT_HIDDEN = "history_hidden_states"
EAGLE_INPUT_PAST_KEY = "past_key"
EAGLE_INPUT_PAST_VALUE = "past_value"
EAGLE_LOOP_OUTPUT_TOKENS = "draft_tokens"
EAGLE_LOOP_OUTPUT_HIDDEN = "draft_hidden_states"
EAGLE_OUTPUT_LOGITS = "logits"
EAGLE_OUTPUT_HIDDEN = "hidden_states"
EAGLE_OUTPUT_PRESENT_KEY = "present_key"
EAGLE_OUTPUT_PRESENT_VALUE = "present_value"


def load_eagle_head(vocab_size=32000, hidden_size=2048, num_attention_heads=4, weights_path=None):
    """
    Load Eagle Head model.
    """
    config = EagleConfig(vocab_size, hidden_size, num_attention_heads)
    model = EagleHead(config, hidden_size)

    if weights_path and os.path.exists(weights_path):
        print(f"Loading Eagle weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        print("Initializing Eagle Head with random weights (Dummy Mode)")

    model.eval()
    return model


def compile_eagle_head(
    model,
    output_dir=".",
    batch_size=1,
    seq_len=1,
    dtype: torch.dtype = torch.float32,
    use_cache: bool = False,
    past_len: int = 1,
):
    """
    Export Eagle Head to ONNX for compilation/execution.
    """
    hidden_size = model.hidden_size
    num_heads = model.num_heads
    head_dim = model.head_dim
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = str(output_path / "eagle_head.onnx")

    print(f"Exporting Eagle Head to {onnx_path}...")

    model = model.to(dtype)
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), dtype=torch.int64)
    dummy_features = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    if use_cache:
        dummy_past_key = torch.zeros(batch_size, num_heads, past_len, head_dim, dtype=dtype)
        dummy_past_value = torch.zeros(batch_size, num_heads, past_len, head_dim, dtype=dtype)
        export_args = (dummy_input_ids, dummy_features, dummy_past_key, dummy_past_value)
        input_names = [EAGLE_INPUT_IDS, EAGLE_INPUT_HIDDEN, EAGLE_INPUT_PAST_KEY, EAGLE_INPUT_PAST_VALUE]
        output_names = [
            EAGLE_OUTPUT_LOGITS,
            EAGLE_OUTPUT_HIDDEN,
            EAGLE_OUTPUT_PRESENT_KEY,
            EAGLE_OUTPUT_PRESENT_VALUE,
        ]
        dynamic_axes = {
            EAGLE_INPUT_IDS: {0: "batch_size", 1: "seq_len"},
            EAGLE_INPUT_HIDDEN: {0: "batch_size", 1: "seq_len"},
            EAGLE_INPUT_PAST_KEY: {0: "batch_size", 2: "past_len"},
            EAGLE_INPUT_PAST_VALUE: {0: "batch_size", 2: "past_len"},
            EAGLE_OUTPUT_LOGITS: {0: "batch_size", 1: "seq_len"},
            EAGLE_OUTPUT_HIDDEN: {0: "batch_size", 1: "seq_len"},
            EAGLE_OUTPUT_PRESENT_KEY: {0: "batch_size", 2: "total_len"},
            EAGLE_OUTPUT_PRESENT_VALUE: {0: "batch_size", 2: "total_len"},
        }
    else:
        export_args = (dummy_input_ids, dummy_features)
        input_names = [EAGLE_INPUT_IDS, EAGLE_INPUT_HIDDEN]
        output_names = [EAGLE_OUTPUT_LOGITS, EAGLE_OUTPUT_HIDDEN]
        dynamic_axes = {
            EAGLE_INPUT_IDS: {0: "batch_size", 1: "seq_len"},
            EAGLE_INPUT_HIDDEN: {0: "batch_size", 1: "seq_len"},
            EAGLE_OUTPUT_LOGITS: {0: "batch_size", 1: "seq_len"},
            EAGLE_OUTPUT_HIDDEN: {0: "batch_size", 1: "seq_len"},
        }

    torch.onnx.export(
        model,
        export_args,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    print("Export complete.")
    return onnx_path



def compile_eagle_loop(
    model,
    num_steps: int,
    output_dir=".",
    batch_size=1,
    dtype: torch.dtype = torch.float32,
    use_cache: bool = False,
    past_len: int = 1,
):
    """
    Export Eagle Draft Loop to ONNX to remove Python-side iteration.
    """
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    hidden_size = model.hidden_size
    num_heads = model.num_heads
    head_dim = model.head_dim
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = str(output_path / f"eagle_loop_{num_steps}.onnx")

    loop_model = EagleDraftLoop(model, num_steps=num_steps, use_cache=use_cache).to(dtype)
    dummy_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, 1), dtype=torch.int64)
    dummy_features = torch.randn(batch_size, 1, hidden_size, dtype=dtype)

    if use_cache:
        dummy_past_key = torch.zeros(batch_size, num_heads, past_len, head_dim, dtype=dtype)
        dummy_past_value = torch.zeros(batch_size, num_heads, past_len, head_dim, dtype=dtype)
        export_args = (dummy_input_ids, dummy_features, dummy_past_key, dummy_past_value)
        input_names = [EAGLE_INPUT_IDS, EAGLE_INPUT_HIDDEN, EAGLE_INPUT_PAST_KEY, EAGLE_INPUT_PAST_VALUE]
        output_names = [
            EAGLE_LOOP_OUTPUT_TOKENS,
            EAGLE_LOOP_OUTPUT_HIDDEN,
            EAGLE_OUTPUT_PRESENT_KEY,
            EAGLE_OUTPUT_PRESENT_VALUE,
        ]
        dynamic_axes = {
            EAGLE_INPUT_IDS: {0: "batch_size"},
            EAGLE_INPUT_HIDDEN: {0: "batch_size"},
            EAGLE_INPUT_PAST_KEY: {0: "batch_size", 2: "past_len"},
            EAGLE_INPUT_PAST_VALUE: {0: "batch_size", 2: "past_len"},
            EAGLE_LOOP_OUTPUT_TOKENS: {0: "batch_size"},
            EAGLE_LOOP_OUTPUT_HIDDEN: {0: "batch_size"},
            EAGLE_OUTPUT_PRESENT_KEY: {0: "batch_size", 2: "total_len"},
            EAGLE_OUTPUT_PRESENT_VALUE: {0: "batch_size", 2: "total_len"},
        }
    else:
        export_args = (dummy_input_ids, dummy_features)
        input_names = [EAGLE_INPUT_IDS, EAGLE_INPUT_HIDDEN]
        output_names = [EAGLE_LOOP_OUTPUT_TOKENS, EAGLE_LOOP_OUTPUT_HIDDEN]
        dynamic_axes = {
            EAGLE_INPUT_IDS: {0: "batch_size"},
            EAGLE_INPUT_HIDDEN: {0: "batch_size"},
            EAGLE_LOOP_OUTPUT_TOKENS: {0: "batch_size"},
            EAGLE_LOOP_OUTPUT_HIDDEN: {0: "batch_size"},
        }

    # ... existing code ...
    torch.onnx.export(
        loop_model,
        export_args,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
    )
    return onnx_path

def compile_eagle_qpc(
    onnx_path: str,
    output_dir: str,
    *,
    batch_size: int = 1,
    seq_len: int = 1,
    past_len: int = 1,
    total_len: int = 1,
    num_cores: int = 2,
    mxfp6: bool = True,
    aic_enable_depth_first: bool = True,
    allow_mxint8_mdp_io: bool = False,
    device_group: Optional[list] = None,
    custom_io: Optional[dict] = None,
):
    """
    Compile Eagle ONNX to QAIC QPC.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    qaic_exec = Path("/opt/qti-aic/exec/qaic-exec")
    if not qaic_exec.exists():
        raise FileNotFoundError("qaic-exec not found at /opt/qti-aic/exec/qaic-exec.")
    specializations_json = output_path / "specializations.json"
    specialization = {"batch_size": str(batch_size)}
    if seq_len is not None:
        specialization["seq_len"] = str(seq_len)
    if past_len is not None:
        specialization["past_len"] = str(past_len)
    if total_len is not None:
        specialization["total_len"] = str(total_len)
    specializations = {"specializations": [specialization]}
    with open(specializations_json, "w") as fp:
        json.dump(specializations, fp, indent=4)

    if custom_io is None:
        custom_io = {
            EAGLE_INPUT_HIDDEN: "float16",
            EAGLE_INPUT_PAST_KEY: "float16",
            EAGLE_INPUT_PAST_VALUE: "float16",
            EAGLE_LOOP_OUTPUT_HIDDEN: "float16",
            EAGLE_OUTPUT_PRESENT_KEY: "float16",
            EAGLE_OUTPUT_PRESENT_VALUE: "float16",
        }
    custom_io_yaml = output_path / "custom_io.yaml"
    with open(custom_io_yaml, "w") as fp:
        for io_name, dtype in custom_io.items():
            fp.write(f" - IOName: {io_name}\n   Precision: {dtype}\n\n")

    _, qpc_path = compile_kv_model_on_cloud_ai_100(
        onnx_path=onnx_path,
        specializations_json=str(specializations_json),
        num_cores=num_cores,
        base_path=str(output_path),
        mxfp6=mxfp6,
        custom_io_path=str(custom_io_yaml),
        aic_enable_depth_first=aic_enable_depth_first,
        allow_mxint8_mdp_io=allow_mxint8_mdp_io,
        device_group=device_group,
    )
    return qpc_path


def normalize_token_ids(token_ids: np.ndarray) -> np.ndarray:
    if token_ids.ndim == 1:
        token_ids = token_ids.reshape(-1, 1)
    if token_ids.ndim != 2:
        raise ValueError(f"token_ids must have shape [batch, seq_len], got {token_ids.shape}")
    return token_ids.astype(np.int64, copy=False)


def normalize_hidden_states(hidden_states: np.ndarray, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hidden_states.ndim == 2:
        hidden_states = hidden_states[:, None, :]
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must have shape [batch, seq_len, hidden], got {hidden_states.shape}")
    if dtype is not None:
        hidden_states = hidden_states.astype(dtype, copy=False)
    return hidden_states


def select_last_hidden_states(hidden_states: np.ndarray) -> np.ndarray:
    hidden_states = normalize_hidden_states(hidden_states)
    return hidden_states[:, -1:, :]



@dataclass
class EagleFeatureCache:
    max_len: int = 1

    def __post_init__(self):
        self._token_cache = []
        self._feature_cache = []

    def append(self, token_ids: np.ndarray, hidden_states: np.ndarray, dtype: Optional[np.dtype] = None):
        token_ids = normalize_token_ids(token_ids)
        hidden_states = normalize_hidden_states(hidden_states, dtype=dtype)
        self._token_cache.append(token_ids)
        self._feature_cache.append(hidden_states)
        if len(self._token_cache) > self.max_len:
            self._token_cache.pop(0)
            self._feature_cache.pop(0)

    def window(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._token_cache or not self._feature_cache:
            raise ValueError("feature cache is empty")
        return np.concatenate(self._token_cache, axis=1), np.concatenate(self._feature_cache, axis=1)
