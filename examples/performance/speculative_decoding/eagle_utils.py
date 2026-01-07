# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import torch
import os
from QEfficient.transformers.spd.eagle import EagleHead

def load_eagle_head(vocab_size=32000, hidden_size=2048, weights_path=None):
    """
    Load Eagle Head model.
    """
    class EagleConfig:
        def __init__(self, vocab_size, hidden_size):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.rms_norm_eps = 1e-5
            self.num_attention_heads = 4 # Default

    config = EagleConfig(vocab_size, hidden_size)
    model = EagleHead(config, hidden_size)
    
    if weights_path and os.path.exists(weights_path):
        print(f"Loading Eagle weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path))
    else:
        print("Initializing Eagle Head with random weights (Dummy Mode)")
        
    model.eval()
    return model

def compile_eagle_head(model, output_dir=".", batch_size=1):
    """
    Export Eagle Head to ONNX for compilation/execution.
    """
    hidden_size = model.hidden_size
    onnx_path = os.path.join(output_dir, "eagle_head.onnx")
    
    print(f"Exporting Eagle Head to {onnx_path}...")
    
    dummy_input_ids = torch.randint(0, 100, (batch_size, 1), dtype=torch.int64)
    dummy_features = torch.randn(batch_size, 1, hidden_size, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_features),
        onnx_path,
        input_names=["input_ids", "history_features"],
        output_names=["logits", "new_features"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "history_features": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "new_features": {0: "batch_size"}
        },
        opset_version=14
    )
    print("Export complete.")
    return onnx_path
