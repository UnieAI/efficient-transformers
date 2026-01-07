# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import torch
import onnxruntime as ort
import time
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM as AutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession
from eagle_utils import load_eagle_head, compile_eagle_head

def eagle_spec_decode_inference(
    prompts,
    target_model_name,
    eagle_weights_path,
    num_speculative_tokens=3,
    prefill_seq_len=128,
    ctx_len=512,
    device_group=[0],
):
    # 1. Setup Target Model
    print(f"Loading Target Model: {target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # We assume the modified spd_transform_forward.py is active, 
    # so the model forward returns hidden_states.
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name, 
        qaic_config={"speculative_model_type": "target"} 
    )
    
    # 2. Compile Target Model
    # Important: To expose 'hidden_states' output from the compiled QPC, 
    # we might need to verify if QEfficient's exporter captures it automatically.
    # For this script, we assume the graph has an output named 'hidden_states'.
    print("Compiling Target Model...")
    target_qpc = target_model.compile(
        num_devices=len(device_group),
        prefill_seq_len=prefill_seq_len,
        ctx_len=ctx_len,
        aic_enable_depth_first=True
    )
    target_session = QAICInferenceSession(target_qpc, device_ids=device_group)
    
    # 3. Setup Eagle Head
    print("Setting up Eagle Head...")
    eagle_model = load_eagle_head(vocab_size=len(tokenizer), hidden_size=target_model.config.hidden_size, weights_path=eagle_weights_path)
    eagle_onnx_path = compile_eagle_head(eagle_model, batch_size=1)
    eagle_session = ort.InferenceSession(eagle_onnx_path)
    
    # 4. Inference Loop
    print(f"Starting Inference on {len(prompts)} prompts...")
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="np", padding=True)
        # Simple single-batch reasoning
        input_ids = inputs.input_ids
        
        # PREFILL Phase
        # Run Target to get p(x_1 | x_0) and H_1
        # Mocking input dict creation for QAIC
        target_inputs = {
            "input_ids": input_ids,
            "attention_mask": inputs.attention_mask,
            "position_ids": np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
        }
        
        # Note: Actual session run needs proper padding/batching handling as per QEfficient examples
        # Assuming simplified run for demonstration of logic
        # target_outs = target_session.run(target_inputs)
        # target_logits = target_outs['logits'] 
        # target_hidden = target_outs['hidden_states'] # This is the critical piece
        
        print(" [Simulating Prefill] -> Generated Token x_1, Feature H_1")
        
        # Taking last token and feature
        current_token = np.array([[1234]], dtype=np.int64) # Placeholder
        current_feature = np.random.randn(1, 1, 2048).astype(np.float32) # Placeholder
        
        # DRAFT Phase (Eagle Loop)
        draft_tokens = []
        for i in range(num_speculative_tokens):
            print(f"  [Draft Step {i+1}/{num_speculative_tokens}] Eagle Forward...")
            eagle_inputs = {
                "input_ids": current_token,
                "history_features": current_feature
            }
            eagle_outs = eagle_session.run(None, eagle_inputs)
            
            # Helper to get next token from logits
            logits = eagle_outs[0]
            next_token = np.argmax(logits, axis=-1)
            next_feature = eagle_outs[1]
            
            draft_tokens.append(next_token.item())
            
            # Update for next step
            current_token = next_token
            current_feature = next_feature
            
        print(f" Drafted Tokens: {draft_tokens}")
        
        # VERIFY Phase
        # Construct verification input (Prefix + Drafts)
        print(" [Verify] Running Target on Drafts...")
        # target_session.run(...)
        
    return "Done"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-name", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--prompts", type=str, nargs="+", default=["Hello, my name is"])
    parser.add_argument("--eagle-weights", type=str, default=None)
    args = parser.parse_args()
    
    eagle_spec_decode_inference(args.prompts, args.target_model_name, args.eagle_weights)
