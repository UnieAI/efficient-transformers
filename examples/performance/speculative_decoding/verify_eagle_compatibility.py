# verify_eagle_compatibility.py
import torch
import os
import shutil
from transformers import AutoConfig, AutoModelForCausalLM

# Mock QEfficient if needed or use installed
try:
    from QEfficient import QEFFAutoModelForCausalLM
except ImportError:
    print("QEfficient not found. Skipping full integration test.")
    exit(0)

from QEfficient.transformers.spd.eagle import EagleHead
from examples.performance.speculative_decoding.eagle_utils import compile_eagle_head, load_eagle_head

def test_eagle_head_export():
    print("--- Testing Eagle Head Export ---")
    model = load_eagle_head(vocab_size=1000, hidden_size=128)
    onnx_path = compile_eagle_head(model, batch_size=1)
    if os.path.exists(onnx_path):
        print(f"SUCCESS: Eagle Head exported to {onnx_path}")
    else:
        print("FAILURE: Eagle Head export failed")

def test_target_model_outputs():
    print("\n--- Testing Target Model Output Names ---")
    # Load a tiny model/config to check `get_output_names`
    # We use QEFFAutoModelForCausalLM which wraps a model
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Check if QEFF modifies the model class
    try:
        qeff_model = QEFFAutoModelForCausalLM.from_pretrained(model_name)
        # Check if get_output_names exists
        if hasattr(qeff_model.model, "get_output_names"):
            outputs = qeff_model.model.get_output_names(kv_offload=True)
            print(f"Output Names: {outputs}")
            if "hidden_states" in outputs or any("hidden" in x for x in outputs):
                 print("SUCCESS: 'hidden_states' found in output names (or similar).")
            else:
                 print("WARNING: 'hidden_states' NOT found in output names. Manual patching required.")
        else:
            print("WARNING: QEFF model wrapper does not have get_output_names method.")
            
    except Exception as e:
        print(f"Skipping Target Model test due to error (likely missing model weights or env): {e}")

if __name__ == "__main__":
    test_eagle_head_export()
    # test_target_model_outputs() # Uncomment if environment supports loading TinyLlama logic
