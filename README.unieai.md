# UnieAI Eagle 3 Implementation Report

本文件整理目前在 `efficient-transformers` 內的 Eagle 3 Speculative Decoding 實作現況與執行方式，並將後續優化方向收錄在 Roadmap。

## Overview
我們聚焦在 Eagle 3 Draft Head 的可用性與 throughput，目標是讓 Draft/Target 的資料流能在 QAIC 執行環境中穩定運作，並逐步移除 Python 迴圈的 per‑token overhead。

## Current Implementation
- **EagleHead + EagleConfig**: `QEfficient/transformers/spd/eagle.py` 提供 Draft Head，支援 `past_key/past_value`，回傳 `present_key/present_value`，並允許 `seq_len` 輸入。
- **Hidden States 對齊**: `spd_transform_forward.py` 回傳 `hidden_states`；`QEFFAutoModelForCausalLM` 在 `qaic_config={"return_hidden_states": True}` 時輸出 `hidden_states` 以供 Eagle 使用。
- **ONNX Export**: `examples/performance/speculative_decoding/eagle_utils.py` 提供 `compile_eagle_head` 與 `compile_eagle_loop`（固定步數的 ONNX loop），含 dynamic axes 與 KV cache I/O。
- **Inference Flow**: `examples/performance/speculative_decoding/eagle_inference.py` 支援：
  - Target prefill chunking
  - PyTorch cold‑start 產生 initial KV
  - ORT/ONNX 的 Draft loop（可一次執行 K 步）
  - `eagle_cache_max_len` 截斷 cache

## How to Run (Draft Loop)
```bash
python examples/performance/speculative_decoding/eagle_inference.py \
  --target-model-name "meta-llama/Llama-3.2-1B" \
  --eagle-weights path/to/eagle_head.pt \
  --eagle-use-cache \
  --eagle-on-device-loop \
  --num-speculative-tokens 4
```
注意：Target 端需 QAIC 環境；Eagle 端使用 ONNX Runtime 執行。

## Known Limitations
- **Verify loop 尚未實作**：目前只完成 Draft 產生與輸出。
- **On-device loop 為 ONNX unroll**：已移除 Python per‑token loop，但尚未整合 QAIC 控制流。
- **記憶體占用**：推論時會同時持有 PyTorch（cold‑start）與 ORT Session。

## Roadmap (原始規劃)

### Speculative Decoding Architecture Comparison
| Optimization | Eagle 3 (Current Focus) | Draft-based | Multi-Projection (Turbo) |
| :--- | :--- | :--- | :--- |
| **KV Cache** | ✅ Implemented | ✅ Supported (via QEfficient) | ❌ Not Applicable |
| **On-Device Loop** | ✅ ONNX Unrolled | 🚀 **High Impact** | ❌ Not Applicable (Parallel) |
| **Tree Verification** | ⚡ **Advanced** | ⚠️ Optional (Complex) | 🚀 **High Impact** |

### 1. KV Cache Integration
**Target**: `Eagle 3`  
- **Why**: Eagle 需要完整 Draft history。  
- **Benefit**: 提升 Acceptance Rate。  
- **Status**: **Done**（已支援 `past_key/past_value`）。

### 2. On-Device Autoregressive Loop
**Target**: `Eagle 3`, `Draft-based`  
- **Why**: 移除 Python per‑token overhead。  
- **Benefit**: 大幅提高 throughput。  
- **Status**: **Done**（已完成 ONNX Loop Unrolling 與 Inference 整合）。  

### 3. Tree Verification (Tree Speculative Decoding)
**Target**: `Multi-Projection`, `Eagle 3`  
- **Why**: 同步驗證多條候選路徑。  
- **Benefit**: 提升 acceptance rate。  
- **Status**: **TODO**（需 tree mask + target attention 修改）。

## Implementation Priorities
1. **Verify loop 實作**（Target 驗證 Draft tokens，完成 End-to-End）。  
2. **Tree Verification**（提升 multi‑path acceptance）。  

## Verified Components
已新增單元測試 `tests/test_eagle_components.py` 驗證以下核心邏輯：
- `EagleFeatureCache`: Sliding window buffer 正確性。
- `EagleHead`: Forward pass 與 KV Cache 形狀對齊。
- `EagleDraftLoop`: K-step unrolling 邏輯驗證。  
