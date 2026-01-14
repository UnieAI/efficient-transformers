# UnieAI Eagle 3 Implementation Report

本文件整理目前在 `efficient-transformers` 內的 Eagle 3 Speculative Decoding 實作現況與執行方式，並將後續優化方向收錄在 Roadmap。

## Overview
我們聚焦在 Eagle 3 Draft Head 的可用性與 throughput，目標是讓 Draft/Target 的資料流能在 QAIC 執行環境中穩定運作，並逐步移除 Python 迴圈的 per‑token overhead。

## Current Implementation
- **EagleHead + EagleConfig**: `QEfficient/transformers/spd/eagle.py` 提供 Draft Head，支援 `past_key/past_value`，回傳 `present_key/present_value`，並允許 `seq_len` 輸入。
- **Hidden States 對齊**: `spd_transform_forward.py` 回傳 `hidden_states`；`QEFFAutoModelForCausalLM` 在 `qaic_config={"return_hidden_states": True}` 時輸出 `hidden_states` 以供 Eagle 使用。
- **ONNX Export**: `examples/performance/speculative_decoding/eagle_utils.py` 提供 `compile_eagle_head` 與 `compile_eagle_loop`（固定步數的 ONNX loop），含 dynamic axes 與 KV cache I/O。
- **QAIC Inference Flow**: `examples/performance/speculative_decoding/eagle_inference.py` 支援：
  - Target prefill chunking
  - Eagle Head/Loop 以 QPC 在 QAIC 執行（不落回 CPU）
  - `eagle_cache_max_len` 截斷 cache
  - Draft/Target 同卡（`--device-group`）與 cores split（`--target-num-cores`/`--eagle-num-cores`）
  - Verify loop（Target 驗證 Draft tokens，計算接受數量）
  - QAIC compiler 產物集中在 `qaic_workdir/`（避免 `shared_kernels*` 掉到根目錄）

## How to Run (QAIC Draft + Verify)
```bash
python examples/performance/speculative_decoding/eagle_inference.py \
  --target-model-name "meta-llama/Llama-3.2-1B" \
  --eagle-weights JKroller/llama3.2-1b-eagle \
  --eagle-use-cache \
  --eagle-on-device-loop \
  --num-speculative-tokens 4 \
  --device-group "2" \
  --target-num-cores 14 \
  --eagle-num-cores 2 \
  --max-tokens 64
```

## How to Run (QAIC N-gram Draft)
```bash
python examples/performance/speculative_decoding/ngram_inference.py \
  --target-model-name "meta-llama/Llama-3.2-1B" \
  --max-ngram-size 3 \
  --lookahead-tokens 8 \
  --num-speculative-tokens 4 \
  --device-group "2" \
  --target-num-cores 14 \
  --max-tokens 64
```

## How to Run (QAIC N-gram Chat Completion)
```bash
python examples/performance/speculative_decoding/ngram_inference.py \
  --target-model-name "meta-llama/Llama-3.2-1B" \
  --messages '[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"List colors of rainbow: red,"}]' \
  --max-ngram-size 3 \
  --lookahead-tokens 8 \
  --num-speculative-tokens 4 \
  --device-group "2" \
  --target-num-cores 14 \
  --max-tokens 64
```
注意：`--eagle-weights` 會自動對應本地 `./models/JKroller/llama3.2-1b-eagle/model.safetensors`。  
如需指定其他位置，使用 `--eagle-weights /path/to/model.safetensors`。

## Known Limitations
- **Verify loop 為 greedy match**：目前只做 token-by-token greedy 驗證，未支援 tree verification。
- **生成迭代**：每次驗證後接受正確 draft tokens，否則接受一個 verified token，直到 `--max-tokens`。
- **On-device loop 為固定步數 unroll**：仍需依 `num_speculative_tokens` 重新編譯 Eagle loop QPC。

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
1. **Tree Verification**（提升 multi‑path acceptance）。  
2. **Acceptance rate 與延遲評估**（統計實際 acceptance、QAIC 負載與 NSP 使用）。  

## Verified Components
尚未新增對應單元測試；目前以實機 QAIC 執行結果為準。
