# SV 模組追蹤表 — Llama 3.2 1B 推論引擎

> **目標：** FPGA 原型驗證 → ASIC 下線，參數化設計
> **程式風格：** 2-process（`always_ff` 僅做暫存器更新，`always_comb` 做 FSM + 組合邏輯），`assign` 做不在乎順序的組合電路
> **參考範本：** `fft.sv` 風格（`_reg/_next` 命名，`typedef enum` 明確 FSM）

---

## 模組清單

基於 `runq_c_report.md` 資料流與 `runq.c` forward pass。

### 基礎設施

| 模組 | 檔案 | 狀態 | 說明 |
|---|---|---|---|
| fifo | `fifo.sv` | ✅ 完成 | 非同步雙時鐘 FIFO（已有） |
| quantize | `quantize.sv` | ⬜ TODO | float32 → int8（每 64 個一組找最大絕對值，算 scale，四捨五入） |
| dequantize | `dequantize.sv` | ⬜ TODO | int8 → float32（int8 × scale） |

### 運算模組

| 模組 | 檔案 | 狀態 | 說明 |
|---|---|---|---|
| matmul_int8 | `matmul_int8.sv` | ⬜ TODO | **核心模組。** W(d,n)×x(n,)→xout(d,)，int8 分組量化（GS=64）。用 `NUM_PE` 參數控制 FPGA↔ASIC 擴展 |
| rmsnorm | `rmsnorm.sv` | ⬜ TODO | 1/√(Σx²/n) × weight × x，逐元素 float32 |
| rope | `rope.sv` | ⬜ TODO | 旋轉位置編碼，cos/sin 查表 (LUT) |
| dot_product | `dot_product.sv` | ⬜ TODO | Q·K float32 內積（64 個元素），用於 attention |
| softmax | `softmax.sv` | ⬜ TODO | exp 查表 (LUT) + 找最大值 + 減去 + 求和 + 除法 |
| weighted_sum | `weighted_sum.sv` | ⬜ TODO | Σ(scalar × vector)，attention V 加權累加 |
| swiglu | `swiglu.sv` | ⬜ TODO | silu(x) × gate，sigmoid 查表 (LUT) |

### 記憶體 / 控制

| 模組 | 檔案 | 狀態 | 說明 |
|---|---|---|---|
| kv_cache | `kv_cache.sv` | ⬜ TODO | (layer, seq_len, kv_dim) 讀寫，底層用 SRAM 或 BRAM |
| weight_loader | `weight_loader.sv` | ⬜ TODO | 從外部記憶體串流權重 → FIFO → matmul |
| transformer_layer | `transformer_layer.sv` | ⬜ TODO | 單層 transformer FSM：RMSNorm → QKV → Attn → FFN → Residual |
| transformer_top | `transformer_top.sv` | ⬜ TODO | 頂層模組：16 層依序執行、embedding lookup、最終 logits |

### 驗證

| 模組 | 檔案 | 狀態 | 說明 |
|---|---|---|---|
| tb_matmul_int8 | `tb/tb_matmul_int8.sv` | ⬜ TODO | Testbench：RTL 輸出 vs C golden data 比對 |
| golden_dumper | `golden_dumper.c` | ⬜ TODO | 改造 runq.c：逐層 dump 中間值到二進位檔案 |

---

## 參數策略（FPGA → ASIC）

```
parameter NUM_PE = 4;         // FPGA: 4-16 個 PE，時間複用
                              // ASIC: 設成 d（2048 或 8192）完全平行
parameter GS = 64;            // int8 量化分組大小（固定）
parameter DIM = 2048;         // 模型維度
parameter HIDDEN_DIM = 8192;  // FFN 隱藏層維度
```

**同一份 RTL 代碼**跑在兩個目標上，只改 `NUM_PE`：
- FPGA：`NUM_PE=4` → 每個 PE 處理 1 個輸出列，4 列同時運算，跑 d/4 輪
- ASIC：`NUM_PE=DIM` → 所有列 1 輪完成（完全平行展開）

---

## 驗證策略

### 確定性
LLM forward pass 是**完全確定性**的，條件：
- 相同的輸入 token
- 相同的權重
- Temperature = 0（greedy argmax）或固定 seed

`runq.c` 的 `forward()` 本身沒有隨機性。取樣的隨機性只在最後的 token 選擇（可設為 greedy/argmax）。

### 逐模組 Golden Data 驗證法
1. 改造 `runq.c` → `golden_dumper.c`：在每一步 dump 輸入/輸出到二進位檔案
2. Testbench 讀取二進位檔案，餵入 RTL 模組，比對輸出是否 **bit-accurate**

### Matmul 驗證（第一個目標）
1. 選一層（如 layer 0），選一個 matmul（如 W1）
2. 從 C dump：`xq`（int8 輸入 + scales）、`W1`（int8 權重 + scales）、`hb`（float32 輸出）
3. 把 `xq` 和 `W1` 透過 FIFO 餵入 `tb_matmul_int8.sv`
4. RTL 輸出 vs `hb` — int8 運算部分應 **bit-exact**

### Bit-Accuracy 預期
| 運算 | 預期精度 |
|---|---|
| int8 × int8 → int32 累加 | Bit-exact |
| int32 × float32 scale | 相同浮點捨入則 bit-exact |
| float32 跨 group 累加 | 可能有微小捨入誤差（≤1 ULP） |

---

## 建造順序（建議）

1. ✅ `fifo.sv` — 已完成
2. ⬜ `matmul_int8.sv` + `tb_matmul_int8.sv` — **當前優先**
3. ⬜ `golden_dumper.c` — dump 測試資料
4. ⬜ `quantize.sv` + `dequantize.sv`
5. ⬜ `rmsnorm.sv`
6. ⬜ `rope.sv`
7. ⬜ `softmax.sv`（exp LUT）
8. ⬜ `swiglu.sv`（sigmoid LUT）
9. ⬜ `dot_product.sv` + `weighted_sum.sv`
10. ⬜ `kv_cache.sv`
11. ⬜ `transformer_layer.sv`
12. ⬜ `transformer_top.sv`
