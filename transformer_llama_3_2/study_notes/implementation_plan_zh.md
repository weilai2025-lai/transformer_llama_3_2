# 實作計畫 — INT8 Matmul RTL 模組

## 目標

實作 LLM 推論引擎的第一個 RTL 模組（`matmul_int8.sv`），搭配 C golden data dumper 與 SystemVerilog testbench。這個模組將建立後續所有模組的設計規範。

---

## FIFO 評估

現有 `fifo.sv` 是**非同步雙時鐘** FIFO。對於 matmul 模組：

> [!WARNING]
> 目前的 FIFO 使用分開的 `wr_clk` 和 `rd_clk`。運算模組通常在**同一個時鐘域**工作。兩種做法：
> - (A) 把同一個 clock 接到 `wr_clk` 和 `rd_clk`（可以用，但浪費了 CDC 邏輯）
> - (B) 新建更簡潔的 **sync FIFO**（`sync_fifo.sv`），單時鐘 — 更乾淨
>
> 建議 **(B)**：compute pipeline 內部用 sync FIFO，保留 async 版本給外部記憶體介面用。

---

## 提議的變更

### 基礎設施

#### [NEW] [sync_fifo.sv](file:///Users/laiweiin/Documents/Github/transformer_llama_3_2/transformer_llama_3_2/sv/sync_fifo.sv)

單時鐘同步 FIFO，給 compute pipeline 用。介面風格與現有 `fifo.sv` 一致但更簡單（一個 clock）。

---

### 運算模組

#### [NEW] [matmul_int8.sv](file:///Users/laiweiin/Documents/Github/transformer_llama_3_2/transformer_llama_3_2/sv/matmul_int8.sv)

**從 `runq.c` 對應的演算法：**

```c
// 對每個輸出列 i (0..d-1):
//   對每個 group j (0..n/GS-1):
//     ival = Σ(k=0..GS-1) x.q[j*GS+k] * w.q[i*n+j*GS+k]   // int8×int8→int32
//     val += (float)ival * w_scale[group] * x_scale[group]     // int32→float32
//   xout[i] = val
```

**RTL 架構：**

```
             ┌─────────────┐
  x_fifo ──→│             │──→ xout_fifo
  w_fifo ──→│ matmul_int8 │
             │ (NUM_PE PEs)│
             └─────────────┘

每個 PE 計算一個輸出列：
  - 每個 group 讀 GS=64 對 (x_int8, w_int8)
  - MAC: int8 × int8 → 累加到 int32
  - 每個 group 結束時: int32 × w_scale × x_scale → 累加到 float32
  - 所有 group 結束後: 輸出 float32 結果
```

**關鍵參數：**

```systemverilog
parameter NUM_PE    = 4,     // FPGA: 4-16, ASIC: 最大到 DIM
parameter N         = 2048,  // 輸入向量長度
parameter D         = 2048,  // 輸出向量長度
parameter GS        = 64     // 分組大小
```

**FSM 狀態：** `IDLE → LOAD_X → COMPUTE → OUTPUT → (下一列或 DONE)`

- `LOAD_X`：讀取整個 x 向量（n 個 int8 值 + n/GS 個 scales）到內部 buffer（廣播給所有 PE）
- `COMPUTE`：每個 PE 從 w_fifo 讀取自己的權重列，每次處理 GS 個元素
- `OUTPUT`：寫入結果到 xout_fifo

**FPGA → ASIC 擴展：** 只改 `NUM_PE`。`NUM_PE=D` 時所有列同時運算（1 輪）。`NUM_PE=4` 時需要 `D/4` 輪。

---

### 驗證

#### [NEW] [golden_dumper.c](file:///Users/laiweiin/Documents/Github/transformer_llama_3_2/transformer_llama_3_2/sv/tb/golden_dumper.c)

改造 `runq.c`，在第一個 W1 matmul（layer 0）dump 二進位檔案：
- `x_int8.bin`：輸入 int8 值（2048 bytes）
- `x_scales.bin`：輸入 scale 因子（2048/64 = 32 個 float）
- `w_int8.bin`：權重 int8 值（8192 × 2048 bytes）
- `w_scales.bin`：權重 scale 因子（8192 × 2048/64 個 float）
- `output.bin`：預期 float32 輸出（8192 個 float）

#### [NEW] [tb_matmul_int8.sv](file:///Users/laiweiin/Documents/Github/transformer_llama_3_2/transformer_llama_3_2/sv/tb/tb_matmul_int8.sv)

Testbench 流程：
1. 讀取 golden 二進位檔案
2. 把 x 和 w 資料塞入 FIFO
3. 執行 matmul_int8
4. 比對輸出 vs golden — 回報 bit-accuracy 與最大誤差

---

## 驗證計畫

### 自動化測試

1. **編譯 golden_dumper.c 並產生測試資料：**
   ```bash
   cd transformer_llama_3_2/sv/tb
   gcc -o golden_dumper golden_dumper.c -lm -lpcre
   ./golden_dumper ../../llama3_2_1b.bin -z ../../tokenizer.bin -p "hello" -n 1
   # 產出: x_int8.bin, x_scales.bin, w_int8.bin, w_scales.bin, output.bin
   ```

2. **執行 RTL 模擬（iverilog 或 Verilator）：**
   ```bash
   cd transformer_llama_3_2/sv
   iverilog -g2012 -o tb_matmul tb/tb_matmul_int8.sv matmul_int8.sv sync_fifo.sv
   vvp tb_matmul
   # 預期: "PASS: all outputs match golden data" 或回報最大 ULP 誤差
   ```

### 驗收標準
- INT8×INT8 累加：**bit-exact** 吻合
- Float32 scale 乘法：允許 **≤ 1 ULP** 差異（浮點捨入）
- 全部 8192 個輸出值都必須通過

---

## 建造順序

1. `sync_fifo.sv` — 單時鐘 FIFO 給 compute pipeline 用
2. `golden_dumper.c` — 從 C 模型產生測試資料
3. `matmul_int8.sv` — 核心運算模組
4. `tb_matmul_int8.sv` — testbench 與驗證
