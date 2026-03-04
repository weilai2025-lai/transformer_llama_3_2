# matmul_int8 RTL 連線說明 + Tiling 策略

> **設計目標：** ASIC 完全平行（方案 B），FPGA 透過 tiling 縮放

---

## 模組與 Port 清單

| 單元 | 類型 | 說明 |
|---|---|---|
| **x_fifo** | 外部輸入 FIFO | 提供 int8 輸入向量 + scale 因子 |
| **w_fifo** | 外部輸入 FIFO | 提供 int8 權重 + scale 因子 |
| **X Buffer** | 內部 SRAM/暫存器 | 存放完整 x 向量（N 個 int8 + N/GS 個 float32 scale） |
| **W Dispatch** | 內部分配器 | 從 w_fifo 讀取 weight，同時分配給各 PE |
| **PE[0..NUM_PE-1]** | 運算單元 | 每個 PE 算一個輸出列，有兩組資料輸入（x + w） |
| **Output Mux** | 多工器 | 收集各 PE 結果，依序寫入 xout_fifo |
| **FSM Controller** | 控制器 | 發出控制訊號給以上所有單元 |
| **xout_fifo** | 外部輸出 FIFO | 接收 float32 輸出結果 |

---

## 參數

```systemverilog
parameter NUM_PE = 4,       // PE 數量（FPGA: 4~16, ASIC: D）
parameter N      = 2048,    // 輸入向量長度
parameter D      = 2048,    // 輸出向量長度（依 matmul 而異）
parameter GS     = 64       // int8 量化分組大小
```

D 值依用途不同：

| 用途 | Weight 維度 | N | D |
|---|---|---|---|
| QKV matmul (Wq) | W(2048,2048) | 2048 | 2048 |
| FFN matmul (W1, W3) | W(8192,2048) | 2048 | 8192 |
| FFN matmul (W2) | W(2048,8192) | 8192 | 2048 |

---

## 連線說明

### ① x_fifo → X Buffer

```
x_fifo.dout ──[8-bit int8]──→ X Buffer 寫入端
```

- FSM 進入 **LOAD_X** 狀態時，從 x_fifo 連續讀出 N 個 int8 值 + N/GS 個 float32 scale
- 存入 X Buffer 後內容固定不變，所有 tile 共用（不需重讀）

### ② X Buffer → 所有 PE（x_data 廣播，青色實線）

```
X Buffer 讀出端 ──[8-bit bus]──→ 垂直 bus ──→ PE[0].x_in
                                         ├──→ PE[1].x_in
                                         ├──→ PE[2].x_in
                                         └──→ PE[3].x_in
```

- **一對多廣播**：同一個 cycle，所有 PE 讀到相同的 x 值
- FSM 控制 X Buffer 的讀取地址（0 → 1 → ... → N-1）

### ③ X Buffer → 所有 PE（x_scale 廣播，青色虛線）

```
X Buffer scale 讀出端 ──[32-bit float]──→ 垂直 bus ──→ PE[0].x_scale
                                                   ├──→ PE[1].x_scale
                                                   ├──→ PE[2].x_scale
                                                   └──→ PE[3].x_scale
```

- 跟連線 ② 平行，位寬 32-bit float
- 每 GS=64 個 x_data 對應 1 個 x_scale，每個 group 結束時讀一次

### ④ w_fifo → W Dispatch → 各 PE（w_row，橘色實線）

```
w_fifo.dout ──[8-bit int8]──→ W Dispatch ──→ PE[0].w_in  (w_row[0])
                                          ──→ PE[1].w_in  (w_row[1])
                                          ──→ PE[2].w_in  (w_row[2])
                                          ──→ PE[3].w_in  (w_row[3])
```

- **一對多、各自不同資料**：每個 PE 拿到不同的 weight row
- W Dispatch 同時輸出 NUM_PE 份資料
- 每輪 tile，PE[i] 拿到第 `tile_cnt × NUM_PE + i` 列的 weight

### ⑤ W Dispatch → 各 PE（w_scale，橘色虛線）

```
W Dispatch ──→ PE[0].w_scale  (w_scale[0])
           ──→ PE[1].w_scale  (w_scale[1])
           ──→ PE[2].w_scale  (w_scale[2])
           ──→ PE[3].w_scale  (w_scale[3])
```

- 跟連線 ④ 平行，每個 group 結束時送一次（32-bit float）
- 每個 PE 的 w_scale 不同（來自各自的 weight row）

### ⑥ PE 內部 datapath

```
x_in [8-bit] ──┐
               ├──→ 64 個乘法器 (int8×int8) ──→ 加法樹 (Σ64) ──→ int32
w_in [8-bit] ──┘                                                    │
                                                                    ▼
x_scale [float32] ──┐
                    ├──→ 乘法器 (int32 × w_scale × x_scale) ──→ float32
w_scale [float32] ──┘                                               │
                                                                    ▼
                                                          FP32 累加器 (Σ N/GS groups)
                                                                    │
                                                                    ▼
                                                              result [float32]
```

- 每個 cycle 讀 1 對 (x, w)，連續 64 cycles 完成一個 group
- INT32 累加完一個 group → 乘 scale → 加到 FP32 累加器
- 重複 N/GS 個 group 後，FP32 累加器的值就是這一列的最終輸出

### ⑦ 各 PE → Output Mux → xout_fifo（綠色實線）

```
PE[0].result ──→ ┐
PE[1].result ──→ ├──→ Output Mux ──[float32]──→ xout_fifo.din
PE[2].result ──→ ┤
PE[3].result ──→ ┘
```

- FSM 進入 **OUTPUT_TILE** 狀態時，依序把各 PE 結果寫入 xout_fifo

### ⑧ FSM Controller → 所有單元（控制訊號）

```
FSM → x_fifo.rd_en      (LOAD_X 時拉高)
FSM → X Buffer.wr_en    (LOAD_X 時拉高)
FSM → X Buffer.rd_addr  (COMPUTE_TILE 時遞增 0~N-1)
FSM → w_fifo.rd_en      (COMPUTE_TILE 時拉高)
FSM → PE[*].enable      (COMPUTE_TILE 時拉高)
FSM → Output Mux.sel    (OUTPUT_TILE 時選擇)
FSM → xout_fifo.wr_en   (OUTPUT_TILE 時拉高)
FSM → done              (DONE 時拉高)
```

---

## Tiling 策略

### 為什麼需要 Tiling

全模型最大的 matmul 是 FFN W1：D=8192 個輸出列。如果 `NUM_PE=4`，不可能一次算完所有列，需要分成 `D/NUM_PE = 2048` 輪 tile。

### FSM（含 Tiling）

```
IDLE → LOAD_X → COMPUTE_TILE → OUTPUT_TILE ──→ DONE
                    ↑               │
                    └── tile 未完成 ─┘
```

| 狀態 | 動作 | 次數 |
|---|---|---|
| IDLE | 等待 start 訊號 | 1 次 |
| LOAD_X | 從 x_fifo 讀取整個 x 向量到 X Buffer | **1 次**（所有 tile 共用） |
| COMPUTE_TILE | NUM_PE 個 PE 同時運算，處理第 tile_cnt 批 rows | **D/NUM_PE 輪** |
| OUTPUT_TILE | 把本輪 NUM_PE 個結果寫入 xout_fifo | **D/NUM_PE 輪** |
| DONE | 拉高 done 訊號 | 1 次 |

### 計數器

```systemverilog
logic [$clog2(D/NUM_PE)-1:0] tile_cnt_reg, tile_cnt_next;
// tile_cnt: 0, 1, 2, ..., (D/NUM_PE - 1)
// 每輪 tile，PE[i] 計算第 (tile_cnt × NUM_PE + i) 列
```

### 關鍵優化

- **x 只讀一次**：LOAD_X 只執行一次，X Buffer 的內容在所有 tile 中共用
- **w 每輪讀新的**：每個 tile 從 w_fifo 讀取 NUM_PE 列新的 weight rows
- **ASIC 不需要 tiling**：當 `NUM_PE = D` 時，tile 只有 1 輪，FSM 直接 COMPUTE → OUTPUT → DONE
