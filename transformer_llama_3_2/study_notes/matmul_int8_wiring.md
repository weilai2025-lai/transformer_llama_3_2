# matmul_int8 RTL Wiring Description + Tiling Strategy

> **Design target:** ASIC full parallel (Scheme B), FPGA scales via tiling

---

## Modules and Ports

| Unit | Type | Description |
|---|---|---|
| **x_fifo** | External input FIFO | Provides int8 input vector + scale factors |
| **w_fifo** | External input FIFO | Provides int8 weights + scale factors |
| **X Buffer** | Internal SRAM/register file | Stores complete x vector (N int8 + N/GS float32 scales) |
| **W Dispatch** | Internal distributor | Reads weights from w_fifo, dispatches to each PE simultaneously |
| **PE[0..NUM_PE-1]** | Processing Element | Each PE computes one output row, has two data inputs (x + w) |
| **Output Mux** | Multiplexer | Collects PE results, writes to xout_fifo in order |
| **FSM Controller** | Controller | Issues control signals to all units |
| **xout_fifo** | External output FIFO | Receives float32 output results |

---

## Parameters

```systemverilog
parameter NUM_PE = 4,       // Number of PEs (FPGA: 4-16, ASIC: D)
parameter N      = 2048,    // Input vector length
parameter D      = 2048,    // Output vector length (varies by matmul)
parameter GS     = 64       // Int8 quantization group size
```

D varies by usage:

| Usage | Weight Dimensions | N | D |
|---|---|---|---|
| QKV matmul (Wq) | W(2048,2048) | 2048 | 2048 |
| FFN matmul (W1, W3) | W(8192,2048) | 2048 | 8192 |
| FFN matmul (W2) | W(2048,8192) | 8192 | 2048 |

---

## Wiring Description

### ① x_fifo → X Buffer

```
x_fifo.dout ──[8-bit int8]──→ X Buffer write port
```

- FSM enters **LOAD_X**: reads N int8 values + N/GS float32 scales from x_fifo
- Once stored, X Buffer contents are fixed — shared across all tiles (no re-read)

### ② X Buffer → All PEs (x_data broadcast, cyan solid line)

```
X Buffer read port ──[8-bit bus]──→ vertical bus ──→ PE[0].x_in
                                                ├──→ PE[1].x_in
                                                ├──→ PE[2].x_in
                                                └──→ PE[3].x_in
```

- **One-to-many broadcast**: same cycle, all PEs read the same x value
- FSM controls X Buffer read address (0 → 1 → ... → N-1)

### ③ X Buffer → All PEs (x_scale broadcast, cyan dashed line)

```
X Buffer scale read ──[32-bit float]──→ vertical bus ──→ PE[0].x_scale
                                                    ├──→ PE[1].x_scale
                                                    ├──→ PE[2].x_scale
                                                    └──→ PE[3].x_scale
```

- Parallel to wire ②, 32-bit float width
- One x_scale per GS=64 x_data values, read once per group

### ④ w_fifo → W Dispatch → Each PE (w_row, orange solid line)

```
w_fifo.dout ──[8-bit int8]──→ W Dispatch ──→ PE[0].w_in  (w_row[0])
                                          ──→ PE[1].w_in  (w_row[1])
                                          ──→ PE[2].w_in  (w_row[2])
                                          ──→ PE[3].w_in  (w_row[3])
```

- **One-to-many, different data**: each PE gets a different weight row
- W Dispatch outputs NUM_PE streams simultaneously
- Each tile, PE[i] receives weight row `tile_cnt × NUM_PE + i`

### ⑤ W Dispatch → Each PE (w_scale, orange dashed line)

```
W Dispatch ──→ PE[0].w_scale  (w_scale[0])
           ──→ PE[1].w_scale  (w_scale[1])
           ──→ PE[2].w_scale  (w_scale[2])
           ──→ PE[3].w_scale  (w_scale[3])
```

- Parallel to wire ④, one per group (32-bit float)
- Each PE's w_scale is different (from its own weight row)

### ⑥ PE Internal Datapath

```
x_in [8-bit] ──┐
               ├──→ 64 multipliers (int8×int8) ──→ adder tree (Σ64) ──→ int32
w_in [8-bit] ──┘                                                         │
                                                                         ▼
x_scale [float32] ──┐
                    ├──→ multiplier (int32 × w_scale × x_scale) ──→ float32
w_scale [float32] ──┘                                                    │
                                                                         ▼
                                                              FP32 accumulator (Σ N/GS groups)
                                                                         │
                                                                         ▼
                                                                   result [float32]
```

- Each cycle reads 1 pair (x, w), 64 consecutive cycles complete one group
- INT32 accumulated per group → multiply scales → add to FP32 accumulator
- After N/GS groups, FP32 accumulator holds the final output for this row

### ⑦ All PEs → Output Mux → xout_fifo (green solid line)

```
PE[0].result ──→ ┐
PE[1].result ──→ ├──→ Output Mux ──[float32]──→ xout_fifo.din
PE[2].result ──→ ┤
PE[3].result ──→ ┘
```

- FSM enters **OUTPUT_TILE**: writes each PE's result to xout_fifo in order

### ⑧ FSM Controller → All Units (control signals)

```
FSM → x_fifo.rd_en      (assert during LOAD_X)
FSM → X Buffer.wr_en    (assert during LOAD_X)
FSM → X Buffer.rd_addr  (increment 0~N-1 during COMPUTE_TILE)
FSM → w_fifo.rd_en      (assert during COMPUTE_TILE)
FSM → PE[*].enable      (assert during COMPUTE_TILE)
FSM → Output Mux.sel    (select PE during OUTPUT_TILE)
FSM → xout_fifo.wr_en   (assert during OUTPUT_TILE)
FSM → done              (assert at DONE)
```

---

## Tiling Strategy

### Why Tiling

The largest matmul is FFN W1: D=8192 output rows. With `NUM_PE=4`, compute in `D/NUM_PE = 2048` tile rounds.

### FSM (with Tiling)

```
IDLE → LOAD_X → COMPUTE_TILE → OUTPUT_TILE ──→ DONE
                    ↑               │
                    └── tile not done ┘
```

| State | Action | Count |
|---|---|---|
| IDLE | Wait for start signal | 1 |
| LOAD_X | Read entire x vector into X Buffer | **1 time** (shared by all tiles) |
| COMPUTE_TILE | NUM_PE PEs compute simultaneously, processing tile_cnt batch | **D/NUM_PE rounds** |
| OUTPUT_TILE | Write this round's NUM_PE results to xout_fifo | **D/NUM_PE rounds** |
| DONE | Assert done signal | 1 |

### Counters

```systemverilog
logic [$clog2(D/NUM_PE)-1:0] tile_cnt_reg, tile_cnt_next;
// tile_cnt: 0, 1, 2, ..., (D/NUM_PE - 1)
// Each tile, PE[i] computes row (tile_cnt × NUM_PE + i)
```

### Key Optimization

- **x read once**: LOAD_X executes once, X Buffer contents reused across all tiles
- **w read per tile**: each tile reads NUM_PE new weight rows from w_fifo
- **ASIC needs no tiling**: when `NUM_PE = D`, only 1 tile round, FSM goes COMPUTE → OUTPUT → DONE directly
