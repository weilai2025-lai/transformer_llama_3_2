# SV Module Tracker — Llama 3.2 1B Inference Engine

> **Goal:** FPGA prototyping → ASIC tapeout, parameterized design
> **Coding style:** 2-process (`always_ff` for register update, `always_comb` for FSM + logic), `assign` for order-independent combinational
> **Reference:** `fft.sv` style (`_reg/_next` naming, explicit FSM `typedef enum`)

---

## Module List

Based on `runq_c_report.md` data flow and `runq.c` forward pass.

### Infrastructure

| Module | File | Status | Description |
|---|---|---|---|
| fifo | `fifo.sv` | ✅ Done | Async dual-clock FIFO (existing) |
| quantize | `quantize.sv` | ⬜ TODO | float32 → int8 (find max abs per group of 64, compute scale, round) |
| dequantize | `dequantize.sv` | ⬜ TODO | int8 → float32 (int8 × scale) |

### Compute Modules

| Module | File | Status | Description |
|---|---|---|---|
| matmul_int8 | `matmul_int8.sv` | ⬜ TODO | **Core module.** W(d,n)×x(n,)→xout(d,), int8 grouped (GS=64). Parameterized `NUM_PE` for FPGA↔ASIC scaling |
| rmsnorm | `rmsnorm.sv` | ⬜ TODO | 1/√(Σx²/n) × weight × x, element-wise float32 |
| rope | `rope.sv` | ⬜ TODO | Rotary positional encoding, cos/sin via LUT |
| dot_product | `dot_product.sv` | ⬜ TODO | Q·K float32 dot product (64 elements), used in attention |
| softmax | `softmax.sv` | ⬜ TODO | exp via LUT, find-max + subtract + sum + divide |
| weighted_sum | `weighted_sum.sv` | ⬜ TODO | Σ(scalar × vector), attention V accumulation |
| swiglu | `swiglu.sv` | ⬜ TODO | silu(x) × gate, sigmoid via LUT |

### Memory / Control

| Module | File | Status | Description |
|---|---|---|---|
| kv_cache | `kv_cache.sv` | ⬜ TODO | (layer, seq_len, kv_dim) read/write, SRAM or BRAM backed |
| weight_loader | `weight_loader.sv` | ⬜ TODO | Stream weights from external memory → FIFO → matmul |
| transformer_layer | `transformer_layer.sv` | ⬜ TODO | One transformer layer FSM: RMSNorm → QKV → Attn → FFN → Residual |
| transformer_top | `transformer_top.sv` | ⬜ TODO | Top-level: 16 layers sequential, embedding lookup, final logits |

### Verification

| Module | File | Status | Description |
|---|---|---|---|
| tb_matmul_int8 | `tb/tb_matmul_int8.sv` | ⬜ TODO | Testbench: compare RTL output vs C golden data |
| golden_dumper | `golden_dumper.c` | ⬜ TODO | Modified runq.c: dump intermediate values per layer to binary files |

---

## Parameter Strategy (FPGA → ASIC)

```
parameter NUM_PE = 4;      // FPGA: 4-16 PEs, time-multiplexed
                           // ASIC: set to d (2048 or 8192) for full parallel
parameter GS = 64;         // Group size for int8 quantization (fixed)
parameter DIM = 2048;      // Model dimension
parameter HIDDEN_DIM = 8192; // FFN hidden dimension
```

The **same RTL code** runs on both targets; only `NUM_PE` changes:
- FPGA: `NUM_PE=4` → each PE processes 1 output row, 4 rows computed in parallel, iterate d/4 rounds
- ASIC: `NUM_PE=DIM` → all rows computed in 1 round (fully parallel)

---

## Verification Strategy

### Determinism
LLM forward pass is **fully deterministic** given:
- Same input token
- Same weights
- Temperature = 0 (greedy argmax) or fixed seed for sampling

The `runq.c` code has no randomness in `forward()` itself. Sampling randomness is only in the final token selection (can be set to greedy/argmax).

### Per-Module Golden Data Approach
1. Modify `runq.c` → `golden_dumper.c`: at each step, dump inputs/outputs to binary files
2. Testbench reads binary files, feeds input to RTL module, compares output **bit-accurately**

### Matmul Verification (First Target)
1. Pick one layer (e.g., layer 0), one matmul (e.g., W1)
2. Dump from C: `xq` (int8 input + scales), `W1` (int8 weights + scales), `hb` (float32 output)
3. Feed `xq` and `W1` into `tb_matmul_int8.sv` via FIFO
4. Compare RTL output vs `hb` — should be **bit-exact** since all operations are integer arithmetic + float multiply

### Bit-Accuracy Expectations
| Operation | Expected accuracy |
|---|---|
| int8 × int8 → int32 accumulate | Bit-exact |
| int32 × float32 scale | Bit-exact if same float rounding |
| float32 accumulate across groups | May have tiny rounding differences (≤1 ULP) |

---

## Build Order (Suggested)

1. ✅ `fifo.sv` — already done
2. ⬜ `matmul_int8.sv` + `tb_matmul_int8.sv` — **current priority**
3. ⬜ `golden_dumper.c` — dump test data
4. ⬜ `quantize.sv` + `dequantize.sv`
5. ⬜ `rmsnorm.sv`
6. ⬜ `rope.sv`
7. ⬜ `softmax.sv` (exp LUT)
8. ⬜ `swiglu.sv` (sigmoid LUT)
9. ⬜ `dot_product.sv` + `weighted_sum.sv`
10. ⬜ `kv_cache.sv`
11. ⬜ `transformer_layer.sv`
12. ⬜ `transformer_top.sv`
