# Llama 3.2 1B Int4 Quantization Model - Weekly Report

## 1. Progress: Model Download

✅ **Downloaded Model**: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- 📊 File Size: ~1.03 GB
- 🔢 Parameters: 1 Billion (1B)
- 🎯 Precision: 4-bit quantization

---

## 2. C Reference Code Status

### Available Resource: runq.c

The project includes an **int8 quantization** reference implementation `runq.c` (~1,200 lines of pure C code), but it **does not support int4 version**.

### Quantization Precision Comparison

| Precision | Weight Size (1B Model) | Quantization Range | Notes |
|-----------|------------------------|-------------------|-------|
| int8 | 1.06 GB | -128 ~ 127 | Already implemented in runq.c |
| **int4** | **0.53 GB** | **-8 ~ 7** | **Requires code modification** |

To implement int4, modifications to quantization formulas and bit-packing logic in runq.c are needed.

---

## 3. Int4 Quantization Principles

### 3.1 Quantization Formulas

**Forward Quantization**:
```
1. Find maximum absolute value in each group:
   wmax = max(|w₁|, |w₂|, ..., |w₁₂₈|)

2. Calculate scaling factor:
   scale = wmax / 7.0    (int4 range: -8~7)

3. Quantize:
   int4_value = round(float_value / scale)
```

**Dequantization**:
```
float_value = int4_value × scale
```

**Important**: After dequantization, values are converted to **float32** for computation because:
- GPU/CPU matrix operations require floating-point numbers
- int4 is only used for **storage** to save memory
- Computation **must use float32**

### 3.2 Example

Given a weight group: `[0.1, 0.5, 1.2, 2.8, -0.3]`

```
wmax = 2.8
scale = 2.8 / 7 = 0.4

Quantization:
  0.1  → round(0.1/0.4) = 0
  0.5  → round(0.5/0.4) = 1
  1.2  → round(1.2/0.4) = 3
  2.8  → round(2.8/0.4) = 7
 -0.3  → round(-0.3/0.4) = -1

Storage: [0, 1, 3, 7, -1] + scale(0.4)

Dequantization:
  0 × 0.4 = 0.0   (error: 0.1)
  1 × 0.4 = 0.4   (error: 0.1)
  3 × 0.4 = 1.2   (error: 0.0)
  7 × 0.4 = 2.8   (error: 0.0)
 -1 × 0.4 = -0.4  (error: 0.1)
```

### 3.3 Bit Packing

Int4 requires packing 2 values into 1 byte:
```c
uint8_t packed = (val1 & 0x0F) | ((val2 & 0x0F) << 4);
```

---

## 4. Memory Analysis (1GB DDR4 Constraint)

### 4.1 Model Weights (Int4 Version - Storage)

**Formula**:
```
Weight Size = Parameters × bits_per_param / 8 + scale_overhead

int4 weights (stored on disk):
  1B × 4 bits / 8 = 0.5 GB

Scale factors (group_size=128, float32):
  (1B / 128) × 4 bytes = 31.25 MB

Total (storage): 0.53 GB
```

### 4.2 Embedding Table (Runtime Memory)

**Critical Discovery**: The embedding table is **dequantized to float32** at runtime for fast lookup.

**Formula**:
```
Embedding Memory = vocab_size × dim × sizeof(float32)
                 = 128,256 × 2,048 × 4 bytes
                 = 1,050,214,400 bytes
                 ≈ 1.0 GB
```

⚠️ **This is a major memory consumer that was initially overlooked.**

### 4.3 KV Cache (Critical Memory Bottleneck)

**Formula**:
```
KV_Cache = n_layers × 2 × seq_len × kv_dim × sizeof(float)
```

**Calculation** (assuming n_layers=16, kv_dim=2048):

| seq_len | KV Cache Size |
|---------|---------------|
| 2048 | 16 × 2 × 2048 × 2048 × 4 = **1.0 GB** |
| 512  | 16 × 2 × 512 × 2048 × 4 = **0.27 GB** |
| **256**  | **16 × 2 × 256 × 2048 × 4 = 0.13 GB** |
| 128  | 16 × 2 × 128 × 2048 × 4 = 0.065 GB |

### 4.4 Activations

```
- Intermediate layer buffers (x, xb, q, k, v, etc.): ~50-100 MB
- Attention scores: n_heads × seq_len × 4 bytes
  (32 × 256 × 4 = 32 KB per layer)
```

### 4.5 Logits

```
vocab_size × 4 bytes = 128,256 × 4 = 0.5 MB
```

### 4.6 System Overhead

```
- OS kernel + drivers: ~150-200 MB
- Tokenizer (program + vocabulary): ~30-40 MB
- Total: ~200-250 MB
```

### 4.7 Total Memory Requirements (Int4 Version)

| Context Length | Weights | Embedding | KV Cache | Act. | System | **Total** | Feasibility |
|---------------|---------|-----------|----------|------|--------|-----------|-------------|
| seq_len=512 | 0.53 GB | 1.0 GB | 0.27 GB | 0.1 GB | 0.25 GB | **2.15 GB** | ❌ Exceeds |
| seq_len=256 | 0.53 GB | 1.0 GB | 0.13 GB | 0.1 GB | 0.25 GB | **2.01 GB** | ❌ Exceeds |
| **seq_len=128** | **0.53 GB** | **1.0 GB** | **0.065 GB** | **0.1 GB** | **0.25 GB** | **1.95 GB** | ❌ **Still Exceeds** |

---

## 5. 💡 Conclusions and Recommendations

### ⚠️ Critical Finding

**Using int4 quantization with dequantized embedding table**:
- seq_len=256: Total requirement **2.01 GB** (exceeds 1GB limit)
- seq_len=128: Total requirement **1.95 GB** (still exceeds)

**Root Cause**: The embedding table (vocab_size × dim) requires 1.0 GB when dequantized to float32 at runtime, which dominates memory usage.

### Feasible Solutions

| Solution | Description | Feasibility |
|----------|-------------|-------------|
| **Solution A: Upgrade to 2GB RAM** | Use 2GB DDR4 FPGA board | ✅ **Required**, not optional |
| **Solution B: Keep Embedding Quantized** | Modify code to keep embedding table in int4 | ⚠️ Slower lookup, complex implementation |
| **Solution C: Use Smaller Vocab** | Reduce vocabulary size | ⚠️ Requires model retraining |

### Weekly Conclusion

**Updated Recommendations**:

1. **Hardware Upgrade is Mandatory**
   - **Minimum requirement**: 2GB DDR4 RAM
   - Even with int4 and seq_len=128, we need ~1.95 GB
   - 1GB DDR4 is **insufficient** for this model

2. **With 2GB RAM, two options**:
   - **Option A (Simpler)**: Use int8 with existing runq.c code
     - Total: ~2.5 GB (need to reduce seq_len or optimize further)
   - **Option B (Optimal)**: Implement int4 with quantized embedding lookup
     - Could fit in 2GB with careful optimization
     - Requires significant code modification

3. **Critical Decision Required**:
   - ✅ **Must upgrade FPGA board to at least 2GB RAM**
   - ✅ Evaluate if 4GB RAM board is available (recommended)
   - ✅ If hardware upgrade impossible, consider smaller model (e.g., 500M parameters)
