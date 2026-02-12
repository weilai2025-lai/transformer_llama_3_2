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

### 4.1 Model Parameters (Verified from Safetensors)

**Actual parameter breakdown** (inspected from `model.safetensors`):

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Embedding (`embed_tokens`) | 128,256 × 2,048 = 262,668,288 | 21.3% |
| 16 Transformer Layers | 973,144,064 | 78.7% |
| Final LayerNorm | 2,048 | ~0% |
| **Total** | **1,235,814,400 (1.236B)** | **100%** |

> [!IMPORTANT]
> `tie_word_embeddings = true` in config.json — `lm_head` and `embed_tokens` **share the same weights**. No separate `lm_head` tensor exists in the safetensors file.

> [!WARNING]
> The current bnb-4bit model stores embedding in **bfloat16** (0.525 GB, NOT quantized). Only linear layers are quantized to nf4.

### 4.2 Model Weights in Int4 (Storage)

**Scenario A: All weights quantized to int4 (including embedding)**
```
Weight Size = 1.236B × 4 bits / 8 = 0.618 GB

Scale factors (group_size=128, float32):
  (1.236B / 128) × 4 bytes ≈ 0.04 GB

Total weights (int4): ~0.66 GB
```

**Scenario B: Embedding kept in bfloat16 (like current bnb model)**
```
Embedding (bfloat16): 262M × 2 bytes = 0.525 GB
Other weights (int4): 973M × 4/8 + scale ≈ 0.52 GB
Total: ~1.04 GB (already exceeds 1 GB!)
```

### 4.3 KV Cache

> [!WARNING]
> **Correction**: This model uses **GQA (Grouped Query Attention)** with `num_key_value_heads = 8` and `head_dim = 64`, so `kv_dim = 8 × 64 = 512` (NOT 2048).

**Formula**:
```
KV_Cache = n_layers × 2 × seq_len × kv_dim × sizeof(float)
         = 16 × 2 × seq_len × 512 × 4
```

| seq_len | KV Cache Size |
|---------|---------------|
| 2048 | 16 × 2 × 2048 × 512 × 4 = **0.134 GB** |
| 512  | 16 × 2 × 512 × 512 × 4 = **0.034 GB** |
| 256  | 16 × 2 × 256 × 512 × 4 = **0.017 GB** |
| 128  | 16 × 2 × 128 × 512 × 4 = **0.008 GB** |

### 4.4 Activations

For token-by-token inference (no batching), buffers are reused across layers:
```
- Reusable buffers (x, xb, q, k, v, hb, etc.):
  dim=2048 and intermediate_size=8192 vectors
  ~100 KB (reused across layers)
- Attention scores: num_heads × seq_len × 4 bytes
  (32 × 256 × 4 = 32 KB per layer)
- Total activation buffers: ~1 MB
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

**Scenario A: All weights in int4 (including embedding)**

| Context Length | Weights (int4) | KV Cache | Act. | System | **Total** | Feasibility |
|---------------|---------------|----------|------|--------|-----------|-------------|
| seq_len=512 | 0.66 GB | 0.034 GB | 0.001 GB | 0.25 GB | **0.95 GB** | ✅ Fits |
| seq_len=256 | 0.66 GB | 0.017 GB | 0.001 GB | 0.25 GB | **0.93 GB** | ✅ Fits |
| seq_len=128 | 0.66 GB | 0.008 GB | 0.001 GB | 0.25 GB | **0.92 GB** | ✅ Fits |

**Scenario B: Embedding in bfloat16, rest in int4**

| Context Length | Weights | KV Cache | Act. | System | **Total** | Feasibility |
|---------------|---------|----------|------|--------|-----------|-------------|
| seq_len=256 | 1.04 GB | 0.017 GB | 0.001 GB | 0.25 GB | **1.31 GB** | ❌ Exceeds |

---

## 5. 💡 Conclusions and Recommendations

### ✅ Key Corrections from Safetensors Verification

1. **Embedding is part of model weights** — the original report double-counted it as a separate 1.0 GB item
2. **KV cache was 4× overestimated** — `kv_dim = 512` (GQA), not 2048
3. **Activations were overestimated** — token-by-token inference only needs ~1 MB, not 50-100 MB

### 📊 Updated Feasibility

**If embedding is quantized to int4** (requires custom C implementation):
- seq_len=512: Total **~0.95 GB** → ✅ **Fits in 1 GB DDR4!**
- Headroom is tight (~50 MB), careful memory management required

**If embedding remains in bfloat16/float32** (simpler implementation):
- Total **~1.04-1.3 GB** → ❌ Exceeds 1 GB limit
- Would require 2 GB DDR4 upgrade

### Recommended Path Forward

1. **Primary approach**: Implement int4 quantization for **ALL** weights including embedding
   - Requires custom int4 embedding lookup in C code
   - With seq_len=512, total memory fits within 1 GB
   - Tight margin (~50 MB headroom) — careful memory management required

2. **Fallback**: If int4 embedding is too complex to implement, upgrade to 2 GB DDR4 board
