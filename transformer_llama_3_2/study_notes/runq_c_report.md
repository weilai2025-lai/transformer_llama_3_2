# runq.c Forward Pass Analysis Report

> **Target:** Llama 3.2 1B Int8 Quantized Inference
> **Scope:** `forward()` function (L390-532) — single-token forward pass

---

## 1. Operation Type Summary

The `forward()` function includes the following operations, categorized by precision and type:

| Step | Operation | Type | Precision | Dimensions | Notes |
|---|---|---|---|---|---|
| 1 | RMSNorm | Element-wise | float32 | (2048,) | 1/√(Σx²/n)×weight |
| 2 | QKV matmul (Wq,Wk,Wv) | Matrix×Vector | int8 | W(2048,2048)×x(2048,)→(2048,) | |
| 3 | RoPE | Pairwise (2D rotation) | float32 | 2 elements at a time | (x',y')=(x·cosθ−y·sinθ, x·sinθ+y·cosθ), θ=pos/500000^(d/64) |
| 4 | Attention Q·K | Dot Product | float32 | (64,)·(64,)→scalar | Per head |
| 5 | Softmax | Element-wise | float32 | (pos+1,) | softmax(xᵢ) = e^xᵢ / Σe^xⱼ |
| 6 | Attention scores×V | Scalar×Vector Sum | float32 | Σ(scalar×(64,))→(64,) | Weighted sum |
| 7 | Output matmul (Wo) | Matrix×Vector | int8 | W(2048,2048)×x(2048,)→(2048,) | |
| 8 | Residual | Element-wise add | float32 | (2048,) | x += xb |
| 9 | FFN matmul (W1,W3) | Matrix×Vector | int8 | W(8192,2048)×x(2048,)→(8192,) | ~70% of compute |
| 10 | SwiGLU activation | Element-wise | float32 | (8192,) | output = silu(W1(x)) × W3(x), where silu(x) = x × σ(x) |
| 11 | FFN matmul (W2) | Matrix×Vector | int8 | W(2048,8192)×x(8192,)→(2048,) | |
| 12 | Residual | Element-wise add | float32 | (2048,) | x += xb |

**Key observations:**
- **Matrix-vector multiplications (int8)** account for 90%+ of compute, suitable for hardware acceleration (systolic array)
- **Nonlinear functions** (softmax, SwiGLU, RoPE, RMSNorm) require float32, as int8 lacks exp, cos, sqrt operations

---

## 2. Int8 Quantization Strategy

### Why Quantize?

Matrix-vector multiplications dominate the computation. Using int8 instead of float32:
- Faster computation (int8 multiply is faster and requires less hardware area)
- Reduced memory usage (1 byte vs 4 bytes per weight)

### Which Operations Can Be Quantized?

**Only matmul can use int8**, as it only requires multiplication and addition.
Other operations (softmax's exp, RoPE's cos/sin, RMSNorm's sqrt) do not exist in integer arithmetic.

### Quantization Method

Every GS=64 floats form a group. The maximum absolute value determines the scale factor:

```c
scale = max_abs / 127;
quantized_value = round(float_value / scale);   // float32 → int8
```

During matmul, int8 × int8 accumulates into int32. After each group, multiply back by scale factors to restore float32:

```c
val += (float)ival * w_scale * x_scale;
```

![Int8 grouped matmul computation diagram](matmul_visualization.png)

### Data Flow Within One Transformer Layer

```
Step                        Precision      Quantize?
───────────────────────────────────────────────────────
1. RMSNorm(x → xb)         float32        ❌
   ── quantize(xb → xq) ──                ✅
2. matmul(xq × Wq → q)    int8×int8       ← QKV projection
3. matmul(xq × Wk → k)    int8×int8
4. matmul(xq × Wv → v)    int8×int8
5. RoPE(q, k)              float32         ❌
6. Save K,V to cache       float32         ❌
7. Q·K dot product + softmax  float32      ❌
8. scores × V              float32         ❌
   ── quantize(attn → xq) ──              ✅
9. matmul(xq × Wo)         int8×int8
10. Residual x += xb        float32        ❌
11. RMSNorm → FFN           float32        ❌
   ── quantize ──                          ✅
12. matmul(xq × W1 → hb)   int8×int8
13. matmul(xq × W3 → hb2)  int8×int8
14. SwiGLU(hb, hb2)         float32        ❌  silu(hb) × hb2
   ── quantize ──                          ✅
15. matmul(hq × W2 → xb)   int8×int8
16. Residual x += xb        float32        ❌
```

Per layer: **5 quantizations, 7 int8 matmuls, 1 SwiGLU activation**.

---

## 3. Forward Pass Step-by-Step Analysis

### 3.1 Model Parameters (Llama 3.2 1B)

| Parameter | Value |
|---|---|
| dim | 2048 |
| n_heads (Q) | 32 |
| n_kv_heads (K,V) | 8 |
| head_size | 64 |
| kv_dim | 512 |
| kv_mul (GQA) | 4 |
| hidden_dim (FFN) | 8192 |
| n_layers | 16 |

### 3.2 Embedding Lookup

```c
memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));
```

Look up the token ID in the embedding table, copy a 2048-dimensional vector into `x`.

### 3.3 RMSNorm

```c
rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
```

Formula: o[j] = weight[j] × x[j] / RMS(x)

- Input x (2048,), output xb (2048,), weight (2048,) is an element-wise learnable scale factor
- After RMSNorm, the vector's RMS value = 1, but L2 norm = √2048 ≈ 45.25

### 3.4 QKV Projection

```c
quantize(&s->xq, s->xb, dim);                    // float32 → int8
matmul(s->q, &s->xq, w->wq + l, dim, dim);       // xq × Wq → q (2048,)
matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);   // xq × Wk → k (512,)
matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);   // xq × Wv → v (512,)
```

Q has full 2048 dimensions (32 heads × 64), while K/V only have 512 dimensions (8 heads × 64) — this is the GQA design.

### 3.5 RoPE Positional Encoding

```c
vec[i]   = v0 * cosθ - v1 * sinθ
vec[i+1] = v0 * sinθ + v1 * cosθ
```

θ(pos, d) = pos / 500000^(d/64)

Each pair of adjacent elements is treated as a 2D point and rotated by an angle determined by both position and dimension.
The cos/sin values can be precomputed into a lookup table (size = seq_len × 32 × 2 floats).

### 3.6 KV Cache Storage

```c
int loff = l * p->seq_len * kv_dim;
float *key_cache_row = s->key_cache + loff + pos * kv_dim;
memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
```

The KV Cache is a 3D structure `(layer, seq_len, kv_dim)` flattened into 1D.
Each forward pass stores only the current position's K/V (512 dimensions each) for subsequent Attention.

### 3.7 Multihead Attention

```c
for (h = 0; h < n_heads; h++) {
    // Q·K dot product (iterate over all cached positions)
    for (t = 0; t <= pos; t++) {
        k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        score = dot_product(q, k, head_size=64);
        score /= sqrt(64);       // Scaled Dot-Product
        att[t] = score;
    }
    softmax(att, pos + 1);       // Convert to attention weights

    // Weighted sum of V
    for (t = 0; t <= pos; t++) {
        xb += att[t] * V[t];     // (64,)
    }
}
```

- 32 Q heads independently compute Attention
- GQA: every 4 Q heads share 1 KV head (via `h / kv_mul`)
- Scores divided by √head_size to prevent softmax saturation
- 32 heads each produce (64,), concatenated into (2048,)

#### Why Softmax Cannot Be Replaced by Argmax

Softmax output length varies with token position — at `pos`, softmax operates on `pos+1` scores:

```
pos=0:  scores = [Q₀·K₀]                        → softmax → [1.0]
pos=1:  scores = [Q₁·K₀, Q₁·K₁]                 → softmax → [0.3, 0.7]
pos=3:  scores = [Q₃·K₀, Q₃·K₁, Q₃·K₂, Q₃·K₃]  → softmax → [0.1, 0.2, 0.1, 0.6]
```

Each time only **one Q** (current token's) is used, but it dots with **all cached K's** (hence KV cache).

The softmax weights are used for a **weighted sum** of V vectors:

```
output = 0.1 × V₀ + 0.2 × V₁ + 0.1 × V₂ + 0.6 × V₃    → (64,) vector
```

If argmax were used instead, the result would be `[0, 0, 0, 1]` — only one V vector selected, all others discarded. This degrades **soft attention** (blend multiple positions) into **hard attention** (pick one position), severely harming model quality. The exp operation in softmax is therefore unavoidable in hardware.

#### Hardware Implementation: Softmax with LUT

Softmax itself cannot be a single LUT (output depends on all inputs), but can be decomposed into a 5-step pipeline:

```
Step 1: Find max           → comparator tree (no exp needed)
Step 2: xᵢ - max           → subtractor
Step 3: exp(xᵢ - max)      → LUT  ← same idea as sigmoid LUT
Step 4: Σ exp results       → adder tree
Step 5: each / sum          → divider (or multiply by 1/sum)
```

**exp() input range:** After subtracting max, inputs are always in (-∞, 0]. In practice, `e⁻¹⁶ ≈ 1.1×10⁻⁷` is negligible for float32. So the LUT only needs to cover **[-16, 0]**; values below -16 can be clamped to 0.

**Value ranges after max subtraction:**

```
Numerator:   exp(xᵢ - max)    ∈ (0, 1]       ← max element always = 1
Denominator: Σ exp(xⱼ - max)  ∈ [1, seq_len] ← worst case: all scores equal, each contributes 1
Result:      numerator / denom ∈ (0, 1]       ← all results sum to 1
```

Denominator upper bound = `seq_len` (model's max sequence length). For Llama 3.2 this could be 2048+.

**Numerical example — with vs without max subtraction:**

```
scores = [3, 1, -2],  max = 3

Without -max:                          With -max (x - 3 = [0, -2, -5]):
  e³  = 20.09                           e⁰  = 1.00
  e¹  = 2.72                            e⁻² = 0.14
  e⁻² = 0.14                            e⁻⁵ = 0.0067
  sum  = 22.95                           sum  = 1.147
  → [0.875, 0.119, 0.006]               → [0.872, 0.122, 0.006]  ← same result
```

Both produce the same softmax output (minor difference is rounding). But with -max, numerators stay in (0, 1] — no overflow risk, and the LUT input/output ranges are bounded.

**Note:** The softmax dimension varies per token position (`pos+1`), but this only affects Step 1/4 (how many values to compare/sum). The LUT itself is fixed regardless of sequence length.

---

## 4. Key Design Observations

| Design | Purpose |
|---|---|
| Int8 quantized matmul | Reduce memory, accelerate computation (90%+ of compute) |
| GQA (Grouped Query Attention) | KV Cache only needs 512 dims instead of 2048, saving 75% memory |
| RoPE positional encoding | Relative position info, cos/sin precomputable |
| Autoregressive + KV Cache | Only compute new token's K/V, read history from cache |
| Activations remain float32 | Nonlinear functions require floating-point precision; only matmul uses int8 |
