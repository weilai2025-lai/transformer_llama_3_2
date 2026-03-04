# runq.c Forward Pass Analysis Report

> **Target:** Llama 3.2 1B Int8 Quantized Inference
> **Scope:** `forward()` function (L390-532) — single-token forward pass

---

## 1. Operation Type Summary

The `forward()` function includes the following operations, categorized by precision and type:

| Operation | Type | Precision | Dimensions | Notes |
|---|---|---|---|---|
| QKV matmul (Wq,Wk,Wv) | Matrix×Vector | int8 | W(2048,2048)×x(2048,)→(2048,) | Largest compute |
| Output matmul (Wo) | Matrix×Vector | int8 | W(2048,2048)×x(2048,)→(2048,) | |
| FFN matmul (W1,W3) | Matrix×Vector | int8 | W(8192,2048)×x(2048,)→(8192,) | ~70% of compute |
| FFN matmul (W2) | Matrix×Vector | int8 | W(2048,8192)×x(8192,)→(2048,) | |
| Attention Q·K | Dot Product | float32 | (64,)·(64,)→scalar | Per head |
| Attention scores×V | Scalar×Vector Sum | float32 | Σ(scalar×(64,))→(64,) | Weighted sum |
| RMSNorm | Element-wise | float32 | (2048,) | 1/√(Σx²/n)×weight |
| RoPE | Pairwise (2D rotation) | float32 | 2 elements at a time | cos/sin |
| Softmax | Element-wise | float32 | (pos+1,) | exp→sum→divide |

**Key observations:**
- **Matrix-vector multiplications (int8)** account for 90%+ of compute, suitable for hardware acceleration (systolic array)
- **Nonlinear functions** (softmax, RoPE, RMSNorm) require float32, as int8 lacks exp, cos, sqrt operations

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
12-14. FFN matmul (W1,W3,W2)  int8×int8
15. Residual x += xb        float32        ❌
```

Per layer: **4 quantizations, 7 int8 matmuls**.

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

---

## 4. Key Design Observations

| Design | Purpose |
|---|---|
| Int8 quantized matmul | Reduce memory, accelerate computation (90%+ of compute) |
| GQA (Grouped Query Attention) | KV Cache only needs 512 dims instead of 2048, saving 75% memory |
| RoPE positional encoding | Relative position info, cos/sin precomputable |
| Autoregressive + KV Cache | Only compute new token's K/V, read history from cache |
| Activations remain float32 | Nonlinear functions require floating-point precision; only matmul uses int8 |
