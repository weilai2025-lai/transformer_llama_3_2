# Transformer Attention (GQA) Prefill Walkthrough  
## Example: "I love you → Thank you" (Aligned with LLaMA 3.2 Architecture)

---

## 0. Model Configuration (Fixed Assumptions for This Note)

- Hidden size = 3072  
- Query heads = 24  
- Key / Value heads = 8  
- Head dimension = 128  

Verification:
- Q total dimension = 24 × 128 = 3072  
- K / V total dimension = 8 × 128 = 1024  

---

## 1. Input Prompt (Prefill)

Prompt:
```
I love you
```

Token count (sequence length):
```
seq_len = 3
```

Tensor after embedding (layer input):
```
X.shape = [3, 3072]
```

---

## 2. Pre-Norm (RMSNorm, Before Attention)

LLaMA 3.2 uses the **Pre-LayerNorm (Pre-LN) architecture**,  
so RMSNorm is applied before any attention computation.

```
X_norm = RMSNorm(X)
X_norm.shape = [3, 3072]
```

Key points:
- Normalization occurs at the **very beginning of attention**
- All subsequent Q / K / V projections use `X_norm`

---

## 3. Q / K / V Linear Projections (Prefill Stage)

Projection matrix dimensions:
```
Wq.shape = [3072, 3072]
Wk.shape = [3072, 1024]
Wv.shape = [3072, 1024]
```

Projection results (note: input is X_norm):
```
Q_raw = X_norm · Wq   → [3, 3072]
K_raw = X_norm · Wk   → [3, 1024]
V_raw = X_norm · Wv   → [3, 1024]
```

---

## 4. Reshape into Multi-Head Format

Query:
```
Q.shape = [3, 24, 128]
```

Key:
```
K.shape = [3, 8, 128]
```

Value:
```
V.shape = [3, 8, 128]
```

---

## 5. RoPE (Rotary Positional Embedding)

RoPE is **only applied to Query and Key**, not to Value.

```
Q_rot = RoPE(Q)
K_rot = RoPE(K)
```

Key points:
- RoPE occurs **after linear projection and reshape**
- RoPE occurs **before attention score computation**
- V contains no positional information

---

## 6. GQA (Grouped Query Attention) Head Mapping

```
Q head 0,1,2     → K/V head 0
Q head 3,4,5     → K/V head 1
Q head 6,7,8     → K/V head 2
Q head 9,10,11   → K/V head 3
Q head 12,13,14  → K/V head 4
Q head 15,16,17  → K/V head 5
Q head 18,19,20  → K/V head 6
Q head 21,22,23  → K/V head 7
```

Notes:
- Multiple Q heads share the same K / V
- Q heads are **not summed together**
- They only share the KV cache (memory optimization)

---

## 7. Attention Computation for a Single Head

Example: Q head 0 (corresponding to K/V head 0)

```
Q0.shape = [3, 128]
K0.shape = [3, 128]
```

Attention score:
```
score0 = Q0 · K0ᵀ
score0.shape = [3, 3]
```

Scaling + Softmax (each head independently):
```
attn0 = softmax(score0 / √128)
attn0.shape = [3, 3]
```

Multiply by Value:
```
V0.shape = [3, 128]
out0 = attn0 · V0
out0.shape = [3, 128]
```

---

## 8. Output After All Heads Complete

```
out0  ∈ [3, 128]
out1  ∈ [3, 128]
...
out23 ∈ [3, 128]
```

Concatenation (concat, not addition):
```
out_concat.shape = [3, 3072]
```

---

## 9. Core Flow Summary (Faithfully Aligned with LLaMA 3.2)

```
X
→ RMSNorm
→ Linear (Wq, Wk, Wv)
→ Reshape to heads
→ RoPE (Q, K only)
→ QKᵀ / √d
→ Softmax (per head)
→ × V
→ Concat heads
```

---
