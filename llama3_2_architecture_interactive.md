# Llama 3.2 3B 模型架構分析（互動版）

## 模型基本資訊

| 參數 | 值 |
|------|-----|
| model_type | llama |
| hidden_size | 3072 |
| num_hidden_layers | **28** |
| num_attention_heads | 24 |
| num_key_value_heads | 8 (GQA) |
| head_dim | 128 |
| intermediate_size | 8192 |
| vocab_size | 128256 |
| rope_theta | 500000.0 |

---

## 完整架構（可展開每一層）

### 🔹 輸入層

```
input_ids [batch, seq_len]
    ↓
embed_tokens (128256 → 3072)
    ↓
hidden_states [batch, seq_len, 3072]
```

---

### 🔁 Decoder Stack（28 層 Transformer Blocks）

> 💡 **點擊每一層可以展開詳細結構**

<details>
<summary><b>📦 Layer 0</b> — 第一層 Decoder</summary>

```
hidden_states [batch, seq, 3072]
        ↓
┌─────────────────────────────────────────────────────────────┐
│  input_layernorm (RMSNorm, eps=1e-05)                       │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Self-Attention (GQA)                                       │
│  ├── q_proj: 3072 → 3072 (24 heads × 128 dim)              │
│  ├── k_proj: 3072 → 1024 (8 heads × 128 dim)               │
│  ├── v_proj: 3072 → 1024 (8 heads × 128 dim)               │
│  ├── 🔄 RoPE: 套用到 Q, K (theta=500000)                    │
│  ├── ⚡ Attention: softmax(Q @ K^T / √128) @ V              │
│  └── o_proj: 3072 → 3072                                    │
└─────────────────────────────────────────────────────────────┘
        ↓ (+ residual)
┌─────────────────────────────────────────────────────────────┐
│  post_attention_layernorm (RMSNorm)                         │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  MLP (SwiGLU)                                               │
│  ├── gate_proj: 3072 → 8192                                 │
│  ├── up_proj:   3072 → 8192                                 │
│  ├── SiLU(gate) × up                                        │
│  └── down_proj: 8192 → 3072                                 │
└─────────────────────────────────────────────────────────────┘
        ↓ (+ residual)
hidden_states [batch, seq, 3072] → Layer 1
```
</details>

<details>
<summary><b>📦 Layer 1</b></summary>

```
hidden_states [batch, seq, 3072]
        ↓
┌─────────────────────────────────────────────────────────────┐
│  input_layernorm (RMSNorm)                                  │
│  → Self-Attention (GQA) → 🔄 RoPE → ⚡ Attention → o_proj    │
│  → (+ residual)                                             │
│  → post_attention_layernorm (RMSNorm)                       │
│  → MLP (SwiGLU): gate_proj, up_proj, SiLU×, down_proj       │
│  → (+ residual)                                             │
└─────────────────────────────────────────────────────────────┘
        ↓
hidden_states → Layer 2
```
</details>

<details>
<summary><b>📦 Layer 2</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 3
```
</details>

<details>
<summary><b>📦 Layer 3</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 4
```
</details>

<details>
<summary><b>📦 Layer 4</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 5
```
</details>

<details>
<summary><b>📦 Layer 5</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 6
```
</details>

<details>
<summary><b>📦 Layer 6</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 7
```
</details>

<details>
<summary><b>📦 Layer 7</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 8
```
</details>

<details>
<summary><b>📦 Layer 8</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 9
```
</details>

<details>
<summary><b>📦 Layer 9</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 10
```
</details>

<details>
<summary><b>📦 Layer 10</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 11
```
</details>

<details>
<summary><b>📦 Layer 11</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 12
```
</details>

<details>
<summary><b>📦 Layer 12</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 13
```
</details>

<details>
<summary><b>📦 Layer 13</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 14
```
</details>

<details>
<summary><b>📦 Layer 14</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 15
```
</details>

<details>
<summary><b>📦 Layer 15</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 16
```
</details>

<details>
<summary><b>📦 Layer 16</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 17
```
</details>

<details>
<summary><b>📦 Layer 17</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 18
```
</details>

<details>
<summary><b>📦 Layer 18</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 19
```
</details>

<details>
<summary><b>📦 Layer 19</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 20
```
</details>

<details>
<summary><b>📦 Layer 20</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 21
```
</details>

<details>
<summary><b>📦 Layer 21</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 22
```
</details>

<details>
<summary><b>📦 Layer 22</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 23
```
</details>

<details>
<summary><b>📦 Layer 23</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 24
```
</details>

<details>
<summary><b>📦 Layer 24</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 25
```
</details>

<details>
<summary><b>📦 Layer 25</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 26
```
</details>

<details>
<summary><b>📦 Layer 26</b></summary>

```
[Same structure as Layer 1]
hidden_states → Layer 27
```
</details>

<details>
<summary><b>📦 Layer 27</b> — 最後一層 Decoder</summary>

```
hidden_states [batch, seq, 3072]
        ↓
┌─────────────────────────────────────────────────────────────┐
│  input_layernorm (RMSNorm, eps=1e-05)                       │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Self-Attention (GQA)                                       │
│  ├── q_proj: 3072 → 3072 (24 heads × 128 dim)              │
│  ├── k_proj: 3072 → 1024 (8 heads × 128 dim)               │
│  ├── v_proj: 3072 → 1024 (8 heads × 128 dim)               │
│  ├── 🔄 RoPE: 套用到 Q, K (theta=500000)                    │
│  ├── ⚡ Attention: softmax(Q @ K^T / √128) @ V              │
│  └── o_proj: 3072 → 3072                                    │
└─────────────────────────────────────────────────────────────┘
        ↓ (+ residual)
┌─────────────────────────────────────────────────────────────┐
│  post_attention_layernorm (RMSNorm)                         │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  MLP (SwiGLU)                                               │
│  ├── gate_proj: 3072 → 8192                                 │
│  ├── up_proj:   3072 → 8192                                 │
│  ├── SiLU(gate) × up                                        │
│  └── down_proj: 8192 → 3072                                 │
└─────────────────────────────────────────────────────────────┘
        ↓ (+ residual)
hidden_states [batch, seq, 3072] → 輸出層
```
</details>

---

### 🔹 輸出層

```
hidden_states [batch, seq, 3072]
        ↓
┌─────────────────────────────────────────────────────────────┐
│  Final RMSNorm (eps=1e-05)                                  │
└─────────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────────┐
│  lm_head (Linear: 3072 → 128256)                            │
└─────────────────────────────────────────────────────────────┘
        ↓
logits [batch, seq, 128256]
```

---

## 關鍵組件說明

### 1. RoPE (Rotary Position Embedding)
- **位置**：在 Q, K 計算後、Attention Score 計算前
- **作用**：將位置資訊編碼到 query 和 key 中
- **參數**：`rope_theta = 500000.0`

### 2. GQA (Grouped Query Attention)
- **設計**：24 個 query heads 共享 8 個 key/value heads
- **比例**：每 3 個 Q heads 共享 1 個 KV head
- **優點**：減少 KV cache 記憶體使用

### 3. Attention Score 計算
```
Attention(Q, K, V) = softmax(Q @ K^T / √head_dim) @ V
```
- head_dim = 128

### 4. SwiGLU MLP
```
MLP(x) = down_proj(SiLU(gate_proj(x)) × up_proj(x))
```
- intermediate_size = 8192

---

## 參數量統計

| 組件 | 每層參數量 | 總參數量 |
|------|-----------|----------|
| embed_tokens | - | 394M |
| q_proj (×28) | 9.4M | 264M |
| k_proj (×28) | 3.1M | 88M |
| v_proj (×28) | 3.1M | 88M |
| o_proj (×28) | 9.4M | 264M |
| gate_proj (×28) | 25.2M | 705M |
| up_proj (×28) | 25.2M | 705M |
| down_proj (×28) | 25.2M | 705M |
| lm_head | - | 394M |
| **Total** | - | **3.6B** |
