# Llama 3.2 3B 完整架構圖（28 層）

## 模型基本資訊

| 參數 | 值 |
|------|-----|
| hidden_size | 3072 |
| num_hidden_layers | **28** |
| num_attention_heads | 24 (GQA: 8 KV heads) |
| head_dim | 128 |
| intermediate_size | 8192 |
| vocab_size | 128256 |
| rope_theta | 500000.0 |

---

## 完整架構圖

> ⚠️ 此圖包含完整 28 層，請向下滾動查看所有層

```mermaid
graph TD
    %% ==================== 輸入層 ====================
    subgraph Input["🔹 輸入層"]
        IN[input_ids<br/>batch, seq_len]
    end

    subgraph Embed["🔹 Embedding"]
        EMB[embed_tokens<br/>128256 → 3072]
    end

    IN --> EMB

    %% ==================== Layer 0 ====================
    subgraph L0["📦 Layer 0"]
        L0_norm1[RMSNorm]
        subgraph L0_attn["Self-Attention"]
            L0_qkv[Q K V proj]
            L0_rope["🔄 RoPE"]
            L0_score["⚡ Attention"]
            L0_out[o_proj]
        end
        L0_norm2[RMSNorm]
        subgraph L0_mlp["MLP"]
            L0_gate[gate + up]
            L0_silu[SiLU ×]
            L0_down[down_proj]
        end
    end
    EMB --> L0_norm1
    L0_norm1 --> L0_qkv --> L0_rope --> L0_score --> L0_out
    L0_out -->|+res| L0_norm2
    L0_norm2 --> L0_gate --> L0_silu --> L0_down

    %% ==================== Layer 1 ====================
    subgraph L1["📦 Layer 1"]
        L1_norm1[RMSNorm]
        subgraph L1_attn["Self-Attention"]
            L1_qkv[Q K V proj]
            L1_rope["🔄 RoPE"]
            L1_score["⚡ Attention"]
            L1_out[o_proj]
        end
        L1_norm2[RMSNorm]
        subgraph L1_mlp["MLP"]
            L1_gate[gate + up]
            L1_silu[SiLU ×]
            L1_down[down_proj]
        end
    end
    L0_down -->|+res| L1_norm1
    L1_norm1 --> L1_qkv --> L1_rope --> L1_score --> L1_out
    L1_out -->|+res| L1_norm2
    L1_norm2 --> L1_gate --> L1_silu --> L1_down

    %% ==================== Layer 2 ====================
    subgraph L2["📦 Layer 2"]
        L2_norm1[RMSNorm]
        subgraph L2_attn["Self-Attention"]
            L2_qkv[Q K V proj]
            L2_rope["🔄 RoPE"]
            L2_score["⚡ Attention"]
            L2_out[o_proj]
        end
        L2_norm2[RMSNorm]
        subgraph L2_mlp["MLP"]
            L2_gate[gate + up]
            L2_silu[SiLU ×]
            L2_down[down_proj]
        end
    end
    L1_down -->|+res| L2_norm1
    L2_norm1 --> L2_qkv --> L2_rope --> L2_score --> L2_out
    L2_out -->|+res| L2_norm2
    L2_norm2 --> L2_gate --> L2_silu --> L2_down

    %% ==================== Layer 3 ====================
    subgraph L3["📦 Layer 3"]
        L3_norm1[RMSNorm]
        subgraph L3_attn["Self-Attention"]
            L3_qkv[Q K V proj]
            L3_rope["🔄 RoPE"]
            L3_score["⚡ Attention"]
            L3_out[o_proj]
        end
        L3_norm2[RMSNorm]
        subgraph L3_mlp["MLP"]
            L3_gate[gate + up]
            L3_silu[SiLU ×]
            L3_down[down_proj]
        end
    end
    L2_down -->|+res| L3_norm1
    L3_norm1 --> L3_qkv --> L3_rope --> L3_score --> L3_out
    L3_out -->|+res| L3_norm2
    L3_norm2 --> L3_gate --> L3_silu --> L3_down

    %% ==================== Layer 4 ====================
    subgraph L4["📦 Layer 4"]
        L4_norm1[RMSNorm]
        subgraph L4_attn["Self-Attention"]
            L4_qkv[Q K V proj]
            L4_rope["🔄 RoPE"]
            L4_score["⚡ Attention"]
            L4_out[o_proj]
        end
        L4_norm2[RMSNorm]
        subgraph L4_mlp["MLP"]
            L4_gate[gate + up]
            L4_silu[SiLU ×]
            L4_down[down_proj]
        end
    end
    L3_down -->|+res| L4_norm1
    L4_norm1 --> L4_qkv --> L4_rope --> L4_score --> L4_out
    L4_out -->|+res| L4_norm2
    L4_norm2 --> L4_gate --> L4_silu --> L4_down

    %% ==================== Layer 5 ====================
    subgraph L5["📦 Layer 5"]
        L5_norm1[RMSNorm]
        subgraph L5_attn["Self-Attention"]
            L5_qkv[Q K V proj]
            L5_rope["🔄 RoPE"]
            L5_score["⚡ Attention"]
            L5_out[o_proj]
        end
        L5_norm2[RMSNorm]
        subgraph L5_mlp["MLP"]
            L5_gate[gate + up]
            L5_silu[SiLU ×]
            L5_down[down_proj]
        end
    end
    L4_down -->|+res| L5_norm1
    L5_norm1 --> L5_qkv --> L5_rope --> L5_score --> L5_out
    L5_out -->|+res| L5_norm2
    L5_norm2 --> L5_gate --> L5_silu --> L5_down

    %% ==================== Layer 6 ====================
    subgraph L6["📦 Layer 6"]
        L6_norm1[RMSNorm]
        subgraph L6_attn["Self-Attention"]
            L6_qkv[Q K V proj]
            L6_rope["🔄 RoPE"]
            L6_score["⚡ Attention"]
            L6_out[o_proj]
        end
        L6_norm2[RMSNorm]
        subgraph L6_mlp["MLP"]
            L6_gate[gate + up]
            L6_silu[SiLU ×]
            L6_down[down_proj]
        end
    end
    L5_down -->|+res| L6_norm1
    L6_norm1 --> L6_qkv --> L6_rope --> L6_score --> L6_out
    L6_out -->|+res| L6_norm2
    L6_norm2 --> L6_gate --> L6_silu --> L6_down

    %% ==================== Layer 7 ====================
    subgraph L7["📦 Layer 7"]
        L7_norm1[RMSNorm]
        subgraph L7_attn["Self-Attention"]
            L7_qkv[Q K V proj]
            L7_rope["🔄 RoPE"]
            L7_score["⚡ Attention"]
            L7_out[o_proj]
        end
        L7_norm2[RMSNorm]
        subgraph L7_mlp["MLP"]
            L7_gate[gate + up]
            L7_silu[SiLU ×]
            L7_down[down_proj]
        end
    end
    L6_down -->|+res| L7_norm1
    L7_norm1 --> L7_qkv --> L7_rope --> L7_score --> L7_out
    L7_out -->|+res| L7_norm2
    L7_norm2 --> L7_gate --> L7_silu --> L7_down

    %% ==================== Layer 8 ====================
    subgraph L8["📦 Layer 8"]
        L8_norm1[RMSNorm]
        subgraph L8_attn["Self-Attention"]
            L8_qkv[Q K V proj]
            L8_rope["🔄 RoPE"]
            L8_score["⚡ Attention"]
            L8_out[o_proj]
        end
        L8_norm2[RMSNorm]
        subgraph L8_mlp["MLP"]
            L8_gate[gate + up]
            L8_silu[SiLU ×]
            L8_down[down_proj]
        end
    end
    L7_down -->|+res| L8_norm1
    L8_norm1 --> L8_qkv --> L8_rope --> L8_score --> L8_out
    L8_out -->|+res| L8_norm2
    L8_norm2 --> L8_gate --> L8_silu --> L8_down

    %% ==================== Layer 9 ====================
    subgraph L9["📦 Layer 9"]
        L9_norm1[RMSNorm]
        subgraph L9_attn["Self-Attention"]
            L9_qkv[Q K V proj]
            L9_rope["🔄 RoPE"]
            L9_score["⚡ Attention"]
            L9_out[o_proj]
        end
        L9_norm2[RMSNorm]
        subgraph L9_mlp["MLP"]
            L9_gate[gate + up]
            L9_silu[SiLU ×]
            L9_down[down_proj]
        end
    end
    L8_down -->|+res| L9_norm1
    L9_norm1 --> L9_qkv --> L9_rope --> L9_score --> L9_out
    L9_out -->|+res| L9_norm2
    L9_norm2 --> L9_gate --> L9_silu --> L9_down

    %% ==================== Layer 10 ====================
    subgraph L10["📦 Layer 10"]
        L10_norm1[RMSNorm]
        subgraph L10_attn["Self-Attention"]
            L10_qkv[Q K V proj]
            L10_rope["🔄 RoPE"]
            L10_score["⚡ Attention"]
            L10_out[o_proj]
        end
        L10_norm2[RMSNorm]
        subgraph L10_mlp["MLP"]
            L10_gate[gate + up]
            L10_silu[SiLU ×]
            L10_down[down_proj]
        end
    end
    L9_down -->|+res| L10_norm1
    L10_norm1 --> L10_qkv --> L10_rope --> L10_score --> L10_out
    L10_out -->|+res| L10_norm2
    L10_norm2 --> L10_gate --> L10_silu --> L10_down

    %% ==================== Layer 11 ====================
    subgraph L11["📦 Layer 11"]
        L11_norm1[RMSNorm]
        subgraph L11_attn["Self-Attention"]
            L11_qkv[Q K V proj]
            L11_rope["🔄 RoPE"]
            L11_score["⚡ Attention"]
            L11_out[o_proj]
        end
        L11_norm2[RMSNorm]
        subgraph L11_mlp["MLP"]
            L11_gate[gate + up]
            L11_silu[SiLU ×]
            L11_down[down_proj]
        end
    end
    L10_down -->|+res| L11_norm1
    L11_norm1 --> L11_qkv --> L11_rope --> L11_score --> L11_out
    L11_out -->|+res| L11_norm2
    L11_norm2 --> L11_gate --> L11_silu --> L11_down

    %% ==================== Layer 12 ====================
    subgraph L12["📦 Layer 12"]
        L12_norm1[RMSNorm]
        subgraph L12_attn["Self-Attention"]
            L12_qkv[Q K V proj]
            L12_rope["🔄 RoPE"]
            L12_score["⚡ Attention"]
            L12_out[o_proj]
        end
        L12_norm2[RMSNorm]
        subgraph L12_mlp["MLP"]
            L12_gate[gate + up]
            L12_silu[SiLU ×]
            L12_down[down_proj]
        end
    end
    L11_down -->|+res| L12_norm1
    L12_norm1 --> L12_qkv --> L12_rope --> L12_score --> L12_out
    L12_out -->|+res| L12_norm2
    L12_norm2 --> L12_gate --> L12_silu --> L12_down

    %% ==================== Layer 13 ====================
    subgraph L13["📦 Layer 13"]
        L13_norm1[RMSNorm]
        subgraph L13_attn["Self-Attention"]
            L13_qkv[Q K V proj]
            L13_rope["🔄 RoPE"]
            L13_score["⚡ Attention"]
            L13_out[o_proj]
        end
        L13_norm2[RMSNorm]
        subgraph L13_mlp["MLP"]
            L13_gate[gate + up]
            L13_silu[SiLU ×]
            L13_down[down_proj]
        end
    end
    L12_down -->|+res| L13_norm1
    L13_norm1 --> L13_qkv --> L13_rope --> L13_score --> L13_out
    L13_out -->|+res| L13_norm2
    L13_norm2 --> L13_gate --> L13_silu --> L13_down

    %% ==================== Layer 14 ====================
    subgraph L14["📦 Layer 14"]
        L14_norm1[RMSNorm]
        subgraph L14_attn["Self-Attention"]
            L14_qkv[Q K V proj]
            L14_rope["🔄 RoPE"]
            L14_score["⚡ Attention"]
            L14_out[o_proj]
        end
        L14_norm2[RMSNorm]
        subgraph L14_mlp["MLP"]
            L14_gate[gate + up]
            L14_silu[SiLU ×]
            L14_down[down_proj]
        end
    end
    L13_down -->|+res| L14_norm1
    L14_norm1 --> L14_qkv --> L14_rope --> L14_score --> L14_out
    L14_out -->|+res| L14_norm2
    L14_norm2 --> L14_gate --> L14_silu --> L14_down

    %% ==================== Layer 15 ====================
    subgraph L15["📦 Layer 15"]
        L15_norm1[RMSNorm]
        subgraph L15_attn["Self-Attention"]
            L15_qkv[Q K V proj]
            L15_rope["🔄 RoPE"]
            L15_score["⚡ Attention"]
            L15_out[o_proj]
        end
        L15_norm2[RMSNorm]
        subgraph L15_mlp["MLP"]
            L15_gate[gate + up]
            L15_silu[SiLU ×]
            L15_down[down_proj]
        end
    end
    L14_down -->|+res| L15_norm1
    L15_norm1 --> L15_qkv --> L15_rope --> L15_score --> L15_out
    L15_out -->|+res| L15_norm2
    L15_norm2 --> L15_gate --> L15_silu --> L15_down

    %% ==================== Layer 16 ====================
    subgraph L16["📦 Layer 16"]
        L16_norm1[RMSNorm]
        subgraph L16_attn["Self-Attention"]
            L16_qkv[Q K V proj]
            L16_rope["🔄 RoPE"]
            L16_score["⚡ Attention"]
            L16_out[o_proj]
        end
        L16_norm2[RMSNorm]
        subgraph L16_mlp["MLP"]
            L16_gate[gate + up]
            L16_silu[SiLU ×]
            L16_down[down_proj]
        end
    end
    L15_down -->|+res| L16_norm1
    L16_norm1 --> L16_qkv --> L16_rope --> L16_score --> L16_out
    L16_out -->|+res| L16_norm2
    L16_norm2 --> L16_gate --> L16_silu --> L16_down

    %% ==================== Layer 17 ====================
    subgraph L17["📦 Layer 17"]
        L17_norm1[RMSNorm]
        subgraph L17_attn["Self-Attention"]
            L17_qkv[Q K V proj]
            L17_rope["🔄 RoPE"]
            L17_score["⚡ Attention"]
            L17_out[o_proj]
        end
        L17_norm2[RMSNorm]
        subgraph L17_mlp["MLP"]
            L17_gate[gate + up]
            L17_silu[SiLU ×]
            L17_down[down_proj]
        end
    end
    L16_down -->|+res| L17_norm1
    L17_norm1 --> L17_qkv --> L17_rope --> L17_score --> L17_out
    L17_out -->|+res| L17_norm2
    L17_norm2 --> L17_gate --> L17_silu --> L17_down

    %% ==================== Layer 18 ====================
    subgraph L18["📦 Layer 18"]
        L18_norm1[RMSNorm]
        subgraph L18_attn["Self-Attention"]
            L18_qkv[Q K V proj]
            L18_rope["🔄 RoPE"]
            L18_score["⚡ Attention"]
            L18_out[o_proj]
        end
        L18_norm2[RMSNorm]
        subgraph L18_mlp["MLP"]
            L18_gate[gate + up]
            L18_silu[SiLU ×]
            L18_down[down_proj]
        end
    end
    L17_down -->|+res| L18_norm1
    L18_norm1 --> L18_qkv --> L18_rope --> L18_score --> L18_out
    L18_out -->|+res| L18_norm2
    L18_norm2 --> L18_gate --> L18_silu --> L18_down

    %% ==================== Layer 19 ====================
    subgraph L19["📦 Layer 19"]
        L19_norm1[RMSNorm]
        subgraph L19_attn["Self-Attention"]
            L19_qkv[Q K V proj]
            L19_rope["🔄 RoPE"]
            L19_score["⚡ Attention"]
            L19_out[o_proj]
        end
        L19_norm2[RMSNorm]
        subgraph L19_mlp["MLP"]
            L19_gate[gate + up]
            L19_silu[SiLU ×]
            L19_down[down_proj]
        end
    end
    L18_down -->|+res| L19_norm1
    L19_norm1 --> L19_qkv --> L19_rope --> L19_score --> L19_out
    L19_out -->|+res| L19_norm2
    L19_norm2 --> L19_gate --> L19_silu --> L19_down

    %% ==================== Layer 20 ====================
    subgraph L20["📦 Layer 20"]
        L20_norm1[RMSNorm]
        subgraph L20_attn["Self-Attention"]
            L20_qkv[Q K V proj]
            L20_rope["🔄 RoPE"]
            L20_score["⚡ Attention"]
            L20_out[o_proj]
        end
        L20_norm2[RMSNorm]
        subgraph L20_mlp["MLP"]
            L20_gate[gate + up]
            L20_silu[SiLU ×]
            L20_down[down_proj]
        end
    end
    L19_down -->|+res| L20_norm1
    L20_norm1 --> L20_qkv --> L20_rope --> L20_score --> L20_out
    L20_out -->|+res| L20_norm2
    L20_norm2 --> L20_gate --> L20_silu --> L20_down

    %% ==================== Layer 21 ====================
    subgraph L21["📦 Layer 21"]
        L21_norm1[RMSNorm]
        subgraph L21_attn["Self-Attention"]
            L21_qkv[Q K V proj]
            L21_rope["🔄 RoPE"]
            L21_score["⚡ Attention"]
            L21_out[o_proj]
        end
        L21_norm2[RMSNorm]
        subgraph L21_mlp["MLP"]
            L21_gate[gate + up]
            L21_silu[SiLU ×]
            L21_down[down_proj]
        end
    end
    L20_down -->|+res| L21_norm1
    L21_norm1 --> L21_qkv --> L21_rope --> L21_score --> L21_out
    L21_out -->|+res| L21_norm2
    L21_norm2 --> L21_gate --> L21_silu --> L21_down

    %% ==================== Layer 22 ====================
    subgraph L22["📦 Layer 22"]
        L22_norm1[RMSNorm]
        subgraph L22_attn["Self-Attention"]
            L22_qkv[Q K V proj]
            L22_rope["🔄 RoPE"]
            L22_score["⚡ Attention"]
            L22_out[o_proj]
        end
        L22_norm2[RMSNorm]
        subgraph L22_mlp["MLP"]
            L22_gate[gate + up]
            L22_silu[SiLU ×]
            L22_down[down_proj]
        end
    end
    L21_down -->|+res| L22_norm1
    L22_norm1 --> L22_qkv --> L22_rope --> L22_score --> L22_out
    L22_out -->|+res| L22_norm2
    L22_norm2 --> L22_gate --> L22_silu --> L22_down

    %% ==================== Layer 23 ====================
    subgraph L23["📦 Layer 23"]
        L23_norm1[RMSNorm]
        subgraph L23_attn["Self-Attention"]
            L23_qkv[Q K V proj]
            L23_rope["🔄 RoPE"]
            L23_score["⚡ Attention"]
            L23_out[o_proj]
        end
        L23_norm2[RMSNorm]
        subgraph L23_mlp["MLP"]
            L23_gate[gate + up]
            L23_silu[SiLU ×]
            L23_down[down_proj]
        end
    end
    L22_down -->|+res| L23_norm1
    L23_norm1 --> L23_qkv --> L23_rope --> L23_score --> L23_out
    L23_out -->|+res| L23_norm2
    L23_norm2 --> L23_gate --> L23_silu --> L23_down

    %% ==================== Layer 24 ====================
    subgraph L24["📦 Layer 24"]
        L24_norm1[RMSNorm]
        subgraph L24_attn["Self-Attention"]
            L24_qkv[Q K V proj]
            L24_rope["🔄 RoPE"]
            L24_score["⚡ Attention"]
            L24_out[o_proj]
        end
        L24_norm2[RMSNorm]
        subgraph L24_mlp["MLP"]
            L24_gate[gate + up]
            L24_silu[SiLU ×]
            L24_down[down_proj]
        end
    end
    L23_down -->|+res| L24_norm1
    L24_norm1 --> L24_qkv --> L24_rope --> L24_score --> L24_out
    L24_out -->|+res| L24_norm2
    L24_norm2 --> L24_gate --> L24_silu --> L24_down

    %% ==================== Layer 25 ====================
    subgraph L25["📦 Layer 25"]
        L25_norm1[RMSNorm]
        subgraph L25_attn["Self-Attention"]
            L25_qkv[Q K V proj]
            L25_rope["🔄 RoPE"]
            L25_score["⚡ Attention"]
            L25_out[o_proj]
        end
        L25_norm2[RMSNorm]
        subgraph L25_mlp["MLP"]
            L25_gate[gate + up]
            L25_silu[SiLU ×]
            L25_down[down_proj]
        end
    end
    L24_down -->|+res| L25_norm1
    L25_norm1 --> L25_qkv --> L25_rope --> L25_score --> L25_out
    L25_out -->|+res| L25_norm2
    L25_norm2 --> L25_gate --> L25_silu --> L25_down

    %% ==================== Layer 26 ====================
    subgraph L26["📦 Layer 26"]
        L26_norm1[RMSNorm]
        subgraph L26_attn["Self-Attention"]
            L26_qkv[Q K V proj]
            L26_rope["🔄 RoPE"]
            L26_score["⚡ Attention"]
            L26_out[o_proj]
        end
        L26_norm2[RMSNorm]
        subgraph L26_mlp["MLP"]
            L26_gate[gate + up]
            L26_silu[SiLU ×]
            L26_down[down_proj]
        end
    end
    L25_down -->|+res| L26_norm1
    L26_norm1 --> L26_qkv --> L26_rope --> L26_score --> L26_out
    L26_out -->|+res| L26_norm2
    L26_norm2 --> L26_gate --> L26_silu --> L26_down

    %% ==================== Layer 27 (最後一層) ====================
    subgraph L27["📦 Layer 27 (Final)"]
        L27_norm1[RMSNorm]
        subgraph L27_attn["Self-Attention"]
            L27_qkv[Q K V proj]
            L27_rope["🔄 RoPE"]
            L27_score["⚡ Attention"]
            L27_out[o_proj]
        end
        L27_norm2[RMSNorm]
        subgraph L27_mlp["MLP"]
            L27_gate[gate + up]
            L27_silu[SiLU ×]
            L27_down[down_proj]
        end
    end
    L26_down -->|+res| L27_norm1
    L27_norm1 --> L27_qkv --> L27_rope --> L27_score --> L27_out
    L27_out -->|+res| L27_norm2
    L27_norm2 --> L27_gate --> L27_silu --> L27_down

    %% ==================== 輸出層 ====================
    subgraph Output["🔹 輸出層"]
        FINAL_NORM[Final RMSNorm]
        LM_HEAD[lm_head<br/>3072 → 128256]
        OUT[logits]
    end
    L27_down -->|+res| FINAL_NORM
    FINAL_NORM --> LM_HEAD --> OUT

    %% 樣式
    style L0_rope fill:#e1f5fe
    style L0_score fill:#fff3e0
    style L0_silu fill:#f3e5f5
```

---

## 關鍵組件說明

| 組件 | 說明 |
|------|------|
| **RoPE** 🔄 | Rotary Position Embedding，套用到 Q 和 K |
| **Attention** ⚡ | `softmax(Q @ K^T / √128) @ V` |
| **SiLU ×** | `SiLU(gate) × up`，即 SwiGLU activation |
| **+res** | Residual connection（殘差連接） |
| **GQA** | 24 Q heads 共享 8 KV heads |
