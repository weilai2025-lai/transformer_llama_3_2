# runq.c 導讀筆記

> 這份筆記記錄我逐行學習 `runq.c`（Int8 量化版 Llama 3.2 推論程式）的過程。
> 以 **Llama 3.2 1B** 的參數為基準（來源：`Llama-3.2-1B-Instruct-bnb-4bit/config.json`）
> 搭配可視化工具：`code_visualizer/index.html`

---

## C 語言基礎速查

### 指標三兄弟：`*`、`&`、`->`

用「房子」比喻：

| 符號 | 意思 | 比喻 |
|---|---|---|
| `*` | 「這是一個地址」（指標） | 手上拿著一張寫了門牌號碼的紙條 |
| `&` | 「取得地址」 | 去看這棟房子的門牌號碼是多少 |
| `->` | 「通過地址，進去拿東西」 | 照著紙條走進房子，打開某個抽屜 |

### 什麼時候用指標、什麼時候不用？

| 情況 | 用法 | 原因 |
|---|---|---|
| 小東西（int, float），只讀取 | `int token` 直接傳值 | 只有 4 bytes，複製很便宜 |
| 大東西（struct），或需要修改原始值 | `Transformer* t` 傳指標 | 避免複製幾 GB 的資料 |

---

## forward() 函式導讀

### 位置：runq.c L390-532
### 作用：整個模型的前向傳播（推論核心）

---

### ① 函式宣告（L390）

```c
float* forward(Transformer* transformer, int token, int pos) {
```

| 部分 | 寫法 | 意思 |
|---|---|---|
| 回傳型別 | `float*` | 回傳一個 float 陣列的地址（logits） |
| 函式名稱 | `forward` | 前向傳播 |
| 參數 1 | `Transformer* transformer` | 整個模型（指標，避免複製） |
| 參數 2 | `int token` | 當前 token ID（嵌入表的索引/行號） |
| 參數 3 | `int pos` | 這個 token 在序列中的位置（第幾個字） |

**`token` vs `pos` 的差別：**
- `token` = 「這是什麼字」→ 用來查 embedding table
- `pos` = 「這個字排第幾」→ 後面 RoPE 用來注入位置資訊
- 兩者獨立，不是一起查表的

---

### ② 建立捷徑變數（L393-403）

```c
Config* p = &transformer->config;
TransformerWeights* w = &transformer->weights;
RunState* s = &transformer->state;
```

以 `Config* p = &transformer->config;` 為例，從右讀到左：

1. `transformer` → 手上有房子的地址
2. `transformer->config` → 照地址走進去，打開「config」抽屜
3. `&transformer->config` → 看這個抽屜的地址是多少
4. `Config* p =` → 把地址抄到一張叫 `p` 的紙條上

**三行的目的只是建立捷徑：**
- `p` → Config 的捷徑（之後 `p->dim` 取代 `transformer->config.dim`）
- `w` → 權重的捷徑
- `s` → 運算暫存區的捷徑

```c
float *x = s->x;           // x = 當前的 activation 向量（dim 維）
int dim = p->dim;           // dim = 2048（模型維度）
int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;  // KV 頭的維度
int kv_mul = p->n_heads / p->n_kv_heads;  // KV 共享倍數（GQA）
int hidden_dim = p->hidden_dim;   // FFN 隱藏層維度 (8192)
int head_size = dim / p->n_heads; // 每個 head 的維度 (64)
```

這些也都是捷徑，把常用數值先取出來存在區域變數裡，後面方便使用。

---

### Q&A 補充

#### `&transformer->config` 是「地址的地址」嗎？

不是。是「透過 transformer 的地址走進去，再看裡面 config 這個子物件的地址在哪」。

#### `float *x = s->x` 為什麼不用加 `&`？

因為 `s->x` 在 RunState 的定義中就已經是 `float*`（地址），拿出來就已經是地址了，不需要再取一次。

判斷標準：
| 情況 | 本來是什麼 | 需要 `&` 嗎 |
|---|---|---|
| `s->x` | `float*`（已經是地址） | ❌ 不用 |
| `transformer->config` | `Config`（是值本身） | ✅ 要加 `&` |

#### RunState 是什麼？

推論過程中所有的**中間暫存區**（工作桌面），包含 x, q, k, v, att, logits, key_cache, value_cache 等。

未來 CPU/FPGA 交互時，這些中間暫存就是兩邊要傳遞的資料。

#### Llama 3.2 1B 的實際參數數值（from config.json）

| 變數 | 算式 | 數值 |
|---|---|---|
| `dim` | `p->dim` | **2048** |
| `n_heads` | `p->n_heads` | **32** |
| `n_kv_heads` | `p->n_kv_heads` | **8** |
| `head_size` | `dim / n_heads` | **64** |
| `kv_dim` | `dim * n_kv_heads / n_heads` | **512** |
| `kv_mul` | `n_heads / n_kv_heads` | **4** |
| `hidden_dim` | `p->hidden_dim` | **8192** |

#### 回傳的 `float*` 有多大？

回傳的是 `logits` 陣列，大小 = **vocab_size = 128256** 個 float。
每個 float 對應一個 token 的分數，Sampler 會從中選出下一個 token。

#### logits 是一維陣列嗎？

對！大小 = vocab_size = 128256，每個索引對應一個 token ID 的分數，分數最高的就是模型認為最可能的下一個字。

#### n_heads = 32 是 Q+K+V 加起來嗎？

**不是！** n_heads = 32 只是 Q 自己的 head 數量。

| | head 數量 | 維度 |
|---|---|---|
| Q (Query) | **32** heads | 32 × 64 = 2048 (=dim) |
| K (Key) | **8** heads | 8 × 64 = 512 (=kv_dim) |
| V (Value) | **8** heads | 8 × 64 = 512 (=kv_dim) |

這是 **GQA（Grouped Query Attention）**：每 4 個 Q head 共用 1 個 KV head（`kv_mul=4`）。
目的：減少 KV Cache 的記憶體用量，但 Q 保持多頭不影響表達能力。

#### GQA 具體範例

沒有 GQA 時（MHA）：每個 Q head 搭配專屬的 KV head，KV Cache = 32×64 = 2048 維

有 GQA 時（Llama 3.2）：每 4 個 Q head 共用 1 組 KV：
```
Q head 0~3   → 共用 K head 0, V head 0
Q head 4~7   → 共用 K head 1, V head 1
...
Q head 28~31 → 共用 K head 7, V head 7
```
KV Cache = 8×64 = 512 維，**省 75% 記憶體**

程式碼中用 `h / kv_mul`（整數除法）實現：h=0,1,2,3 除以 4 都 = 0 → 共用 K head 0

---

### QKV 維度變化（以 1 個 token 為例）

> **重要：** `runq.c` 是 autoregressive 推論，一次只處理 **1 個 token**，所以 QKV 都是一維向量。

```
輸入：token ID = 5678

1. Embedding   → x (2048,)
2. RMSNorm     → xb (2048,)
3. QKV 投影：
   xb × Wq (2048,2048) → q (2048,)    # Q 完整維度
   xb × Wk (2048,512)  → k (512,)     # KV 較小
   xb × Wv (2048,512)  → v (512,)

4. 拆 head（只是用指標偏移看不同區段）：
   q (2048,) → 32 個 head，每個 (64,)
   k (512,)  →  8 個 head，每個 (64,)
   v (512,)  →  8 個 head，每個 (64,)
```

PyTorch 訓練時會有 batch 維度 (seq_len, dim) 並用 reshape 拆 head，
但 `runq.c` 一次 1 個 token，用指標偏移 `s->q + h * head_size` 就等於拆 head。

---

### Autoregressive 推論 vs 批次推論

> **`runq.c` 即使是 prompt 也一次只處理 1 個 token**（逐 token 推論）

例如 prompt =「今天好」：
```
pos=0: forward(今, 0) → 不取樣，直接用 prompt[1]
pos=1: forward(天, 1) → 不取樣，直接用 prompt[2]
pos=2: forward(好, 2) → prompt 結束，取樣下一個字
```

#### RoPE 為何仍然有效？

每次呼叫 `pos` 不同，K 存入 cache 前已被該 pos 的 RoPE 旋轉過。
做 Attention 時 Q(pos=2) 和 K(pos=0), K(pos=1), K(pos=2) 點積，位置差異自然編碼在其中。

#### 和批次推論的比較

| | 批次推論（訓練/PyTorch） | `runq.c`（逐 token） |
|---|---|---|
| 輸入 | (seq_len, dim) 一次全丟 | (dim,) 一次一個 |
| QKV | 大矩陣 → reshape 拆 head | 一維向量 → 指標偏移 |
| RoPE | 一次對所有位置做 | 每次只對當前 pos 做 |
| Attention | Q(all) × K(all)^T | Q(current) × K(cache)^T |

兩種做法**數學等價**，只是拆成逐步執行。

---

### KV Cache 完整圖解（以「今天好嗎」為例，附維度）

每次 forward **只算當前 token 的 Q, K, V**，K 和 V 存進 cache。

> **符號說明：**
> - `Q[h3]` = Q 的第 3 個 head（共 32 個 Q head，每個 64 維）
> - `K[p0,h0]` = 位置 0、KV head 0 的 K（共 8 個 KV head，每個 64 維）
> - `V[p0,h0]` = 同上，V 向量
>
> **「丟掉 logits，用 prompt[x]」的意思：**
> prompt 階段下一個字已經知道了（你給的輸入），不需要取樣。
> 但 forward 不是白跑的——它把 K,V 存入 cache，為後續 Attention 做準備。

---

**pos=0「今」：**
```
Embedding 查表 → x (2048,) → RMSNorm → xb (2048,)

QKV 投影（3 次 MatMul）：
  xb (2048,) × Wq → q (2048,)  → 拆成 Q[h0]~Q[h31]，各 (64,)
  xb (2048,) × Wk → k (512,)   → 拆成 K[p0,h0]~K[p0,h7]，各 (64,)
  xb (2048,) × Wv → v (512,)   → 拆成 V[p0,h0]~V[p0,h7]，各 (64,)

RoPE(pos=0) 旋轉 Q 和 K

存入 KV Cache（8 個 KV head × 64 維 = 512 維）：
  K Cache: [ K[p0,h0], K[p0,h1], ..., K[p0,h7] ]
  V Cache: [ V[p0,h0], V[p0,h1], ..., V[p0,h7] ]

Attention（32 個 Q head 各自獨立，但 GQA 共用 KV head）：

  Q[h0] · K[p0,h0] → score → softmax=[1.0] → 1.0 × V[p0,h0] → (64,) ┐
  Q[h1] · K[p0,h0] → score → softmax=[1.0] → 1.0 × V[p0,h0] → (64,) │ 共用
  Q[h2] · K[p0,h0] → ...                                      → (64,) │ KV
  Q[h3] · K[p0,h0] → ...                                      → (64,) ┘ head 0

  Q[h4] · K[p0,h1] → ...   → 1.0 × V[p0,h1] → (64,) ┐ 共用
  Q[h5] · K[p0,h1] → ...                      → (64,) │ KV
  Q[h6] · K[p0,h1] → ...                      → (64,) │ head 1
  Q[h7] · K[p0,h1] → ...                      → (64,) ┘
  ...
  Q[h28]~Q[h31] · K[p0,h7] → ... × V[p0,h7]  → 各 (64,)  ← 共用 KV head 7

32 個 head 拼接：32 × (64,) = (2048,)
× Wo (2048,2048) → (2048,) → 加殘差 → FFN → logits (128256,)
→ logits 丟掉（prompt 階段），直接用 prompt[1] =「天」
```

---

**pos=1「天」：**
```
同上流程 → Q[h0]~Q[h31]，K[p1,h0]~K[p1,h7]，V[p1,h0]~V[p1,h7]

KV Cache 更新（每個 KV head 現在有 2 個位置的資料）：
  K Cache head 0: [ K[p0,h0], K[p1,h0] ]  各 (64,)
  V Cache head 0: [ V[p0,h0], V[p1,h0] ]  各 (64,)
  ...
  K Cache head 7: [ K[p0,h7], K[p1,h7] ]
  V Cache head 7: [ V[p0,h7], V[p1,h7] ]

Attention（每個 Q head 做 2 次點積）：

  Q[h0] · K[p0,h0] = score₀ ┐ 分別做點積
  Q[h0] · K[p1,h0] = score₁ ┘
  softmax → [0.35, 0.65]
  output = 0.35 × V[p0,h0] + 0.65 × V[p1,h0] = (64,)

  Q[h1] · K[p0,h0], K[p1,h0] → 不同的 softmax → 不同的 (64,)
  Q[h2], Q[h3] 同上，都共用 KV head 0
  ...
  Q[h4]~Q[h7] 共用 KV head 1
  ...
  Q[h28]~Q[h31] 共用 KV head 7

32 × (64,) = (2048,) → logits (128256,) → 丟掉，用 prompt[2] =「好」
```

---

**pos=2「好」（prompt 最後一個）：**
```
KV Cache 每個 head 有 3 個位置：
  K Cache head 0: [ K[p0,h0], K[p1,h0], K[p2,h0] ]

Attention（每個 Q head 做 3 次點積）：
  Q[h0] · K[p0,h0] → s₀ ┐
  Q[h0] · K[p1,h0] → s₁ ├ softmax → [0.2, 0.3, 0.5]
  Q[h0] · K[p2,h0] → s₂ ┘
  output = 0.2×V[p0,h0] + 0.3×V[p1,h0] + 0.5×V[p2,h0] = (64,)
  ...（其他 head 同理）

32 × (64,) = (2048,) → logits (128256,) → ✅ 取樣！→ 預測「嗎」
```

---

**注意力權重 vs 權重矩陣的差別：**
- Wq, Wk, Wv：投影用的**可學習權重矩陣** (2048,2048) 或 (2048,512)
- attn_weights：softmax 輸出的**注意力權重** (pos+1,) 個數字，加起來=1

#### pos=0 時 Attention 的細節（1B: 32 heads, head_size=64）

```
32 個 Q head 各自做「點積」（不是矩陣乘法）：
  Q head 0 (64,) · K head 0 (64,) → 1 個數字 → softmax = [1.0]
  Q head 1 (64,) · K head 0 (64,) → 1 個數字 → softmax = [1.0]
  ...（32 個 head 全部都是 [1.0]）

※ softmax 只有 1 個元素時，不管數字多少，結果一定是 1.0

每個 head 的輸出：
  1.0 × v₀ (64,) = (64,)    ← 純量 × 向量 = 向量（不是「還原」）
  ↑注意力權重  ↑V向量

32 個 head 拼接：32 × 64 = (2048,)
```

#### pos=1 起才有「比較」

```
Q head 0 (64,) · K at pos=0 (64,) → score₀  ← 兩次分別做點積
Q head 0 (64,) · K at pos=1 (64,) → score₁
softmax([score₀, score₁]) → [0.35, 0.65] ← 兩個不同的權重！
output = 0.35 × v₀(64,) + 0.65 × v₁(64,) = (64,)
```

從 pos=1 開始，模型才真正在「比較」不同位置的重要性。

---

### ③ Embedding 查表（L405-406）

```c
memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));
```

`memcpy(dst, src, size)` = C 的記憶體複製函式：

| 參數 | 對應 | 意思 |
|---|---|---|
| `dst` | `x` | 複製**到**哪裡（activation 向量） |
| `src` | `w->token_embedding_table + token*dim` | 從**哪裡**複製（嵌入表第 token 行） |
| `size` | `dim * sizeof(float)` | 複製多少位元組 |

`sizeof(float)` = **4 bytes**（C 的 float = float32）。
所以 size = 2048 × 4 = 8192 bytes。

> `runq.c` 裡 activation（x, q, logits 等）都是 **float32**。
> 只有**權重**被量化成 int8（1 byte），activation 維持 float32 精度。

**嵌入表的記憶體排列：**
```
記憶體地址:  [0]        [2048]     [4096]     ...
            ├──────────┼──────────┼──────────┤
token ID 0: | float×2048 |          |          |
token ID 1: |          | float×2048 |          |
token ID 2: |          |          | float×2048 |
            ...

例：token=5678 → 起始位置 = 5678 × 2048 = 第 11,628,544 個 float
```

`w->token_embedding_table + token*dim` 就是指標往後跳 token×dim 格，
跳到第 token 個 row 的起始位置，`memcpy` 把那整個 row（2048 個 float）複製到 `x`。

**白話：用 token ID 查表，取出 2048 維的嵌入向量，放進 x。**

---

### ④ 進入 16 層迴圈（L409）

```c
for (int l = 0; l < p->n_layers; l++) {
```

l = 0,1,2,...,15，共 16 層。每一層做一樣的事（RMSNorm → QKV → RoPE → Attention → FFN），只是用不同層的權重。

---

### ⑤ Attention 前的 RMSNorm（L412）

```c
rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
```

| 參數 | 對應 | 意思 |
|---|---|---|
| `s->xb` | 輸出 | 正規化後的結果（2048 維） |
| `x` | 輸入 | 當前的 activation 向量（2048 維） |
| `w->rms_att_weight + l*dim` | 權重 | 第 `l` 層的 RMSNorm 權重（用 `l*dim` 跳到該層） |
| `dim` | 維度 | 2048 |

`rms_att_weight` 和嵌入表一樣是連續陣列，用 `l*dim` 跳到第 l 層的起始位置。

**白話：把 x 做 RMSNorm 正規化，結果存到 xb。每一層用自己的權重。**

#### 為什麼結果存到 `xb` 而不是覆蓋 `x`？

因為後面有**殘差連接**，`x` 本身不會被改動，它就是「備份」：
```
x (原始值，不動)
├── 直通 ──────────────────────────────┐
│                                      ↓
└→ RMSNorm(x) → xb → Attention → xb' (+) → 新的 x = x + xb'
```
最後才做 `x[i] += xb[i]`，把 Attention 結果加回原始 x。

#### rmsnorm 函式本體（L328-341）

```c
void rmsnorm(float* o, float* x, float* weight, int size) {
    // Step 1: 算平方和
    float ss = 0.0f;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];

    // Step 2: 算 1/RMS
    ss /= size;             // 除以 2048 → 平均
    ss += 1e-5f;            // 加 ε 避免除以 0
    ss = 1.0f / sqrtf(ss);  // 取倒數平方根

    // Step 3: 正規化 × 權重
    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}
```

公式：`o[j] = weight[j] × x[j] / RMS(x)`
作用：讓向量的尺度穩定，再乘可學習權重做縮放。

#### 維度與運算方式

| 變數 | 維度 | 說明 |
|---|---|---|
| `x` | (2048,) | 輸入向量 |
| `weight` | (2048,) | **一維向量**（不是矩陣！） |
| `o` | (2048,) | 輸出向量 |
| `ss` | scalar | 純量，= 1 / RMS(x) |

`o[j] = weight[j] * (ss * x[j])` 是**逐元素相乘**（element-wise），每個 index j 獨立計算。
不是點積（點積會產生 scalar），也不是矩陣乘法（那會改變維度）。
輸入輸出維度不變，都是 (2048,)。

#### RMSNorm 後的向量長度（L2 norm）

RMS(x) = √(Σxⱼ²/n)，L2 norm = √(Σxⱼ²) = RMS × √n

除以 RMS 後：
- 向量的 **RMS 值 = 1**（每個元素的平均能量被壓到 1）
- 向量的 **L2 norm = √n = √2048 ≈ 45.25**（不是 1！）

→ RMSNorm ≠ unit vector normalization（那個是除以 L2 norm，會讓 L2 norm = 1）

#### 乘上 weight 後的平方和固定嗎？

- 乘 weight **之前**：Σ(ss × x[j])² = n = 2048 → **固定**（因為 RMS 被壓成 1）
- 乘 weight **之後**：Σ(weight[j] × ss × x[j])² → **不固定**

因為結果同時取決於 weight 的分佈和每次輸入的 normalized_x 分佈。
weight 的作用就是**刻意打破均勻性**，讓某些維度放大、某些縮小——這正是模型學到的東西。

#### 為什麼 RMSNorm 有 weight？不是只做正規化嗎？

正規化會把所有維度壓到差不多的尺度，但模型可能希望某些維度比較重要（值大）、某些不重要（值小）。
`weight` **(2048,)** 就是可學習的縮放因子（gamma），讓模型自己決定每個維度該有多大的尺度。

#### RMSNorm 是動態的，無法用 LUT 預算

`ss` 取決於輸入向量 x 的所有 2048 個值，每個 token、每一層都不同，所以 `1/√ss` 每次都不一樣。
LUT 不能預算 RMSNorm 結果，但可以加速裡面的 `1/√x` 函數：

| RTL 方法 | 說明 |
|---|---|
| CORDIC | 可算 1/√x，適合 FPGA |
| Newton-Raphson | 迭代近似 1/√x |
| LUT + 插值 | 對 1/√x 函數做表，不是對 RMSNorm 結果做表 |

#### xb 和 xb' 的澄清

筆記裡 `xb'` 只是標記「xb 被 Attention 更新後的值」。在程式碼裡**始終都是同一個 `s->xb`**，
只是內容會被覆寫：先存 RMSNorm 結果 → 後來被 Attention 輸出覆蓋。

---

### ⑥ QKV 投影（L415-418）

```c
quantize(&s->xq, s->xb, dim);
matmul(s->q, &s->xq, w->wq + l, dim, dim);
matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);
```

先做一件事再做三件事：

#### Step 1：`quantize` — 把 xb 從 float32 轉成 int8

```c
quantize(&s->xq, s->xb, dim);
```

| 參數 | 意思 |
|---|---|
| `&s->xq` | 輸出：量化後的結果（QuantizedTensor，含 int8 值 + 縮放因子） |
| `s->xb` | 輸入：RMSNorm 後的 float32 向量 (2048,) |
| `dim` | 2048 |

**為什麼要量化？** 因為接下來的 matmul 要用 **int8 × int8** 做矩陣乘法，比 float32 × float32 快很多。
所以先把 activation（xb）量化成 int8，搭配已經是 int8 的權重（Wq, Wk, Wv）一起算。

#### 用 int8 算不會出事嗎？

**不會大錯特錯**，因為有縮放因子（scale）保護：
```
假設一組 64 個 float 的最大絕對值 = 2.54
scale = 2.54 / 127 = 0.02

量化：float [1.27, -2.54, 0.05] → ÷ scale → round → int8 [64, -127, 2]
還原：int8 × scale → [1.28, -2.54, 0.04]  ← 只有微小誤差！
```
每 64 個值一組，各有自己的 scale，精度損失很小。

#### quantize 函式本體（L149-175）

```c
void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;    // 2048/64 = 32 組
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}
```

輸出 `QuantizedTensor` 包含：
- `qx->q`：2048 個 int8（量化值）
- `qx->s`：32 個 float（每組一個縮放因子，GS=64）

#### `s` 和 `q` 不用額外宣告陣列嗎？

不用，它們在 QuantizedTensor 結構體（L37-40）裡已經宣告：
```c
typedef struct {
    int8_t* q;    // 量化值陣列（指標 = 陣列地址）
    float*  s;    // 縮放因子陣列
} QuantizedTensor;
```
記憶體在初始化時已分配好，直接用 `[index]` 存取即可。

#### `qx->s[group] = scale` 是複製還是 reference？

**是複製。** C 對基本型別（float, int）的賦值一律是複製值，不是 reference。

#### GS = 64 從哪來？

從 `.bin` checkpoint 檔案讀入（L267-271），不是寫死在程式碼裡：
```c
int group_size;
fread(&group_size, sizeof(int), 1, file);
GS = group_size;
```

#### Int8 (Q8_0) vs NF4 的差別

| | `runq.c` 的 Int8 | bnb-4bit（NF4） |
|---|---|---|
| 精度 | 8-bit | 4-bit |
| 方法 | 線性對稱：`value / scale → int8` | 非線性：常態分佈分位數映射 |
| 用在 | 這份 C 程式碼 (.bin) | HuggingFace / PyTorch (.safetensors) |

兩者是不同的量化方案，不能混用。

#### 一層 Transformer 內的完整資料流（含 quantize 位置）

```
步驟                    精度          quantize？
───────────────────────────────────────────────
1. RMSNorm(x → xb)     float32       ❌
   ── quantize(xb → xq) ──           ✅ xb → int8
2. matmul(xq × Wq → q) int8×int8
3. matmul(xq × Wk → k) int8×int8
4. matmul(xq × Wv → v) int8×int8
   （q,k,v 出來是 float32）
5. RoPE(q, k)           float32       ❌
6. 存 K,V 到 cache      float32       ❌
7. Q·K 點積             float32       ❌
8. Softmax              float32       ❌
9. scores × V           float32       ❌
   ── quantize(attn → xq) ──         ✅ attn 輸出 → int8
10. matmul(xq × Wo → xb) int8×int8
11. 殘差 x += xb         float32       ❌
12. RMSNorm(x → xb)     float32       ❌
   ── quantize(xb → xq) ──           ✅ xb → int8
13. matmul(xq × W1 → h1) int8×int8   ← FFN
14. matmul(xq × W3 → h2) int8×int8
15. SwiGLU(h1, h2)       float32       ❌
   ── quantize(hh → xq) ──           ✅ FFN 中間 → int8
16. matmul(xq × W2 → xb) int8×int8
17. 殘差 x += xb         float32       ❌
```

每層：**4 次 quantize，7 次 int8 matmul**。

#### 運算比重（1B，dim=2048，1 token）

| 運算 | 乘加次數 | 佔比 |
|---|---|---|
| QKV matmul (Wq+Wk+Wv) | 6.3M | ~18% |
| Output matmul (Wo) | 4.2M | ~12% |
| FFN matmul (W1+W3+W2) | 50.3M | **~70%** |
| Attention 點積 + 其他 | 很小 | <1% |

**FFN 的三個 matmul 佔約 70%！** 量化 matmul 最有價值。

#### 為什麼這些運算不能用 int8？

**根本原因：int8 是整數，這些函數在整數世界裡不存在。**

| 運算 | 需要的函數 | int8 能做嗎？ |
|---|---|---|
| Softmax | `exp()` + 除法得到 0~1 小數 | ❌ 沒有 exp，沒有小數 |
| RoPE | `cos()`, `sin()` | ❌ 沒有三角函數 |
| SwiGLU | `sigmoid()` = 1/(1+exp(-x)) | ❌ 還是需要 exp |
| RMSNorm | `sqrt()` + 除法 | ❌ 沒有開根號 |
| MatMul | 乘法 + 加法 | ✅ int8 能做！ |

不是「精度不夠」的問題，而是 int8 只有加減乘，**根本沒有 exp、cos、sqrt 這些運算**。

MatMul 可以用 int8 是因為它只需要 `int8 × int8 → 累加成 int32 → 最後乘 scale 還原 float32`。

#### Softmax 公式複習

```
softmax(xᵢ) = exp(xᵢ) / Σexp(x)

例：scores = [3.0, 3.05]
  exp(3.0)  = 20.0855
  exp(3.05) = 21.1150
  總和 = 41.2005
  softmax = [0.4876, 0.5124]  ← 加起來 = 1
```

#### Step 2：三次 `matmul` — 投影出 Q, K, V

```c
matmul(s->q, &s->xq, w->wq + l, dim, dim);      // xq × Wq → q
matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);   // xq × Wk → k
matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);   // xq × Wv → v
```

| 呼叫 | 輸出 | 權重 | 輸入維度 | 輸出維度 |
|---|---|---|---|---|
| 第 1 個 | `s->q` (2048,) | `w->wq` 第 l 層 | 2048 | 2048 |
| 第 2 個 | `s->k` (512,) | `w->wk` 第 l 層 | 2048 | 512 |
| 第 3 個 | `s->v` (512,) | `w->wv` 第 l 層 | 2048 | 512 |

**白話：把 xb 分別乘以三個不同的權重矩陣，得到 Q (2048維)、K (512維)、V (512維)。**

> `w->wq + l` 的 `+ l` 不是跳 l 個 float，而是跳 l 個 QuantizedTensor 結構體，
> 因為 `w->wq` 的型別是 `QuantizedTensor*`，指標運算會自動按結構體大小跳。

#### `w->wq + l` 到底在做什麼？（完整拆解）

**`w->wq` 是一個有 16 個格子的陣列**（16 層），在 `memory_map_weights`（L213）建立：
```c
w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
//                                    16 層         size_each = 2048×2048 = 4,194,304
```

每個格子是一個 QuantizedTensor，只存**兩個門牌**（指標），本身很小（16 bytes）：
```
格子 0 = { .q = 門牌A,  .s = 門牌B }   ← 指向 layer 0 的資料
格子 1 = { .q = 門牌C,  .s = 門牌D }   ← 指向 layer 1 的資料
```

**`+ l` 選名片，`->q` 和 `->s` 看名片上寫的地址：**
```
w->wq + l   →  拿到第 l 張名片（格子）
matmul 收到名片  →  用 w->q 讀門牌 → 從那個地址開始讀 int8 權重
                   用 w->s 讀門牌 → 從那個地址開始讀 scale
```

**具體數字範例（假設 Wq 資料從地址 50000 開始）：**

`init_quantized_tensors` 裡的 for 迴圈會這樣設定門牌：

| 格子 | `.q`（int8 起點） | `.s`（scale 起點） | 怎麼算的 |
|---|---|---|---|
| 0 | **50,000** | 50,000 + 4,194,304 = **54,194,304** | .q 後面跳 size_each 個 int8 |
| 1 | 54,194,304 + 262,144 = **54,456,448** | **58,650,752** | .s 後面跳 size_each/GS 個 float |

格子 0 和格子 1 的 `.q` 差了 ~4.5M（一整層的 int8 + scale）。
但 `w->wq + 0` 到 `w->wq + 1` 只差 16 bytes（名片本身的大小）。

**所以 matmul 拿到不同的 l 時：**
```c
matmul(..., w->wq + 0, ...);  // w->q = 50,000       → layer 0 的 int8
matmul(..., w->wq + 1, ...);  // w->q = 54,456,448   → layer 1 的 int8
```

#### matmul 函式本體（L363-388）

```c
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;       // 最終 float32 結果
        int32_t ival = 0;       // int8×int8 累加器（用 int32 避免溢位）
        int in = i * n;         // 權重矩陣第 i 個 row 的起始位置

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
                //      ↑ input int8              ↑ weight int8
                //      兩個 int8 相乘，累加到 int32
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            //     ↑ int32→float   ↑ weight scale         ↑ input scale
            //     乘回兩個 scale → 還原成 float32
            ival = 0;  // 重置，下一組重新累加
        }

        xout[i] = val;   // 存入輸出
    }
}
```

重點：
- `#pragma omp parallel for`：用多線程平行處理（每個 i 可以獨立算）
- 每 GS=64 個 int8 累加成 int32 → 乘回 scale → 加到 float32 結果
- `int32_t` 是因為 64 個 int8×int8 的最大累加 = 64×127×127 = 1,032,256，超過 int16 上限

#### `w->wq + l` 怎麼跳到下一層？

C 的指標加法：`QuantizedTensor* + l` = 跳過 l 個**完整的 QuantizedTensor**。
```
w->wq 指向：
  [QuantizedTensor 0] [QuantizedTensor 1] ... [QuantizedTensor 15]
   ↑ l=0 的 Wq         ↑ l=1 的 Wq              ↑ l=15 的 Wq
   含 2048×2048 int8    含 2048×2048 int8
```
`+ l` 不是加 l 個 byte，是跳 l 整個結構體（含 4,194,304 個 int8 + scale）。

#### matmul 是矩陣乘法還是內積？

是**矩陣-向量乘法（Matrix-Vector Multiplication）**：
`W (d,n) × x (n,) → xout (d,)` → 每個 `xout[i]` = W 第 i 個 row 跟 x 的內積。
`in = i * n` 就是在矩陣裡切換到下一個 row 的功能。

#### scale factor 的 index 怎麼看？

```c
val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
```
- `w->s[(in+j)/GS]`：W 是二維展平成一維，`in` 跳到第 i 個 row，`j` 是 row 內偏移，`/GS` 得到組號
- `x->s[j/GS]`：x 是一維向量，`j/GS` 就是組號。每次新的 i（新 row）都重複使用同一組 x scale

**具體範例（以 Wq 為例：n=2048, GS=64）：**

| i (row) | in = i×2048 | j 範圍 | `(in+j)/GS` | W 的第幾組 scale | `j/GS` | x 的第幾組 |
|---|---|---|---|---|---|---|
| 0 | 0 | 0~63 | 0 | row 0 的第 0 組 | 0 | 第 0 組 |
| 0 | 0 | 64~127 | 1 | row 0 的第 1 組 | 1 | 第 1 組 |
| 0 | 0 | 1984~2047 | 31 | row 0 的第 31 組 | 31 | 第 31 組 |
| 1 | 2048 | 0~63 | **32** | row 1 的第 0 組 | 0 | 第 0 組 |
| 1 | 2048 | 64~127 | **33** | row 1 的第 1 組 | 1 | 第 1 組 |

W 的 scale 是一個攤平的大陣列：`[row0 的 32 個, row1 的 32 個, ..., row2047 的 32 個]`。
x 的 scale **一直在 0~31 來回**，因為不管算哪個 `q[i]`，輸入向量 x 都是同一個。

**scale factor 的數量：**
- x 的 scale：2048 / 64 = **32 個**（整個 x 共用）
- W 的 scale：每行 32 個 × 2048 行 = **65,536 個**（Wq 為例）

#### x 和 W 的 scale 從哪來？

| | x 的 scale | W 的 scale |
|---|---|---|
| 何時算 | 推論時，每次 `quantize()` 動態算 | 離線（export 模型時）預先算好 |
| 存在哪 | RunState 的 `s->xq.s` | checkpoint `.bin` 裡 memory map |
| 會變嗎 | 每個 token、每一層都不同 | 固定不變 |

W 的 scale 在 `init_quantized_tensors`（L179）載入時，直接從檔案 memory map 指過去：
```c
res[i].q = (int8_t*)p;           // int8 量化值
p = (int8_t*)p + size_each;
res[i].s = (float*)p;            // scale factors（緊接在 int8 後面）
p = (float*)p + size_each / GS;
```
檔案裡的排列：`[int8 值 × size_each][scale × size_each/GS]`，指標直接指過去即可。

#### matmul 可視化

![matmul int8 分組運算示意圖](matmul_visualization.png)

#### matmul 展開範例（以 Wk 為例：d=512, n=2048, GS=64）

```
matmul(s->k, &s->xq, w->wk+l, n=2048, d=512)
→ Wk (512, 2048) × xq (2048,) → k (512,)

外層 i = 0,1,...,511  ← 512 次內積

i=0（算 k[0]）：in = 0×2048 = 0  ← W 第 0 個 row
  j=0,   group 0:  x.q[0..63]  × w.q[0..63]    → ival → ×scale → val
  j=64,  group 1:  x.q[64..127] × w.q[64..127]  → val +=
  ...
  j=1984,group 31: x.q[1984..2047] × w.q[1984..2047] → val +=
  k[0] = val（32 組累加完畢）

i=1（算 k[1]）：in = 1×2048 = 2048  ← W 第 1 個 row 起點
  j=0,   group 0:  x.q[0..63] × w.q[2048..2111]  ← w 的 index 變了！
  ...
  k[1] = val

  ...一直到 i=511 → k[511] = val

完整的 k (512,) 算完。
```

---

### ⑦ RoPE 旋轉位置編碼（L422-437）

核心想法：把 Q 和 K 的每兩個相鄰元素當作一個 2D 點，旋轉一個角度。

#### 2D 旋轉公式

```c
vec[i]   = v0 * fcr - v1 * fci;   // x' = x×cosθ - y×sinθ
vec[i+1] = v0 * fci + v1 * fcr;   // y' = x×sinθ + y×cosθ
```

就是把 (v0, v1) 這個 2D 點旋轉 θ 度。

#### 角度 θ 怎麼算？

```c
int head_dim = i % head_size;   // 在 head 內的位置 (0,2,4,...,62)
float freq = 1.0f / powf(500000.0f, head_dim / (float)head_size);  // 頻率
float val = pos * freq;         // θ = 位置 × 頻率
float fcr = cosf(val);          // cos θ
float fci = sinf(val);          // sin θ
```

**對應數學公式：**

θ(pos, d) = pos / 500000^(d/64)

其中 d = head_dim（0,2,4,...,62），64 = head_size。500000 是 Llama 3.2 的 base（有些模型用 10000）。

#### RoPE 的 cos/sin 可以事先計算

d 是固定的（0,2,4,...,62，共 32 個值），pos 的範圍也已知（0 ~ seq_len-1），
所以所有 cos(θ) 和 sin(θ) 都可以預先算好，存成 lookup table：

```
RoPE_table[pos][d/2] = (cosθ, sinθ)
```

大小 = seq_len × 32 × 2 個 float，非常小。
PyTorch 版通常在初始化時就算好 `cos_cached` / `sin_cached`，推論時直接查表。
FPGA 實作可以直接存在 ROM 裡，不需要每次算 `powf`、`cosf`、`sinf`。

| head_dim | head_dim/64 | freq | 旋轉速度 |
|---|---|---|---|
| 0 | 0.000 | 1.0 | 最快 |
| 2 | 0.031 | 0.74 | ↓ |
| 62 | 0.969 | 0.000004 | 最慢 |

前面的元素對 → 高頻（轉得快），後面 → 低頻（轉得慢）。
`val = pos × freq`：pos 越大 → θ 越大 → 旋轉越多。

#### RoPE 的直覺：為什麼 pos 越後面角度越大？

重點不在絕對角度，而在**兩個 token 的角度差**：
- 角度差越大 → cos(差) 越小 → Q·K 點積越小 → 注意力越低
- 就像時鐘：重點是兩個時間的角度差能告訴你它們**隔了多遠**

```
假設 freq=1，原始 Q 和 K 都是 (1.0, 0.0)

pos=0: Q 旋轉 0°    → Q = (1.00, 0.00)
pos=1: K 旋轉 57°   → K = (0.54, 0.84)
pos=5: K 旋轉 286°  → K = (0.28, -0.96)

Q · K(pos=1) = 1.0×0.54 + 0×0.84 = 0.54   ← 近的字，分數高
Q · K(pos=5) = 1.0×0.28 + 0×(-0.96) = 0.28 ← 遠的字，分數低
```

距離越遠 → 點積越小 → 注意力越低。
模型天然傾向關注近的字，但可透過學習 Q,K 權重克服此偏好。

#### 釐清：RoPE 只做旋轉，點積是 Attention 的事

```
RoPE 做的事（每次 forward）：
  matmul → Q, K → RoPE 旋轉 Q 和 K → 存 K 到 cache → 結束

Attention 做的事（RoPE 之後）：
  拿旋轉過的 Q，跟 cache 裡旋轉過的所有 K 做點積 → softmax → 加權 V
```

以「今天好」為例：
```
pos=0(今): Q₀,K₀ 旋轉 θ=0×freq → K₀ 存 cache
pos=1(天): Q₁,K₁ 旋轉 θ=1×freq → K₁ 存 cache
           Attention: Q₁·K₀ (角度差=freq) → 小分數
                      Q₁·K₁ (角度差=0)    → 大分數
```

#### Q 和 K 的 RoPE 角度不會衝突嗎？（GQA）

不會！`head_dim = i % head_size (64)` 讓每個 head 的角度模式都重複：
```
Q head 0: i=0,2,...,62    → head_dim = 0,2,...,62
Q head 1: i=64,66,...,126 → head_dim = 0,2,...,62  ← 一模一樣
K head 0: i=0,2,...,62    → head_dim = 0,2,...,62  ← 也一樣
```
Q head 0~3 做點積時跟 K head 0 的 RoPE 角度完全相同，不會有衝突。

#### `rotn` 迴圈展開

```c
int rotn = i < kv_dim ? 2 : 1;
for (int v = 0; v < rotn; v++) {
    float* vec = v == 0 ? s->q : s->k;
    float v0 = vec[i];
    float v1 = vec[i+1];
    vec[i]   = v0 * fcr - v1 * fci;
    vec[i+1] = v0 * fci + v1 * fcr;
}
```

```
i=0（< 512）：rotn=2 → 旋轉 Q[0,1] 和 K[0,1]
  v=0: q[0],q[1] 旋轉
  v=1: k[0],k[1] 旋轉

i=512（>= 512）：rotn=1 → 只旋轉 Q[512,513]
  v=0: q[512],q[513] 旋轉
  （K 只有 512 維，沒有 index 512，所以不轉）
```

#### 指標 vs 複製的差別

- `float* vec = s->q`：**指標**，vec 指向 q 的地址，修改 `vec[i]` 就是直接改 `s->q[i]`
- `float v0 = vec[i]`：**複製**，備份舊值。因為旋轉公式兩行都需要原始 v0,v1，若不備份第二行會用到已被覆寫的值

---

### ⑧ KV Cache 存入（L439-444）

```c
int loff = l * p->seq_len * kv_dim;
float *key_cache_row = s->key_cache + loff + pos * kv_dim;
float *value_cache_row = s->value_cache + loff + pos * kv_dim;
memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
```

**白話：把剛算好的 K 和 V（各 512 維）存進 cache 的正確位置。**

#### KV Cache 是預先開好的

記憶體在 `malloc_run_state`（L94）一次全部分配：
```c
s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
```
16 層 × seq_len 個位置 × 512 維，一開始全是 0，每次 forward 只填入一格。

`seq_len` 從 checkpoint `.bin` 檔案的 Config header 讀入（`fread(config, sizeof(Config), 1, file)`），
是模型訓練時決定的**上限**，input + output 的 token 總數不能超過它。

#### `loff` 是偏移量，不是最大值

`loff = l × seq_len × kv_dim` 會隨著 `l`（層號）變化：

| l | loff（假設 seq_len=2048） | 指向 |
|---|---|---|
| 0 | 0 | layer 0 的開頭 |
| 1 | 2048 × 512 = 1,048,576 | layer 1 的開頭 |
| 2 | 2,097,152 | layer 2 的開頭 |

`s->key_cache + loff + pos * kv_dim` = 基底地址 + 跳到第 l 層 + 跳到第 pos 個位置。

#### 記憶體佈局

KV Cache 是一個三維陣列攤平成一維：`(layer, seq_len, kv_dim)`

```
s->key_cache 的結構：

  layer 0                          layer 1                    ...
  ├─ pos=0 (512 floats) ─┤         ├─ pos=0 ─┤
  ├─ pos=1 (512 floats) ─┤         ├─ pos=1 ─┤
  ├─ ...                 ─┤         ├─ ...    ─┤
  └─ pos=seq_len-1       ─┘         └─ ...    ─┘
  ↑                                 ↑
  loff=0                            loff = 1 × seq_len × 512
```

| 變數 | 算式 | 意思 |
|---|---|---|
| `loff` | `l × seq_len × 512` | 跳到第 l 層的起始位置 |
| `pos * kv_dim` | `pos × 512` | 在該層內跳到第 pos 個位置 |
| `key_cache_row` | `s->key_cache + loff + pos × 512` | 第 l 層、第 pos 個位置的 K 起始**地址** |

#### memcpy 怎麼讀

```c
memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
//     ↑ 目的地(地址)  ↑ 來源(地址)  ↑ 複製多少 bytes
```

| 參數 | 值 | 白話 |
|---|---|---|
| `key_cache_row` | cache 中該格的地址 | 複製**到哪裡** |
| `s->k` | 剛算好的 K 向量的地址 | 從**哪裡**複製 |
| `kv_dim * sizeof(*key_cache_row)` | 512 × 4 = 2048 bytes | 複製**多少** |

`memcpy` 只認 bytes，不認 float。所以要乘 `sizeof`：
- `sizeof(*key_cache_row)` = `sizeof(float)` = **4 bytes**
- 512 個 float × 4 bytes = 2048 bytes
- 如果只寫 `512`，它只會複製 512 bytes = 128 個 float，少了！

地址是你在前一行算好的，`memcpy` 只是照搬，不會自己決定放哪裡。

#### C 指標語法釐清：`*` 的兩種用法

`*` 出現在**宣告**和**使用**時意思完全不同：

| 出現位置 | 語法 | 意思 |
|---|---|---|
| 宣告時 | `float *k;` | k 是指標，存的是**地址**（可指向一整排 float） |
| 使用時 | `*key_cache_row` | 取出指標指向的**值**（解引用） |

對比 `float *k` vs `float k`：
- `float *k;` → 存地址，可以指向 512 個 float 的陣列
- `float k;` → 只能存 1 個 float 值

#### `->` vs `*` 的差別

| 語法 | 用在哪 | 意思 |
|---|---|---|
| `s->k` | 結構體指標 | 走進 s 指向的結構體，取出「k」欄位（= 一個地址） |
| `*p` | 普通指標 | 取出 p 指向的值 |

`s->k` 因為 RunState 裡 k 宣告為 `float *k`，取出來的就已經是地址，不用再加 `&`。

#### `key_cache_row` 加不加 `*` 的差別

```c
float *key_cache_row = s->key_cache + loff + pos * kv_dim;  // 宣告：算出地址

memcpy(key_cache_row, ...)    // 不加 * → 給 memcpy 地址（門牌號碼）✅
       *key_cache_row         // 加 * → 取出地址裡的值（一個 float）❌ memcpy 不需要值
```

`memcpy` 的前兩個參數都要**地址**，所以不加 `*`。
`sizeof(*key_cache_row)` 裡加 `*` 是問「指向的東西多大」→ `sizeof(float)` = 4。

---

### ⑨ Multihead Attention（L446-485）

```c
int h;
#pragma omp parallel for private(h)
for (h = 0; h < p->n_heads; h++) {
```

32 個 Q head 各自獨立做 Attention，可以平行處理（`#pragma omp`）。

#### Step 1：取出 Q head 和 Attention buffer（L450-453）

```c
float *q = s->q + h * head_size;        // Q 的第 h 個 head（64 維）
float *att = s->att + h * p->seq_len;   // 該 head 的 attention scores buffer
```

| 指標 | 偏移 | 取得的區段 |
|---|---|---|
| `s->q + h*64` | 第 h 個 head | q 陣列中 index [h×64 ~ h×64+63] |
| `s->att + h*seq_len` | 第 h 個 head 的 score 空間 | 最多 seq_len 個 float |

**為什麼 att 需要 `n_heads × seq_len` 的空間？**

每個 Q head 在第 pos 個 token 時，要跟 cache 裡 pos+1 個 K 分別做點積，產生 pos+1 個 score：
```
pos=0: Q·K[p0]                     → 1 個 score
pos=1: Q·K[p0], Q·K[p1]            → 2 個 score
pos=2: Q·K[p0], Q·K[p1], Q·K[p2]   → 3 個 score
...
pos=seq_len-1:                      → seq_len 個 score（最多）
```

所以每個 head 最多需要 `seq_len` 個格子，32 個 head = `32 × seq_len`。
分配在 `malloc_run_state`：`s->att = calloc(p->n_heads * p->seq_len, sizeof(float));`

#### Step 2：Q·K 點積（L455-466）

```c
for (int t = 0; t <= pos; t++) {
    float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
        score += q[i] * k[i];
    }
    score /= sqrtf(head_size);    // scaled dot-product
    att[t] = score;
}
```

遍歷所有已存在 cache 的位置（t = 0 到 pos），每次做一次 Q·K 點積。

**K 的指標偏移（GQA 的關鍵）：**

```
s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size
               ↑        ↑              ↑
             第 l 層   第 t 個位置    第幾個 KV head
```

`h / kv_mul`（整數除法）實現 GQA 共用：
- h=0,1,2,3 → 0/4=0 → 用 KV head 0
- h=4,5,6,7 → 4/4=1 → 用 KV head 1
- ...
- h=28~31 → 7 → 用 KV head 7

**帶數字的範例（l=0, pos=2, 基底地址=0）：**

| h | t | `t*512` | `(h/4)*64` | 最終 index | 取到什麼 |
|---|---|---|---|---|---|
| 0 | 1 | 512 | 0 | 512 | pos=1 的 KV head 0 |
| 3 | 1 | 512 | 0 | 512 | 也是 KV head 0（共用） |
| 4 | 1 | 512 | 64 | 576 | pos=1 的 KV head 1 |
| 4 | 0 | 0 | 64 | 64 | pos=0 的 KV head 1 |

`t * 512` 決定去**哪個位置**，`(h/4) * 64` 決定用**哪個 KV head**。

**`q[i]` 和 `k[i]` 的起點由外層決定：**

`q` 的起點由 `float *q = s->q + h * head_size` 決定（h 不同→指向不同的 64 維區段）。
`k` 的起點由 `t`（位置）和 `h/kv_mul`（KV head）決定。
`q[0]~q[63]` 和 `k[0]~k[63]` 是從各自起點連續取 64 個 float，逐位置相乘再加起來 = 64 維內積 → 1 個 score。

**`score /= sqrtf(head_size)`**：除以 √64 = 8，這是 Scaled Dot-Product Attention 的標準做法，
防止點積值隨 head_size 增大而變得過大，導致 softmax 輸出過於尖銳。

#### Step 3：Softmax（L468-469）

```c
softmax(att, pos + 1);
```

把 `att[0..pos]` 共 pos+1 個 score 轉成注意力權重（加起來 = 1）。

#### Step 4：加權求和 V（L471-484）

```c
float *xb = s->xb + h * head_size;
memset(xb, 0, head_size * sizeof(float));   // 先歸零
for (int t = 0; t <= pos; t++) {
    float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
    float a = att[t];                       // 第 t 個位置的注意力權重
    for (int i = 0; i < head_size; i++) {
        xb[i] += a * v[i];                  // 加權累加
    }
}
```

**白話：** 用 softmax 權重對所有位置的 V 做加權平均，結果存到 xb 的第 h 個 head 區段。

- `s->xb + h * head_size` → xb 中第 h 個 head 的 64 維區段
- V 的指標偏移跟 K 完全一樣（也用 `h / kv_mul` 做 GQA）
- `memset(xb, 0, ...)` → 先清零，因為要用 `+=` 累加

32 個 head 各自算好 (64,)，合起來就是 xb (2048,)。

#### Step 5：Attention 輸出投影 Wo（L487-489）

```c
quantize(&s->xq, s->xb, dim);
matmul(s->xb2, &s->xq, w->wo + l, dim, dim);
```

xb (2048,) → 量化 → × Wo (2048,2048) → xb2 (2048,)

**Wo 的作用：** 把 32 個 head 的結果「混合」起來。各 head 算出的 (64,) 只看到自己的子空間，
Wo 讓不同 head 的資訊可以交互影響。

---

### 運算類型彙整（RTL 設計參考）

| 運算 | 類型 | 精度 | 維度 | 說明 |
|---|---|---|---|---|
| QKV matmul (Wq,Wk,Wv) | **矩陣×向量** | int8 | W(2048,2048)×x(2048,)→(2048,) | 最大運算量 |
| Output matmul (Wo) | **矩陣×向量** | int8 | W(2048,2048)×x(2048,)→(2048,) | |
| FFN matmul (W1,W3) | **矩陣×向量** | int8 | W(8192,2048)×x(2048,)→(8192,) | 佔 ~70% |
| FFN matmul (W2) | **矩陣×向量** | int8 | W(2048,8192)×x(8192,)→(2048,) | |
| Attention Q·K | **點積** | float32 | (64,)·(64,)→純量 | 每 head 獨立 |
| Attention scores×V | **純量×向量加總** | float32 | Σ(純量×(64,))→(64,) | 加權求和 |
| RMSNorm | **逐元素** | float32 | (2048,) | 平方→平均→開根號→除→乘 |
| RoPE | **逐 pair (2D 旋轉)** | float32 | 每次 2 個元素 | cos/sin |
| SwiGLU | **逐元素** | float32 | (8192,) | sigmoid×乘法 |
| Softmax | **逐元素** | float32 | (pos+1,) | exp→加總→除 |
| 殘差 x += xb | **逐元素加法** | float32 | (2048,) | |

**RTL 設計啟示：**
- **矩陣×向量**（佔 90%+ 運算量）：適合用 systolic array 加速，int8 精度
- **點積**（Attention Q·K）：運算量小但 latency 敏感，float32
- **逐元素/逐 pair**：運算量小，可由 CPU 處理或簡單硬體實現

---

*（筆記持續更新中...）*
