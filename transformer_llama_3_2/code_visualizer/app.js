// ===== Data: Code Sections mapped to Architecture =====
const CODE_SECTIONS = [
  {
    id: 'config',
    archId: 'arch-config',
    tag: 'infra', tagLabel: '基礎',
    title: 'Config / QuantizedTensor 結構體',
    lines: 'L25-38',
    desc: `<b>C 語法重點：</b><span class="c-tip">typedef struct { ... } Name;</span> 是 C 裡面定義「自訂資料型別」的方式，類似 Python 的 class。<br><br>
<b>作用：</b>Config 儲存模型的「設計藍圖」——維度(dim=2048)、層數(n_layers=16)、注意力頭數(n_heads=32)等超參數。<br>
QuantizedTensor 是 Int8 量化格式：<code>q</code> 存量化後的 int8 值，<code>s</code> 存每個 group 的縮放因子。`,
    code: `typedef struct {
    int dim;        // transformer 維度 (2048)
    int hidden_dim; // FFN 隱藏層維度 (8192)
    int n_layers;   // 層數 (16)
    int n_heads;    // Query 頭數 (32)
    int n_kv_heads; // KV 頭數 (可 < n_heads，GQA)
    int vocab_size; // 詞彙表大小
    int seq_len;    // 最大序列長度
} Config;

typedef struct {
    int8_t* q;  // 量化值 (int8 陣列)
    float*  s;  // 縮放因子 (每 group 一個)
} QuantizedTensor;`
  },
  {
    id: 'weights',
    archId: 'arch-weights',
    tag: 'infra', tagLabel: '基礎',
    title: 'TransformerWeights 權重結構',
    lines: 'L40-61',
    desc: `<b>C 語法重點：</b><span class="c-tip">QuantizedTensor *wq</span> 中的 <code>*</code> 表示「指標」(pointer)，指向記憶體中某個位置。可以把指標想像成「地址」。<br><br>
<b>作用：</b>儲存模型所有可學習的權重矩陣。注意 wq/wk/wv/wo 是 Self-Attention 的權重，w1/w2/w3 是 FFN 的權重。`,
    code: `typedef struct {
    QuantizedTensor *q_tokens;       // 嵌入表 (vocab_size, dim)
    float* token_embedding_table;    // 嵌入表 (反量化版)
    float* rms_att_weight;           // Attention 前的 RMSNorm 權重
    float* rms_ffn_weight;           // FFN 前的 RMSNorm 權重
    QuantizedTensor *wq;  // Query 權重  (layer, dim, dim)
    QuantizedTensor *wk;  // Key 權重    (layer, dim, kv_dim)
    QuantizedTensor *wv;  // Value 權重  (layer, dim, kv_dim)
    QuantizedTensor *wo;  // Output 權重 (layer, dim, dim)
    QuantizedTensor *w1;  // FFN Gate    (layer, hidden_dim, dim)
    QuantizedTensor *w2;  // FFN Down    (layer, dim, hidden_dim)
    QuantizedTensor *w3;  // FFN Up      (layer, hidden_dim, dim)
    float* rms_final_weight;         // 最終 RMSNorm 權重
    QuantizedTensor *wcls;           // 分類器權重 (logits)
} TransformerWeights;`
  },
  {
    id: 'quantize',
    archId: 'arch-quant',
    tag: 'quant', tagLabel: '量化',
    title: 'quantize() / dequantize() 量化函數',
    lines: 'L140-172',
    desc: `<b>C 語法重點：</b><span class="c-tip">qx->q[i]</span> 等同於 <code>(*qx).q[i]</code>，是透過指標存取 struct 成員的語法。<code>-></code> 是 C 最常見的操作符之一。<br><br>
<b>作用：</b>Group Quantization——將 float32 壓縮成 int8 以節省記憶體。每 GS(128) 個元素為一組，共用一個縮放因子。<br>
<b>⚠️ FPGA 筆記：</b>你們討論過可以跳過這部分，改用 fixed-point 取代。`,
    code: `// 反量化：int8 還原為 float
void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
        // qx->q[i] = int8 量化值
        // qx->s[i/GS] = 該 group 的縮放因子
        // 乘起來就還原回近似的 float 值
    }
}

// 量化：float 壓縮為 int8
void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;  // GS=128, 每組 128 個元素
    float Q_MAX = 127.0f;     // int8 最大值 (2^7 - 1)

    for (int group = 0; group < num_groups; group++) {
        // 1. 找這組的最大絕對值
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) wmax = val;
        }
        // 2. 計算縮放因子 scale = wmax / 127
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;
        // 3. 量化：除以 scale 再四捨五入到 int8
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group*GS + i] / scale;
            qx->q[group*GS + i] = (int8_t)round(quant_value);
        }
    }
}`
  },
  {
    id: 'rmsnorm',
    archId: 'arch-rmsnorm',
    tag: 'core', tagLabel: '核心',
    title: 'rmsnorm() — RMS 正規化',
    lines: 'L283-296',
    desc: `<b>C 語法重點：</b>函式參數 <span class="c-tip">float* o</span> 是指標，代表「傳入的是陣列的地址」，函式可以直接修改原始陣列。這就是 C 的「傳址呼叫」。<br><br>
<b>作用：</b>RMS Normalization，公式：o[j] = weight[j] × (x[j] / √(mean(x²) + ε))。在每一層的 Attention 前和 FFN 前各呼叫一次。<br>
<b>🔧 FPGA 需實作：</b>涉及 √ 運算，可用 CORDIC 或查表。`,
    code: `void rmsnorm(float* o, float* x, float* weight, int size) {
    // 步驟 1: 計算平方和
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];  // 累加每個元素的平方
    }
    ss /= size;      // 除以元素數 → 平均值
    ss += 1e-5f;      // 加 ε 防止除以零
    ss = 1.0f / sqrtf(ss);  // 取倒數的平方根

    // 步驟 2: 正規化並乘以可學習權重
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}`
  },
  {
    id: 'softmax',
    archId: 'arch-softmax',
    tag: 'core', tagLabel: '核心',
    title: 'softmax() — Softmax 函數',
    lines: 'L298-316',
    desc: `<b>作用：</b>將任意數值轉換為機率分佈（所有值加起來 = 1）。用在 Attention Score 和最後的 Token 取樣。<br>
先減去最大值是為了數值穩定性，防止 exp() 爆掉。`,
    code: `void softmax(float* x, int size) {
    // 找最大值（數值穩定性）
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    // 計算 exp 並求和
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);  // 減去 max 防溢位
        sum += x[i];
    }
    // 正規化（除以總和 → 變成機率）
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}`
  },
  {
    id: 'matmul',
    archId: 'arch-matmul',
    tag: 'core', tagLabel: '核心',
    title: 'matmul() — 量化矩陣乘法（最耗時！）',
    lines: 'L318-343',
    desc: `<b>C 語法重點：</b><span class="c-tip">#pragma omp parallel for</span> 是 OpenMP 平行化指令，讓 CPU 多執行緒同時計算不同的列。你們在 FPGA 上不需要這個。<br><br>
<b>作用：</b>W(d,n) × x(n) → xout(d)。整個模型最花時間的函式！用 int8 乘法 → int32 累加 → 乘以 scale → float。<br>
<b>🔧 FPGA 需實作：</b>你們計畫用 Block RAM + fixed-point，分塊(8或16)處理。`,
    code: `void matmul(float* xout, QuantizedTensor *x,
            QuantizedTensor *w, int n, int d) {
    // W(d,n) @ x(n) → xout(d)
    int i;
    #pragma omp parallel for private(i)  // CPU 平行化
    for (i = 0; i < d; i++) {  // 對輸出的每一維
        float val = 0.0f;
        int32_t ival = 0;      // int32 避免 int8 乘法溢位
        int in = i * n;

        // 按 group 大小(GS=128)分組做乘加
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                // int8 × int8 → 累加到 int32
                ival += ((int32_t)x->q[j+k]) *
                        ((int32_t)w->q[in+j+k]);
            }
            // 每組結束後：int32→float，乘以兩邊的 scale
            val += ((float)ival) *
                   w->s[(in+j)/GS] * x->s[j/GS];
            ival = 0;  // 重置累加器
        }
        xout[i] = val;
    }
}`
  },
  {
    id: 'embedding',
    archId: 'arch-embed',
    tag: 'core', tagLabel: '核心',
    title: 'forward() — Token Embedding 嵌入',
    lines: 'L345-359',
    desc: `<b>C 語法重點：</b><span class="c-tip">memcpy(dst, src, size)</span> 是 C 的記憶體複製函式，把 src 的 size 位元組複製到 dst。比用 for 迴圈快。<br><br>
<b>作用：</b>根據 token ID 查表取出對應的嵌入向量（dim=2048 維的 float 陣列），作為整個 forward pass 的起點。`,
    code: `float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;     // & 取址運算符
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;       // x 是當前的 activation 向量
    int dim = p->dim;      // 2048
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = dim / p->n_heads;  // 64

    // 從嵌入表複製 token 對應的向量到 x
    memcpy(x, w->token_embedding_table + token*dim,
           dim * sizeof(float));
    // token*dim = 偏移量，跳到正確的行
    // 相當於 x = embedding_table[token]`
  },
  {
    id: 'attn-norm',
    archId: 'arch-attn-norm',
    tag: 'attn', tagLabel: '注意力',
    title: 'Attention Pre-Norm（RMSNorm）',
    lines: 'L362-365',
    desc: `<b>作用：</b>在 Self-Attention 之前先做 RMSNorm。注意 <code>l*dim</code> 是因為每一層有自己的 RMSNorm 權重。`,
    code: `for(int l = 0; l < p->n_layers; l++) {  // 迴圈 16 層
    // Attention 前的 RMSNorm
    rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);
    // s->xb = 正規化後的結果
    // w->rms_att_weight + l*dim = 第 l 層的權重起始位置`
  },
  {
    id: 'qkv',
    archId: 'arch-qkv',
    tag: 'attn', tagLabel: '注意力',
    title: 'Q/K/V 投影（Linear Projection）',
    lines: 'L367-371',
    desc: `<b>作用：</b>把正規化後的向量分別投影成 Query、Key、Value。先 quantize 再 matmul 是因為權重存成 int8。<br>
Q: (dim,dim) → 完整維度<br>K,V: (dim,kv_dim) → GQA 用更少的 KV 頭`,
    code: `    // 先把 xb 量化成 int8（為了和 int8 權重做矩陣乘法）
    quantize(&s->xq, s->xb, dim);
    // 三次矩陣乘法：生成 Q, K, V
    matmul(s->q, &s->xq, w->wq + l, dim, dim);     // Query
    matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);  // Key
    matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);  // Value`
  },
  {
    id: 'rope',
    archId: 'arch-rope',
    tag: 'attn', tagLabel: '注意力',
    title: 'RoPE — 旋轉位置編碼',
    lines: 'L373-388',
    desc: `<b>C 語法重點：</b><span class="c-tip">i += 2</span> 表示每次迴圈 i 增加 2，因為 RoPE 是對「相鄰兩個元素」做複數旋轉。<br><br>
<b>作用：</b>注入位置資訊。把 Q 和 K 的相鄰兩個維度當作複數的實部和虛部，乘以旋轉矩陣。<br>
<b>🔧 FPGA 筆記：</b>開會討論 cos/sin 值是可預計算的，可用查表（LUT）取代 CORDIC。`,
    code: `    // RoPE: 對 Q 和 K 做旋轉位置編碼
    for (int i = 0; i < dim; i += 2) {  // 每次處理 2 個元素
        int head_dim = i % head_size;
        // 計算旋轉頻率
        float freq = 1.0f / powf(500000.0f,
                     head_dim / (float)head_size);
        float val = pos * freq;  // pos = 當前位置
        float fcr = cosf(val);   // cos(θ)
        float fci = sinf(val);   // sin(θ)

        // rotn=2 → 同時旋轉 Q 和 K; rotn=1 → 只旋轉 Q
        int rotn = i < kv_dim ? 2 : 1;
        for (int v = 0; v < rotn; v++) {
            float* vec = v == 0 ? s->q : s->k;
            float v0 = vec[i];
            float v1 = vec[i+1];
            // 複數旋轉: (v0+v1i) × (cos+sin·i)
            vec[i]   = v0 * fcr - v1 * fci;
            vec[i+1] = v0 * fci + v1 * fcr;
        }
    }`
  },
  {
    id: 'kvcache',
    archId: 'arch-kvcache',
    tag: 'attn', tagLabel: '注意力',
    title: 'KV Cache 儲存',
    lines: 'L390-395',
    desc: `<b>作用：</b>把這個位置的 K 和 V 存入快取。下次推論時可以直接取用之前位置的 KV，不需重算。這是 Autoregressive 推論的關鍵優化。`,
    code: `    // 計算這一層在 KV Cache 中的偏移量
    int loff = l * p->seq_len * kv_dim;
    float* key_cache_row = s->key_cache + loff + pos*kv_dim;
    float* value_cache_row = s->value_cache + loff + pos*kv_dim;
    // 把 K, V 存入 cache
    memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
    memcpy(value_cache_row, s->v, kv_dim * sizeof(float));`
  },
  {
    id: 'attention',
    archId: 'arch-mha',
    tag: 'attn', tagLabel: '注意力',
    title: 'Multi-Head Attention 多頭注意力',
    lines: 'L397-435',
    desc: `<b>C 語法重點：</b><span class="c-tip">float* q = s->q + h * head_size</span> 是指標算術——指標加上偏移量，指向第 h 個 head 的起始位置。<br><br>
<b>作用：</b>核心的注意力計算：<br>
1. Q·K^T / √d → Attention Score（點積）<br>
2. Softmax → 注意力權重<br>
3. 權重 × V → 加權輸出`,
    code: `    // 多頭注意力，平行處理每個 head
    for (int h = 0; h < p->n_heads; h++) {
        float* q = s->q + h * head_size;  // 第 h 個 head 的 Q
        float* att = s->att + h * p->seq_len;  // 分數緩衝區

        // 1. 計算 Attention Score = Q · K^T / √d
        for (int t = 0; t <= pos; t++) {  // 所有時間步
            float* k = s->key_cache + loff
                     + t*kv_dim + (h/kv_mul)*head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++)
                score += q[i] * k[i];  // 點積
            att[t] = score / sqrtf(head_size);  // 除以 √d
        }

        // 2. Softmax → 轉換為注意力權重
        softmax(att, pos + 1);

        // 3. 加權求和 V
        float* xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));  // 清零
        for (int t = 0; t <= pos; t++) {
            float* v = s->value_cache + loff
                     + t*kv_dim + (h/kv_mul)*head_size;
            float a = att[t];  // 注意力權重
            for (int i = 0; i < head_size; i++)
                xb[i] += a * v[i];  // 加權累加
        }
    }`
  },
  {
    id: 'attn-out',
    archId: 'arch-attn-out',
    tag: 'attn', tagLabel: '注意力',
    title: 'Attention Output + Residual Connection',
    lines: 'L437-444',
    desc: `<b>C 語法重點：</b><span class="c-tip">x[i] += s->xb2[i]</span> 就是 <code>x[i] = x[i] + s->xb2[i]</code>，這就是殘差連接（Residual Connection），把注意力輸出加回原始輸入。`,
    code: `    // Output Projection: 把多頭結果投影回 dim 維度
    quantize(&s->xq, s->xb, dim);
    matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

    // 殘差連接: x = x + Attention(x)
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb2[i];  // 加回原始輸入！
    }`
  },
  {
    id: 'ffn',
    archId: 'arch-ffn',
    tag: 'ffn', tagLabel: 'FFN',
    title: 'FFN + SwiGLU 前饋神經網路',
    lines: 'L446-472',
    desc: `<b>作用：</b>完整的 FFN 區塊：<br>
1. RMSNorm<br>
2. w1(x) 和 w3(x) 兩個線性投影（dim→hidden_dim）<br>
3. SwiGLU 激活：silu(w1(x)) × w3(x)<br>
4. w2 線性投影（hidden_dim→dim）<br>
5. 殘差連接<br>
<b>SiLU(x) = x × σ(x)</b>，σ 是 sigmoid。`,
    code: `    // FFN 前的 RMSNorm
    rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

    // 計算 w1(x) 和 w3(x)
    quantize(&s->xq, s->xb, dim);
    matmul(s->hb,  &s->xq, w->w1 + l, dim, hidden_dim);
    matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

    // SwiGLU 非線性激活
    for (int i = 0; i < hidden_dim; i++) {
        float val = s->hb[i];
        // SiLU(x) = x * sigmoid(x)
        val *= (1.0f / (1.0f + expf(-val)));
        // 逐元素與 w3(x) 相乘
        val *= s->hb2[i];
        s->hb[i] = val;
    }

    // w2 投影：hidden_dim → dim
    quantize(&s->hq, s->hb, hidden_dim);
    matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

    // 殘差連接
    for (int i = 0; i < dim; i++) {
        x[i] += s->xb[i];
    }
}  // ← 16 層迴圈結束`
  },
  {
    id: 'final',
    archId: 'arch-final',
    tag: 'core', tagLabel: '核心',
    title: 'Final RMSNorm → Logits（輸出預測）',
    lines: 'L475-481',
    desc: `<b>作用：</b>16 層迴圈結束後，做最後一次 RMSNorm，然後投影到詞彙表大小（vocab_size），產生每個 token 的分數（logits）。`,
    code: `    // 最終 RMSNorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // Classifier: dim → vocab_size 的矩陣乘法
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;  // 回傳 logits 陣列
}`
  },
  {
    id: 'tokenizer',
    archId: 'arch-tokenizer',
    tag: 'token', tagLabel: '分詞器',
    title: 'BPE Tokenizer 分詞器',
    lines: 'L484-755',
    desc: `<b>作用：</b>將文字轉換為 token ID 序列（encode），或反過來（decode）。使用 Byte Pair Encoding 演算法。<br>
<b>🔧 FPGA 筆記：</b>開會討論決定 tokenizer 由 CPU 處理，先不移到 FPGA。`,
    code: `// BPE 編碼的核心邏輯（簡化版）：
void encode(Tokenizer* t, char *text, ...) {
    // 1. 用 regex 把文字切成小塊
    // 2. 每塊先查表看有沒有完整 token
    // 3. 如果沒有，用 UTF-8 拆成 byte
    // 4. 反覆合併最佳相鄰 pair (BPE 核心)
    while (1) {
        // 找分數最低的相鄰 pair
        // 合併它 → token 數減 1
        // 直到無法再合併
    }
}`
  },
  {
    id: 'sampler',
    archId: 'arch-sampler',
    tag: 'sample', tagLabel: '取樣',
    title: 'Sampler 取樣策略',
    lines: 'L757-898',
    desc: `<b>作用：</b>從 logits 中選出下一個 token。三種策略：<br>
1. <b>Greedy (temperature=0)</b>：直接選機率最高的<br>
2. <b>Temperature Sampling</b>：除以 temperature 後 softmax，隨機取樣<br>
3. <b>Top-p (nucleus)</b>：只從累積機率前 p% 的 token 中取樣`,
    code: `int sample(Sampler* sampler, float* logits) {
    if (sampler->temperature == 0.0f) {
        // 貪心：直接取最大值
        return sample_argmax(logits, vocab_size);
    } else {
        // 除以 temperature（越高越隨機）
        for (int q = 0; q < vocab_size; q++)
            logits[q] /= sampler->temperature;
        // Softmax → 機率分佈
        softmax(logits, vocab_size);
        // 隨機取樣
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1)
            return sample_mult(logits, vocab_size, coin);
        else
            return sample_topp(logits, vocab_size,
                   sampler->topp, probindex, coin);
    }
}`
  },
  {
    id: 'generate',
    archId: 'arch-generate',
    tag: 'infra', tagLabel: '推論',
    title: 'generate() — 自迴歸生成迴圈',
    lines: 'L913-967',
    desc: `<b>作用：</b>整個推論的主迴圈：<br>
1. 呼叫 forward() 取得 logits<br>
2. 用 sampler 選下一個 token<br>
3. decode 成文字印出來<br>
4. 重複直到結束`,
    code: `void generate(Transformer *transformer, Tokenizer *tokenizer,
              Sampler *sampler, char *prompt, int steps) {
    int token = prompt_tokens[0];  // 從第一個 token 開始
    int pos = 0;

    while (pos < steps) {
        // 1. Forward pass → 取得 logits
        float* logits = forward(transformer, token, pos);

        // 2. 選下一個 token
        if (pos < num_prompt_tokens - 1)
            next = prompt_tokens[pos + 1];  // 還在 prompt 中
        else
            next = sample(sampler, logits);  // 自由生成

        // 3. 解碼並印出
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);

        token = next;
        pos++;
    }
}`
  }
];

// ===== Guided Tour Steps (Forward Pass 順序) =====
const TOUR_STEPS = [
  { sectionId: 'embedding', label: '① Token → Embedding' },
  { sectionId: 'attn-norm', label: '② Attention RMSNorm' },
  { sectionId: 'qkv', label: '③ Q/K/V 投影' },
  { sectionId: 'rope', label: '④ RoPE 位置編碼' },
  { sectionId: 'kvcache', label: '⑤ KV Cache 儲存' },
  { sectionId: 'attention', label: '⑥ Multi-Head Attention' },
  { sectionId: 'attn-out', label: '⑦ Output + Residual' },
  { sectionId: 'ffn', label: '⑧ FFN + SwiGLU' },
  { sectionId: 'final', label: '⑨ Final Norm → Logits' },
  { sectionId: 'sampler', label: '⑩ Token 取樣' },
];

let currentStep = -1;

// ===== Initialize App =====
function init() {
  renderCodeSections();
  renderArchDiagram();
  updateNav();
}

// ===== Render Code Sections =====
function renderCodeSections() {
  const panel = document.getElementById('code-panel');
  panel.innerHTML = CODE_SECTIONS.map(s => `
    <div class="code-section" id="section-${s.id}" data-arch="${s.archId}" data-id="${s.id}">
      <div class="section-header" onclick="toggleSection('${s.id}')">
        <span class="section-toggle">▶</span>
        <span class="section-tag tag-${s.tag}">${s.tagLabel}</span>
        <span class="section-title">${s.title}</span>
        <span class="section-lines">${s.lines}</span>
      </div>
      <div class="section-body">
        <div class="section-desc">${s.desc}</div>
        <pre class="code-block"><code>${escapeHtml(s.code)}</code></pre>
      </div>
    </div>
  `).join('');

  // Hover events for bidirectional highlighting
  document.querySelectorAll('.code-section').forEach(el => {
    el.addEventListener('mouseenter', () => {
      const archId = el.dataset.arch;
      highlightArch(archId, true);
    });
    el.addEventListener('mouseleave', () => {
      const archId = el.dataset.arch;
      if (currentStep === -1 || CODE_SECTIONS[currentStep]?.archId !== archId) {
        highlightArch(archId, false);
      }
    });
  });
}

function toggleSection(id) {
  const el = document.getElementById(`section-${id}`);
  el.classList.toggle('open');
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ===== Highlight Helpers =====
function highlightArch(archId, on) {
  const block = document.getElementById(archId);
  if (block) {
    if (on) block.classList.add('active');
    else block.classList.remove('active');
  }
}

function highlightSection(sectionId, on) {
  const el = document.getElementById(`section-${sectionId}`);
  if (el) {
    if (on) {
      el.classList.add('active');
      el.classList.add('open');
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
      el.classList.remove('active');
    }
  }
}

function clearAllHighlights() {
  document.querySelectorAll('.code-section.active').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.block.active').forEach(e => e.classList.remove('active'));
}

// ===== Architecture Diagram Click =====
function archClick(archId) {
  const section = CODE_SECTIONS.find(s => s.archId === archId);
  if (section) {
    clearAllHighlights();
    highlightSection(section.id, true);
    highlightArch(archId, true);
  }
}

// ===== Navigation (Guided Tour) =====
function goStep(delta) {
  clearAllHighlights();
  currentStep += delta;
  if (currentStep < 0) currentStep = 0;
  if (currentStep >= TOUR_STEPS.length) currentStep = TOUR_STEPS.length - 1;
  const step = TOUR_STEPS[currentStep];
  highlightSection(step.sectionId, true);
  const section = CODE_SECTIONS.find(s => s.id === step.sectionId);
  if (section) highlightArch(section.archId, true);
  updateNav();
}

function resetTour() {
  clearAllHighlights();
  currentStep = -1;
  updateNav();
}

function updateNav() {
  const label = document.getElementById('step-label');
  const prevBtn = document.getElementById('btn-prev');
  const nextBtn = document.getElementById('btn-next');
  const fill = document.getElementById('progress-fill');

  if (currentStep === -1) {
    label.innerHTML = '按 <strong>下一步</strong> 開始 Forward Pass 導讀';
    prevBtn.disabled = true;
    fill.style.width = '0%';
  } else {
    label.innerHTML = `<strong>${TOUR_STEPS[currentStep].label}</strong>`;
    prevBtn.disabled = currentStep === 0;
    fill.style.width = ((currentStep + 1) / TOUR_STEPS.length * 100) + '%';
  }
  nextBtn.disabled = currentStep === TOUR_STEPS.length - 1;
}

// ===== Render SVG Architecture Diagram =====
function renderArchDiagram() {
  const container = document.getElementById('arch-panel');
  const blocks = [
    { id: 'arch-tokenizer', y: 10, h: 36, label: 'Tokenizer (BPE)', sub: 'text→token IDs', color: '#79c0ff' },
    { id: 'arch-embed', y: 60, h: 36, label: 'Token Embedding', sub: 'token→vector (2048-d)', color: '#3fb950' },
    { id: 'arch-config', y: 110, h: 34, label: 'Config 超參數', sub: 'dim, heads, layers', color: '#8b949e', x: 258, w: 150 },
    { id: 'arch-weights', y: 152, h: 34, label: 'Weights 結構', sub: 'wq,wk,wv,wo,w1,w2,w3', color: '#8b949e', x: 258, w: 150 },
    { id: 'arch-quant', y: 194, h: 34, label: 'Quantize / Dequantize', sub: 'Int8 ↔ Float32', color: '#d29922', x: 258, w: 150 },
    { id: 'arch-rmsnorm', y: 236, h: 34, label: 'RMSNorm（通用）', sub: '√ 運算', color: '#3fb950', x: 258, w: 150 },
    { id: 'arch-matmul', y: 278, h: 34, label: 'MatMul（通用）', sub: 'Int8 矩陣乘法', color: '#3fb950', x: 258, w: 150 },
    { id: 'arch-softmax', y: 320, h: 34, label: 'Softmax（通用）', sub: '機率正規化', color: '#3fb950', x: 258, w: 150 },
    { id: 'arch-attn-norm', y: 116, h: 32, label: 'RMSNorm (Attn)', sub: '正規化', color: '#bc8cff' },
    { id: 'arch-qkv', y: 162, h: 32, label: 'Q, K, V 投影', sub: '3×MatMul', color: '#bc8cff' },
    { id: 'arch-rope', y: 208, h: 32, label: 'RoPE 位置編碼', sub: 'cos/sin 旋轉', color: '#bc8cff' },
    { id: 'arch-kvcache', y: 254, h: 32, label: 'KV Cache', sub: '儲存 K,V', color: '#bc8cff' },
    { id: 'arch-mha', y: 300, h: 40, label: 'Multi-Head Attention', sub: 'Q·K^T→Softmax→×V', color: '#bc8cff' },
    { id: 'arch-attn-out', y: 354, h: 32, label: 'Output + Residual', sub: 'Wo·Attn + x', color: '#bc8cff' },
    { id: 'arch-ffn', y: 400, h: 50, label: 'FFN + SwiGLU', sub: 'w1,w3→SiLU→w2 + Residual', color: '#f85149' },
    { id: 'arch-final', y: 468, h: 36, label: 'Final RMSNorm → Logits', sub: 'dim→vocab_size', color: '#3fb950' },
    { id: 'arch-sampler', y: 518, h: 36, label: 'Sampler', sub: 'argmax / top-p', color: '#f778ba' },
    { id: 'arch-generate', y: 568, h: 36, label: '生成迴圈 (generate)', sub: 'autoregressive loop', color: '#8b949e' },
  ];

  const mainX = 20, mainW = 210;

  let svg = `<svg class="arch-diagram" viewBox="0 0 420 620" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#6e7681"/>
      </marker>
    </defs>
    <!-- Layer bracket -->
    <rect class="layer-bracket" x="10" y="108" width="235" height="350" />
    <text x="125" y="466" font-size="10" fill="#6e7681" text-anchor="middle" font-style="italic">× 16 layers</text>
  `;

  blocks.forEach(b => {
    const bx = b.x || mainX;
    const bw = b.w || mainW;
    svg += `
    <g class="block" id="${b.id}" onclick="archClick('${b.id}')"
       onmouseenter="highlightSection('${CODE_SECTIONS.find(s => s.archId === b.id)?.id}',true)"
       onmouseleave="highlightSection('${CODE_SECTIONS.find(s => s.archId === b.id)?.id}',false)">
      <rect x="${bx}" y="${b.y}" width="${bw}" height="${b.h}" rx="6"
            fill="${b.color}15" stroke="${b.color}" stroke-width="1.5"/>
      <text class="label-text" x="${bx + bw / 2}" y="${b.y + b.h / 2 - (b.sub ? 4 : 0)}" font-size="${bw < 180 ? 10 : 12}">${b.label}</text>
      ${b.sub ? `<text class="label-sub" x="${bx + bw / 2}" y="${b.y + b.h / 2 + 10}">${b.sub}</text>` : ''}
    </g>`;
  });

  // Arrows (main flow)
  const arrows = [
    [125, 46, 125, 60],   // tokenizer→embed
    [125, 96, 125, 116],  // embed→attn-norm
    [125, 148, 125, 162], // norm→qkv
    [125, 194, 125, 208], // qkv→rope
    [125, 240, 125, 254], // rope→kvcache
    [125, 286, 125, 300], // kvcache→mha
    [125, 340, 125, 354], // mha→attn-out
    [125, 386, 125, 400], // attn-out→ffn
    [125, 450, 125, 468], // ffn→final
    [125, 504, 125, 518], // final→sampler
    [125, 554, 125, 568], // sampler→generate
  ];
  arrows.forEach(([x1, y1, x2, y2]) => {
    svg += `<line class="arrow" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
  });

  svg += '</svg>';
  container.innerHTML = `<div class="arch-title">Llama 3.2 (1B) 架構圖</div>${svg}`;
}

window.addEventListener('DOMContentLoaded', init);
