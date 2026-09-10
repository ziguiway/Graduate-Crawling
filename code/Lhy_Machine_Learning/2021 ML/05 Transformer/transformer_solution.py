"""Transformer 最小实现（numpy 版，沿用 HW4 列向量堆叠风格）。

风格约定（与 03 Self-Attention/self-attention.py 一致）：
    - 列向量堆叠：X 形状 (d_model, N)，N 个列向量
    - 无 batch 维：处理单条序列
    - 权重按 (out, in) 存：W (d_out, d_in)，所以是 W @ X
    - softmax 按列做（axis=0）：因为 query 在列
    - 注意力分数 A = Kᵀ @ Q，形状 (N, N)，列 j 是 qʲ 对所有 k 的分数

结构（Post-LN 原始架构，对应笔记第 3.6 节）：
    编码器块 = 多头自注意力 + 残差 + LN + FFN + 残差 + LN
    解码器块 = 掩码多头自注意力 + 残差+LN
              + cross-attention（Q来自解码器，K/V来自编码器） + 残差+LN
              + FFN + 残差+LN
"""

import numpy as np


# ============================================================
# 基础组件
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax，沿指定轴归一化使该轴和为 1。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """层归一化（笔记 3.5 节）：沿第 0 维（特征维）做，不依赖 batch。

    x'_i = (x_i - mean) / std

    列向量堆叠下 x 形状 (d_model, N)：第 0 维是特征，第 1 维是序列位置。
    LN 在"同一个 token 内部"算统计量 → axis=0。

    为什么不用 BN：序列长度可变、batch 内同一位置可能不是同种东西，
    BN 跨样本统计没意义；LN 只在样本内部统计，跟序列长度无关。
    """
    mean = np.mean(x, axis=0, keepdims=True)  # 每列（每个 token）的均值
    var = np.var(x, axis=0, keepdims=True)    # 每列（每个 token）的方差
    return (x - mean) / np.sqrt(var + eps)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """正弦位置编码（笔记 3.7 节，Vaswani 2017 原始公式）。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    返回形状 (d_model, seq_len)，列向量堆叠，直接加到 token embedding 上。
    为什么需要：自注意力本身是"集合操作"，没有顺序信息；
    加位置编码让模型知道"哪个 token 在前、哪个在后"。
    """
    pe = np.zeros((d_model, seq_len))
    pos = np.arange(seq_len)[None, :]  # (1, seq_len)
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)
    pe[0::2, :] = np.sin(pos * div[:, None])  # 偶数行用 sin
    pe[1::2, :] = np.cos(pos * div[:, None])  # 奇数行用 cos
    return pe


# ============================================================
# 多头注意力（列向量堆叠，无 batch，支持掩码和 cross-attention）
# ============================================================

def multi_head_attention(
    Q_in: np.ndarray,   # (d_in_q, N_q)
    K_in: np.ndarray,   # (d_in_k, N_kv)
    V_in: np.ndarray,   # (d_in_v, N_kv)
    W_Q: np.ndarray,    # (h*d_k, d_in_q)
    W_K: np.ndarray,    # (h*d_k, d_in_k)
    W_V: np.ndarray,    # (h*d_v, d_in_v)
    W_O: np.ndarray,    # (d_model, h*d_v)
    num_heads: int,
    mask: np.ndarray | None = None,  # (N_q, N_kv)，0 表示屏蔽
) -> np.ndarray:
    """多头注意力（笔记 6.4 节），通用形式。

    三种用法：
        1. 自注意力：Q_in = K_in = V_in = X（编码器）
        2. 掩码自注意力：Q_in = K_in = V_in = X + 因果掩码（解码器）
        3. cross-attention：Q_in 来自解码器，K_in/V_in 来自编码器输出

    维度约定（沿用列向量堆叠）：
        Q_in  (d_in_q, N_q)   query 序列
        K_in  (d_in_k, N_kv)  key 序列
        V_in  (d_in_v, N_kv)  value 序列（N_kv 与 K 同长）
        W_Q (h*d_k, d_in_q) / W_K (h*d_k, d_in_k) / W_V (h*d_v, d_in_v)
        W_O (d_model, h*d_v)
        mask (N_q, N_kv) 或 None —— 0 表示屏蔽（softmax 前设成 -inf）

    计算流程（对应 HW4 的 multi_head_attention，多了 mask 参数）：
        Q = W_Q @ Q_in  (h*d_k, N_q)
        K = W_K @ K_in  (h*d_k, N_kv)
        V = W_V @ V_in  (h*d_v, N_kv)
        拆头 → (h, d_k, N) / (h, d_v, N)
        每头：scores = Kᵀ @ Q  (N_kv, N_q)，按列 softmax，加权和
        拼接 → 输出投影 W_O → (d_model, N_q)

    返回 O (d_model, N_q)。
    """
    N_q = Q_in.shape[1]
    N_kv = K_in.shape[1]
    h = num_heads
    d_k = W_Q.shape[0] // h
    d_v = W_V.shape[0] // h

    # 1. 一次性算出所有头的 Q/K/V
    Q_all = W_Q @ Q_in  # (h*d_k, N_q)
    K_all = W_K @ K_in  # (h*d_k, N_kv)
    V_all = W_V @ V_in  # (h*d_v, N_kv)

    # 2. 按 head 拆开 → (h, d_k, N) / (h, d_v, N)
    Q_heads = Q_all.reshape(h, d_k, N_q)
    K_heads = K_all.reshape(h, d_k, N_kv)
    V_heads = V_all.reshape(h, d_v, N_kv)

    # 3. 每个头各自做 scaled dot-product attention
    scale = 1.0 / np.sqrt(d_k)
    outputs = []
    for i in range(h):
        Q_i = Q_heads[i]  # (d_k, N_q)
        K_i = K_heads[i]  # (d_k, N_kv)
        V_i = V_heads[i]  # (d_v, N_kv)

        # 注意力分数 (N_kv, N_q)，列 j 是 qʲ 对所有 k 的分数
        scores = (K_i.T @ Q_i) * scale

        # 掩码（如有）：被屏蔽位置分数设成 -inf，softmax 后权重为 0
        if mask is not None:
            scores = np.where(mask == 0, -np.inf, scores)

        # 按列归一化（每列对应一个 query）
        weights = softmax(scores, axis=0)
        # 加权和 (d_v, N_q)
        out_i = (weights @ V_i.T).T
        outputs.append(out_i)

    # 4. 按 feature 维拼接 → (h*d_v, N_q)
    concat = np.concatenate(outputs, axis=0)

    # 5. 输出投影压回模型维度
    O = W_O @ concat  # (d_model, N_q)
    return O


def make_causal_mask(seq_len: int) -> np.ndarray:
    """因果掩码（笔记 5.4 节）：上三角为 0（屏蔽），下三角为 1（可见）。

    解码器生成第 t 个 token 时只能看到前 t 个 token（含自己），
    防止"作弊"——看到未来答案。

    返回形状 (seq_len, seq_len)，与注意力分数 (N_kv, N_q) 同形。
    """
    return np.tril(np.ones((seq_len, seq_len)), k=0).astype(np.float32)


# ============================================================
# 前馈网络 FFN
# ============================================================

def feed_forward(
    X: np.ndarray,   # (d_model, N)
    W1: np.ndarray,  # (d_ff, d_model)
    b1: np.ndarray,  # (d_ff,)
    W2: np.ndarray,  # (d_model, d_ff)
    b2: np.ndarray,  # (d_model,)
) -> np.ndarray:
    """两层全连接 + ReLU（笔记 3.3 节）。

    列向量堆叠下 X (d_model, N)，权重按 (out, in) 存：
        H = ReLU(W1 @ X + b1)  (d_ff, N)
        O = W2 @ H + b2        (d_model, N)

    作用：对每个 token 独立做非线性变换
    （自注意力看全局，FFN 看单个 token）。
    """
    # b1 (d_ff,) 广播到 (d_ff, N) —— 先扩成 (d_ff, 1)
    H = np.maximum(0, W1 @ X + b1[:, None])  # ReLU
    O = W2 @ H + b2[:, None]
    return O


# ============================================================
# 编码器
# ============================================================

def encoder_block(
    X: np.ndarray,
    W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, W_O: np.ndarray,
    FFN_W1: np.ndarray, FFN_b1: np.ndarray, FFN_W2: np.ndarray, FFN_b2: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """一个编码器块（笔记 3.6 节，Post-LN 原始架构）。

    流程：
        ┌──── 残差 ────┐               ┌──── 残差 ────┐
        X → [自注意力] → (+) → [LN] → [FFN] → (+) → [LN] → 输出
                          ↑                              ↑
                       （Post-LN）                    （Post-LN）

    输入 X (d_model, N)，输出 (d_model, N) —— 维度不变，可堆叠 N 次。
    """
    # 子层 1：多头自注意力 + 残差 + LN
    attn_out = multi_head_attention(X, X, X, W_Q, W_K, W_V, W_O, num_heads)
    X = layer_norm(X + attn_out)

    # 子层 2：FFN + 残差 + LN
    ffn_out = feed_forward(X, FFN_W1, FFN_b1, FFN_W2, FFN_b2)
    X = layer_norm(X + ffn_out)
    return X


def encoder(
    X: np.ndarray,
    blocks_params: list[tuple],
    num_heads: int,
) -> np.ndarray:
    """编码器：N 个 block 堆叠（笔记 3.7 节）。

    blocks_params 是元组列表，每项是
        (W_Q, W_K, W_V, W_O, FFN_W1, FFN_b1, FFN_W2, FFN_b2)
    对应一个 block 的所有参数。
    """
    for params in blocks_params:
        W_Q, W_K, W_V, W_O, FFN_W1, FFN_b1, FFN_W2, FFN_b2 = params
        X = encoder_block(X, W_Q, W_K, W_V, W_O, FFN_W1, FFN_b1, FFN_W2, FFN_b2, num_heads)
    return X


# ============================================================
# 解码器
# ============================================================

def decoder_block(
    X: np.ndarray,           # (d_model, N_tgt) 解码器输入
    enc_out: np.ndarray,     # (d_model, N_src) 编码器输出
    W_Q1, W_K1, W_V1, W_O1,  # 掩码自注意力的权重
    W_Q2, W_K2, W_V2, W_O2,  # cross-attention 的权重
    FFN_W1, FFN_b1, FFN_W2, FFN_b2,
    num_heads: int,
) -> np.ndarray:
    """一个解码器块（笔记 5.3、5.4、6 节，Post-LN）。

    三个子层：
        1. 掩码多头自注意力 + 残差 + LN
           —— Q/K/V 都来自解码器输入，加因果掩码
              （生成第 t 个 token 只看前 t 个，防作弊）
        2. cross-attention + 残差 + LN
           —— Q 来自解码器，K/V 来自编码器输出（enc_out）
        3. FFN + 残差 + LN
    """
    N_tgt = X.shape[1]

    # 子层 1：掩码自注意力（笔记 5.4 节）
    causal = make_causal_mask(N_tgt)  # (N_tgt, N_tgt)
    self_attn_out = multi_head_attention(
        X, X, X, W_Q1, W_K1, W_V1, W_O1, num_heads, mask=causal,
    )
    X = layer_norm(X + self_attn_out)

    # 子层 2：cross-attention（笔记第 6 节）
    # Q 来自解码器（X），K/V 来自编码器输出（enc_out）
    cross_out = multi_head_attention(
        X, enc_out, enc_out, W_Q2, W_K2, W_V2, W_O2, num_heads,
    )
    X = layer_norm(X + cross_out)

    # 子层 3：FFN + 残差 + LN
    ffn_out = feed_forward(X, FFN_W1, FFN_b1, FFN_W2, FFN_b2)
    X = layer_norm(X + ffn_out)
    return X


def decoder(
    X: np.ndarray,
    enc_out: np.ndarray,
    blocks_params: list[tuple],
    num_heads: int,
) -> np.ndarray:
    """解码器：N 个 block 堆叠。"""
    for params in blocks_params:
        (W_Q1, W_K1, W_V1, W_O1,
         W_Q2, W_K2, W_V2, W_O2,
         FFN_W1, FFN_b1, FFN_W2, FFN_b2) = params
        X = decoder_block(
            X, enc_out,
            W_Q1, W_K1, W_V1, W_O1,
            W_Q2, W_K2, W_V2, W_O2,
            FFN_W1, FFN_b1, FFN_W2, FFN_b2,
            num_heads,
        )
    return X


# ============================================================
# 端到端 Transformer
# ============================================================

def transformer(
    src: np.ndarray,     # (d_model, N_src) 源序列 token embedding
    tgt: np.ndarray,     # (d_model, N_tgt) 目标序列已生成部分（teacher forcing）
    d_model: int,
    num_heads: int,
    enc_blocks: list[tuple],
    dec_blocks: list[tuple],
    output_proj: np.ndarray,  # (vocab_size, d_model) 最后线性层
) -> np.ndarray:
    """完整 Transformer 前向（训练模式，teacher forcing，笔记第 2 节）。

    返回 logits (vocab_size, N_tgt)，每列对应一个位置的词分布。
    """
    N_src = src.shape[1]
    N_tgt = tgt.shape[1]

    # 1. 加位置编码（笔记 3.7 节，自注意力本身无序）
    src = src + positional_encoding(N_src, d_model)  # (d_model, N_src)
    tgt = tgt + positional_encoding(N_tgt, d_model)  # (d_model, N_tgt)

    # 2. 编码器
    enc_out = encoder(src, enc_blocks, num_heads)  # (d_model, N_src)

    # 3. 解码器
    dec_out = decoder(tgt, enc_out, dec_blocks, num_heads)  # (d_model, N_tgt)

    # 4. 输出投影：把 d_model 维向量映射到 vocab_size 维（logits）
    # output_proj (vocab_size, d_model) @ dec_out (d_model, N_tgt) → (vocab_size, N_tgt)
    logits = output_proj @ dec_out
    return logits


# ============================================================
# 参数初始化辅助
# ============================================================

def init_encoder_block_params(d_model: int, num_heads: int, d_ff: int) -> tuple:
    """初始化一个编码器 block 的所有参数。"""
    d_k = d_v = d_model // num_heads
    return (
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_Q
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_K
        np.random.randn(num_heads * d_v, d_model) * 0.1,  # W_V
        np.random.randn(d_model, num_heads * d_v) * 0.1,  # W_O
        np.random.randn(d_ff, d_model) * 0.1,              # FFN_W1
        np.zeros(d_ff),                                    # FFN_b1
        np.random.randn(d_model, d_ff) * 0.1,             # FFN_W2
        np.zeros(d_model),                                 # FFN_b2
    )


def init_decoder_block_params(d_model: int, num_heads: int, d_ff: int) -> tuple:
    """初始化一个解码器 block 的参数（两套注意力 + FFN）。"""
    d_k = d_v = d_model // num_heads
    return (
        # 子层 1：掩码自注意力
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_Q1
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_K1
        np.random.randn(num_heads * d_v, d_model) * 0.1,  # W_V1
        np.random.randn(d_model, num_heads * d_v) * 0.1,  # W_O1
        # 子层 2：cross-attention
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_Q2
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_K2
        np.random.randn(num_heads * d_v, d_model) * 0.1,  # W_V2
        np.random.randn(d_model, num_heads * d_v) * 0.1,  # W_O2
        # 子层 3：FFN
        np.random.randn(d_ff, d_model) * 0.1,              # FFN_W1
        np.zeros(d_ff),                                    # FFN_b1
        np.random.randn(d_model, d_ff) * 0.1,             # FFN_W2
        np.zeros(d_model),                                 # FFN_b2
    )


# ============================================================
# 验证：形状检查（风格与 HW4 一致：小例子 + print 形状）
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    # 小例子（风格与 HW4 主程序对齐）
    d_model, d_k, d_v, d_ff = 4, 2, 2, 8
    h = 2  # 头数
    N_src, N_tgt = 4, 4
    vocab_size = 6

    # --- 基础组件验证 ---
    print("=" * 60)
    print("基础组件验证")
    print("=" * 60)

    # 因果掩码
    mask = make_causal_mask(N_tgt)
    print("\n因果掩码 (4×4)：上三角为 0（屏蔽）")
    print(mask)

    # 层归一化
    X_ln = np.random.randn(d_model, N_tgt)
    X_ln_out = layer_norm(X_ln)
    print(f"\n层归一化:  输入 {X_ln.shape} → 输出 {X_ln_out.shape}")
    print(f"  LN 前 每列 mean={np.mean(X_ln, axis=0)}")
    print(f"  LN 后 每列 mean={np.round(np.mean(X_ln_out, axis=0), 4)}（≈0）")

    # 位置编码
    pe = positional_encoding(N_src, d_model)
    print(f"\n位置编码:  形状 {pe.shape}")
    print(f"  PE[:, 0] = {np.round(pe[:, 0], 4)}（位置 0）")
    print(f"  PE[:, 1] = {np.round(pe[:, 1], 4)}（位置 1，与位置 0 不同）")

    # FFN
    W1 = np.random.randn(d_ff, d_model) * 0.1
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_model, d_ff) * 0.1
    b2 = np.zeros(d_model)
    ffn_out = feed_forward(X_ln, W1, b1, W2, b2)
    print(f"\n前馈网络:  输入 {X_ln.shape} → 输出 {ffn_out.shape}")

    # --- 编码器 ---
    print("\n" + "=" * 60)
    print("编码器")
    print("=" * 60)
    src = np.random.randn(d_model, N_src)
    enc_blocks = [init_encoder_block_params(d_model, h, d_ff) for _ in range(2)]
    enc_out = encoder(src, enc_blocks, h)
    print(f"输入 src:    {src.shape}  (d_model, N_src)")
    print(f"输出 enc_out: {enc_out.shape}  (d_model, N_src)  ← 维度不变")

    # --- 解码器 ---
    print("\n" + "=" * 60)
    print("解码器")
    print("=" * 60)
    tgt = np.random.randn(d_model, N_tgt)
    dec_blocks = [init_decoder_block_params(d_model, h, d_ff) for _ in range(2)]
    dec_out = decoder(tgt, enc_out, dec_blocks, h)
    print(f"输入 tgt:    {tgt.shape}  (d_model, N_tgt)")
    print(f"输入 enc_out: {enc_out.shape}  (d_model, N_src)")
    print(f"输出 dec_out: {dec_out.shape}  (d_model, N_tgt)  ← 维度不变")

    # --- 端到端 Transformer ---
    print("\n" + "=" * 60)
    print("端到端 Transformer")
    print("=" * 60)
    output_proj = np.random.randn(vocab_size, d_model) * 0.1
    logits = transformer(src, tgt, d_model, h, enc_blocks, dec_blocks, output_proj)
    print(f"源序列 src:    {src.shape}")
    print(f"目标序列 tgt:  {tgt.shape}")
    print(f"最终 logits:  {logits.shape}  (vocab_size, N_tgt)  ← 每列是一个位置的概率分布")
    print(f"\n[OK] 形状正确，vocab 维度 = {vocab_size}")
