"""Transformer 最小实现（numpy 版，列向量堆叠风格，骨架版）。

风格约定（与 03 Self-Attention/self-attention.py 一致）：
    - 列向量堆叠：X 形状 (d_model, N)，N 个列向量
    - 无 batch 维：处理单条序列
    - 权重按 (out, in) 存：W (d_out, d_in)，所以是 W @ X
    - softmax 按列做（axis=0）：因为 query 在列
    - 注意力分数 A = Kᵀ @ Q，形状 (N_kv, N_q)，列 j 是 qʲ 对所有 k 的分数

实现顺序建议：
    1. softmax（HW4 已有，直接搬）
    2. layer_norm
    3. positional_encoding
    4. multi_head_attention（核心，HW4 基础上加 mask 和 Q/K/V 解耦）
    5. make_causal_mask
    6. feed_forward
    7. encoder_block / encoder
    8. decoder_block / decoder
    9. transformer（端到端）

跑通主程序即实现正确。
"""

import numpy as np


# ============================================================
# 基础组件
# ============================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax，沿指定轴归一化使该轴和为 1。

    输入 x 任意形状，返回同形状，沿 axis 归一化。

    提示：先减最大值（keepdims=True），再 exp、再除以和。
    """
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """层归一化（笔记 3.5 节）：沿第 0 维（特征维）做。

    列向量堆叠下 x 形状 (d_model, N)：第 0 维是特征，第 1 维是序列位置。
    LN 在"同一个 token 内部"算统计量 → axis=0。

    公式：x'_i = (x_i - mean) / sqrt(var + eps)
    返回同形状 (d_model, N)，每列（每个 token）的 mean≈0、std≈1。

    提示：mean 和 var 都沿 axis=0 算，keepdims=True 便于广播。
    """
    mean = np.mean(x, axis=0, keepdims=True)
    var = np.var(x, axis=0, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """正弦位置编码（笔记 3.7 节，Vaswani 2017 公式）。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    返回形状 (d_model, seq_len)，列向量堆叠，直接加到 token embedding 上。
    偶数行用 sin，奇数行用 cos。

    提示：
        pos   形状 (1, seq_len)
        div   形状 (d_model//2,)
        pe[0::2, :] = sin(pos * div[:, None])
        pe[1::2, :] = cos(pos * div[:, None])
    """
    pe = np.zeros((d_model, seq_len))
    pos = np.arange(seq_len)[None, :]  # (1, seq_len)
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model//2,)
    pe[0::2, :] = np.sin(pos * div[:, None])  # 偶数行
    pe[1::2, :] = np.cos(pos * div[:, None])  # 奇数行
    return pe


# ============================================================
# 多头注意力（核心函数，支持自注意力 / 掩码注意力 / cross-attention）
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
        2. 掩码自注意力：Q_in = K_in = V_in = X，mask=因果掩码（解码器）
        3. cross-attention：Q_in 来自解码器，K_in/V_in 来自编码器输出

    计算流程（在 HW4 multi_head_attention 基础上加 mask 和 Q/K/V 解耦）：
        1. 投影：Q = W_Q @ Q_in  (h*d_k, N_q)
                 K = W_K @ K_in  (h*d_k, N_kv)
                 V = W_V @ V_in  (h*d_v, N_kv)
        2. 拆头：(h*d_k, N) → (h, d_k, N) / (h, d_v, N)
        3. 每头：
            scores = Kᵀ @ Q  (N_kv, N_q)，列 j 是 qʲ 对所有 k 的分数
            scores *= 1/sqrt(d_k)
            如有 mask：scores = where(mask==0, -inf, scores)
            weights = softmax(scores, axis=0)  按列归一化
            out_i = (weights @ Vᵀ).T  (d_v, N_q)
        4. 拼接 → (h*d_v, N_q)
        5. 输出投影：O = W_O @ concat  (d_model, N_q)

    返回 O (d_model, N_q)。
    """
    # 1. 投影
    Q = W_Q @ Q_in  # (h*d_k, N_q)
    K = W_K @ K_in  # (h*d_k, N_kv)
    V = W_V @ V_in  # (h*d_v, N_kv)

    # 2. 拆头
    h = num_heads
    d_k = W_Q.shape[0] // h
    d_v = W_V.shape[0] // h

    N_q = Q_in.shape[1]
    N_k = K_in.shape[1]
    N_v = V_in.shape[1]

    Q_heads = Q.reshape(h, d_k, N_q)
    K_heads = K.reshape(h, d_k, N_k)   
    V_heads = V.reshape(h, d_v, N_v)

    # 3. 每头：
    scale = 1.0 / np.sqrt(d_k)  # 缩放因子，防止内积过大导致 softmax 饱和
    outputs = []  # 收集每个头的输出 (d_v, N_q)


def make_causal_mask(seq_len: int) -> np.ndarray:
    """因果掩码（笔记 5.4 节）：下三角为 1（可见），上三角为 0（屏蔽）。

    返回形状 (seq_len, seq_len)，与注意力分数 (N_kv, N_q) 同形。
    与 multi_head_attention 的 mask 参数对齐：0 表示屏蔽。

    提示：np.tril(np.ones(...))。
    """
    raise NotImplementedError


# ============================================================
# 前馈网络 FFN
# ============================================================

def feed_forward(
    X: np.ndarray,    # (d_model, N)
    W1: np.ndarray,   # (d_ff, d_model)
    b1: np.ndarray,   # (d_ff,)
    W2: np.ndarray,   # (d_model, d_ff)
    b2: np.ndarray,   # (d_model,)
) -> np.ndarray:
    """两层全连接 + ReLU（笔记 3.3 节）。

    列向量堆叠下 X (d_model, N)，权重按 (out, in) 存：
        H = ReLU(W1 @ X + b1)  (d_ff, N)
        O = W2 @ H + b2        (d_model, N)

    返回 (d_model, N)。

    提示：bias 形状 (d_ff,) 要扩成 (d_ff, 1) 才能广播到 (d_ff, N) → b1[:, None]。
    """
    raise NotImplementedError


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

    输入 X (d_model, N)，输出 (d_model, N)，维度不变可堆叠 N 次。

    提示：两个子层，每个都是"子层输出 + 残差 → LN"。
    """
    raise NotImplementedError


def encoder(
    X: np.ndarray,
    blocks_params: list[tuple],
    num_heads: int,
) -> np.ndarray:
    """编码器：N 个 block 堆叠（笔记 3.7 节）。

    blocks_params 是元组列表，每项是
        (W_Q, W_K, W_V, W_O, FFN_W1, FFN_b1, FFN_W2, FFN_b2)
    """
    raise NotImplementedError


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
           —— Q/K/V 都来自 X，加因果掩码
        2. cross-attention + 残差 + LN
           —— Q 来自 X（解码器），K/V 来自 enc_out（编码器）
        3. FFN + 残差 + LN

    输入 X (d_model, N_tgt)，输出 (d_model, N_tgt)，维度不变。

    提示：子层 1 的 mask 用 make_causal_mask(N_tgt) 生成。
    """
    raise NotImplementedError


def decoder(
    X: np.ndarray,
    enc_out: np.ndarray,
    blocks_params: list[tuple],
    num_heads: int,
) -> np.ndarray:
    """解码器：N 个 block 堆叠。"""
    raise NotImplementedError


# ============================================================
# 端到端 Transformer
# ============================================================

def transformer(
    src: np.ndarray,     # (d_model, N_src)
    tgt: np.ndarray,     # (d_model, N_tgt)
    d_model: int,
    num_heads: int,
    enc_blocks: list[tuple],
    dec_blocks: list[tuple],
    output_proj: np.ndarray,  # (vocab_size, d_model)
) -> np.ndarray:
    """完整 Transformer 前向（训练模式，teacher forcing）。

    流程：
        1. 加位置编码：src + PE(N_src)，tgt + PE(N_tgt)
        2. enc_out = encoder(src)
        3. dec_out = decoder(tgt, enc_out)
        4. logits = output_proj @ dec_out  (vocab_size, N_tgt)

    返回 logits (vocab_size, N_tgt)。
    """
    raise NotImplementedError


# ============================================================
# 参数初始化辅助（这些已实现，直接用即可）
# ============================================================

def init_encoder_block_params(d_model: int, num_heads: int, d_ff: int) -> tuple:
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
    d_k = d_v = d_model // num_heads
    return (
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_Q1
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_K1
        np.random.randn(num_heads * d_v, d_model) * 0.1,  # W_V1
        np.random.randn(d_model, num_heads * d_v) * 0.1,  # W_O1
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_Q2
        np.random.randn(num_heads * d_k, d_model) * 0.1,  # W_K2
        np.random.randn(num_heads * d_v, d_model) * 0.1,  # W_V2
        np.random.randn(d_model, num_heads * d_v) * 0.1,  # W_O2
        np.random.randn(d_ff, d_model) * 0.1,              # FFN_W1
        np.zeros(d_ff),                                    # FFN_b1
        np.random.randn(d_model, d_ff) * 0.1,             # FFN_W2
        np.zeros(d_model),                                 # FFN_b2
    )


# ============================================================
# 验证：跑通这个主程序即实现正确
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    # 小例子
    d_model, d_ff = 4, 8
    h = 2  # 头数（d_k = d_v = d_model // h = 2）
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
    # 预期：下三角 1，上三角 0

    # 层归一化
    X_ln = np.random.randn(d_model, N_tgt)
    X_ln_out = layer_norm(X_ln)
    print(f"\n层归一化:  输入 {X_ln.shape} → 输出 {X_ln_out.shape}")
    print(f"  LN 后 每列 mean={np.round(np.mean(X_ln_out, axis=0), 4)}（应≈0）")
    print(f"  LN 后 每列 std ={np.round(np.std(X_ln_out, axis=0), 4)}（应≈1）")

    # 位置编码
    pe = positional_encoding(N_src, d_model)
    print(f"\n位置编码:  形状 {pe.shape}")
    print(f"  PE[:, 0] = {np.round(pe[:, 0], 4)}（位置 0）")
    print(f"  PE[:, 1] = {np.round(pe[:, 1], 4)}（位置 1，应与位置 0 不同）")

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
    print(f"输出 enc_out: {enc_out.shape}  (d_model, N_src)  ← 维度应不变")

    # --- 解码器 ---
    print("\n" + "=" * 60)
    print("解码器")
    print("=" * 60)
    tgt = np.random.randn(d_model, N_tgt)
    dec_blocks = [init_decoder_block_params(d_model, h, d_ff) for _ in range(2)]
    dec_out = decoder(tgt, enc_out, dec_blocks, h)
    print(f"输入 tgt:    {tgt.shape}  (d_model, N_tgt)")
    print(f"输入 enc_out: {enc_out.shape}  (d_model, N_src)")
    print(f"输出 dec_out: {dec_out.shape}  (d_model, N_tgt)  ← 维度应不变")

    # --- 端到端 Transformer ---
    print("\n" + "=" * 60)
    print("端到端 Transformer")
    print("=" * 60)
    output_proj = np.random.randn(vocab_size, d_model) * 0.1
    logits = transformer(src, tgt, d_model, h, enc_blocks, dec_blocks, output_proj)
    print(f"源序列 src:    {src.shape}")
    print(f"目标序列 tgt:  {tgt.shape}")
    print(f"最终 logits:  {logits.shape}  (vocab_size, N_tgt)")
    print(f"\n[OK] 跑通即实现正确，vocab 维度 = {vocab_size}")
