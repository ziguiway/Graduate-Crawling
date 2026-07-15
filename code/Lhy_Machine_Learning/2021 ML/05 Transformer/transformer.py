"""Transformer 最小实现（numpy 版）。

约定（与 PyTorch 工程实践对齐，行向量堆叠）：
    - 单序列：X 形状 (N, d_model)，N 个 token，每个 d_model 维
    - 带 batch：形状 (B, N, d_model)，B 是 batch size
    - 多头注意力内部：(B, h, N, d_k) —— 第 0 维 batch，第 1 维头，第 2 维序列，第 3 维特征

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
    """数值稳定的 softmax。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """层归一化：沿最后一维（特征维）做，不依赖 batch。

    笔记 3.5 节：x'_i = (x_i - mean) / std
    为什么不用 BN：序列长度可变、batch 内同一位置可能不是同种东西，
    BN 跨样本统计没意义；LN 只在样本内部统计，跟序列长度无关。
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """正弦位置编码（笔记 3.7 节，Vaswani 2017 原始公式）。

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    返回形状 (seq_len, d_model)，直接加到 token embedding 上。
    为什么需要：自注意力本身是"集合操作"，没有顺序信息；
    加位置编码让模型知道"哪个 token 在前、哪个在后"。
    """
    pe = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]  # (seq_len, 1)
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)
    pe[:, 0::2] = np.sin(pos * div)  # 偶数维用 sin
    pe[:, 1::2] = np.cos(pos * div)  # 奇数维用 cos
    return pe


# ============================================================
# 多头注意力（带 batch，行向量堆叠，支持掩码和 cross-attention）
# ============================================================

def multi_head_attention(
    Q_in: np.ndarray,
    K_in: np.ndarray,
    V_in: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """多头注意力，通用形式：自注意力 / cross-attention / 掩码注意力都靠它。

    输入：
        Q_in  (B, N_q, d_in_q)  —— query 序列
        K_in  (B, N_kv, d_in_k) —— key 序列
        V_in  (B, N_kv, d_in_v) —— value 序列（N_kv 与 K 同长）
        W_Q (h*d_k, d_in_q) / W_K (h*d_k, d_in_k) / W_V (h*d_v, d_in_v)
        W_O (d_model, h*d_v)
        mask (B, h, N_q, N_kv) 或 (B, 1, N_q, N_kv) 或 None
              —— 值为 0 的位置会被屏蔽（softmax 前设成 -inf）

    返回：(B, N_q, d_model)

    三种用法：
        1. 自注意力：Q_in = K_in = V_in = X（编码器）
        2. 掩码自注意力：Q_in = K_in = V_in = X + 因果掩码（解码器）
        3. cross-attention：Q_in 来自解码器，K_in/V_in 来自编码器输出
    """
    B, N_q, _ = Q_in.shape
    N_kv = K_in.shape[1]
    h = num_heads
    d_k = W_Q.shape[0] // h
    d_v = W_V.shape[0] // h

    # 1. 投影：(B, N, d_in) @ (d_in, h*d_k) → (B, N, h*d_k)
    Q = Q_in @ W_Q.T  # (B, N_q, h*d_k)
    K = K_in @ W_K.T  # (B, N_kv, h*d_k)
    V = V_in @ W_V.T  # (B, N_kv, h*d_v)

    # 2. 拆头：(B, N, h*d_k) → (B, N, h, d_k) → (B, h, N, d_k)
    Q = Q.reshape(B, N_q, h, d_k).transpose(0, 2, 1, 3)  # (B, h, N_q, d_k)
    K = K.reshape(B, N_kv, h, d_k).transpose(0, 2, 1, 3)  # (B, h, N_kv, d_k)
    V = V.reshape(B, N_kv, h, d_v).transpose(0, 2, 1, 3)  # (B, h, N_kv, d_v)

    # 3. scaled dot-product attention
    #    scores (B, h, N_q, N_kv)：每个 query 对每个 key 的相似度
    scale = 1.0 / np.sqrt(d_k)
    scores = (Q @ K.transpose(0, 1, 3, 2)) * scale  # (B, h, N_q, N_kv)

    # 4. 掩码（如有）：被屏蔽位置分数设成 -inf，softmax 后权重为 0
    if mask is not None:
        scores = np.where(mask == 0, -np.inf, scores)

    weights = softmax(scores, axis=-1)  # 沿最后一维（key 维）归一化
    out = weights @ V  # (B, h, N_q, d_v)

    # 5. 合并头：(B, h, N_q, d_v) → (B, N_q, h, d_v) → (B, N_q, h*d_v)
    out = out.transpose(0, 2, 1, 3).reshape(B, N_q, h * d_v)

    # 6. 输出投影压回 d_model
    return out @ W_O.T  # (B, N_q, d_model)


def make_causal_mask(seq_len: int) -> np.ndarray:
    """因果掩码：上三角为 0（屏蔽），下三角为 1（可见）。

    解码器生成第 t 个 token 时只能看到前 t 个 token（含自己），
    防止"作弊"——看到未来答案。

    用法：multi_head_attention(..., mask=make_causal_mask(N))
    """
    return np.tril(np.ones((seq_len, seq_len)), k=0).astype(np.float32)


# ============================================================
# 前馈网络 FFN
# ============================================================

def feed_forward(x: np.ndarray, w1: np.ndarray, b1: np.ndarray, w2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """两层全连接 + ReLU（笔记 3.3 节）。

    x   (B, N, d_model)
    w1  (d_ff, d_model)  b1 (d_ff,)
    w2  (d_model, d_ff)  b2 (d_model,)
    返回 (B, N, d_model)

    作用：对每个 token 独立做非线性变换（自注意力看全局，FFN 看单个 token）。
    """
    # 注意 numpy 全连接：x @ w1.T + b1，因为权重按 (out, in) 存
    h = np.maximum(0, x @ w1.T + b1)  # ReLU
    return h @ w2.T + b2


# ============================================================
# 编码器
# ============================================================

def encoder_block(
    x: np.ndarray,
    params: dict,
    num_heads: int,
) -> np.ndarray:
    """一个编码器块（笔记 3.6 节，Post-LN）。

    流程：
        x → 多头自注意力 → (+x) → LN → FFN → (+x') → LN → 输出

    params 字典包含该块所有权重：W_Q/W_K/W_V/W_O（注意力）+ w1/b1/w2/b2（FFN）。
    """
    # 子层 1：多头自注意力 + 残差 + LN
    attn_out = multi_head_attention(
        x, x, x,
        params["W_Q"], params["W_K"], params["W_V"], params["W_O"],
        num_heads=num_heads,
    )
    x = layer_norm(x + attn_out)  # 残差 + LN

    # 子层 2：FFN + 残差 + LN
    ffn_out = feed_forward(x, params["w1"], params["b1"], params["w2"], params["b2"])
    x = layer_norm(x + ffn_out)  # 残差 + LN
    return x


def encoder(
    x: np.ndarray,
    blocks_params: list[dict],
    num_heads: int,
) -> np.ndarray:
    """编码器：N 个 block 堆叠（笔记 3.7 节）。"""
    for params in blocks_params:
        x = encoder_block(x, params, num_heads)
    return x


# ============================================================
# 解码器
# ============================================================

def decoder_block(
    x: np.ndarray,
    enc_out: np.ndarray,
    params: dict,
    num_heads: int,
) -> np.ndarray:
    """一个解码器块（笔记 5.3、5.4、6 节，Post-LN）。

    三个子层：
        1. 掩码多头自注意力 + 残差 + LN
           —— Q/K/V 都来自解码器输入，加因果掩码（生成第 t 个 token 只看前 t 个）
        2. cross-attention + 残差 + LN
           —— Q 来自解码器，K/V 来自编码器输出（enc_out）
        3. FFN + 残差 + LN
    """
    _, N, _ = x.shape

    # 子层 1：掩码自注意力
    causal = make_causal_mask(N)  # (N, N)
    causal = causal[None, None, :, :]  # (1, 1, N, N)，广播到 (B, h, N, N)
    self_attn_out = multi_head_attention(
        x, x, x,
        params["W_Q1"], params["W_K1"], params["W_V1"], params["W_O1"],
        num_heads=num_heads,
        mask=causal,
    )
    x = layer_norm(x + self_attn_out)

    # 子层 2：cross-attention（Q 来自解码器，K/V 来自编码器）
    cross_out = multi_head_attention(
        x, enc_out, enc_out,  # Q_in=x, K_in=enc_out, V_in=enc_out
        params["W_Q2"], params["W_K2"], params["W_V2"], params["W_O2"],
        num_heads=num_heads,
    )
    x = layer_norm(x + cross_out)

    # 子层 3：FFN
    ffn_out = feed_forward(x, params["w1"], params["b1"], params["w2"], params["b2"])
    x = layer_norm(x + ffn_out)
    return x


def decoder(
    x: np.ndarray,
    enc_out: np.ndarray,
    blocks_params: list[dict],
    num_heads: int,
) -> np.ndarray:
    """解码器：N 个 block 堆叠。"""
    for params in blocks_params:
        x = decoder_block(x, enc_out, params, num_heads)
    return x


# ============================================================
# 参数初始化（小随机数，demo 用；真实训练用 Xavier/PyTorch 默认）
# ============================================================

def init_params(d_model: int, num_heads: int, d_ff: int) -> dict:
    """初始化一个编码器 block 的所有参数。"""
    d_k = d_v = d_model // num_heads
    return {
        "W_Q": np.random.randn(num_heads * d_k, d_model) * 0.1,
        "W_K": np.random.randn(num_heads * d_k, d_model) * 0.1,
        "W_V": np.random.randn(num_heads * d_v, d_model) * 0.1,
        "W_O": np.random.randn(d_model, num_heads * d_v) * 0.1,
        "w1": np.random.randn(d_ff, d_model) * 0.1,
        "b1": np.zeros(d_ff),
        "w2": np.random.randn(d_model, d_ff) * 0.1,
        "b2": np.zeros(d_model),
    }


def init_decoder_params(d_model: int, num_heads: int, d_ff: int) -> dict:
    """初始化一个解码器 block 的参数（含两套注意力 + FFN）。"""
    d_k = d_v = d_model // num_heads
    return {
        # 子层 1：掩码自注意力
        "W_Q1": np.random.randn(num_heads * d_k, d_model) * 0.1,
        "W_K1": np.random.randn(num_heads * d_k, d_model) * 0.1,
        "W_V1": np.random.randn(num_heads * d_v, d_model) * 0.1,
        "W_O1": np.random.randn(d_model, num_heads * d_v) * 0.1,
        # 子层 2：cross-attention
        "W_Q2": np.random.randn(num_heads * d_k, d_model) * 0.1,
        "W_K2": np.random.randn(num_heads * d_k, d_model) * 0.1,
        "W_V2": np.random.randn(num_heads * d_v, d_model) * 0.1,
        "W_O2": np.random.randn(d_model, num_heads * d_v) * 0.1,
        # 子层 3：FFN
        "w1": np.random.randn(d_ff, d_model) * 0.1,
        "b1": np.zeros(d_ff),
        "w2": np.random.randn(d_model, d_ff) * 0.1,
        "b2": np.zeros(d_model),
    }


# ============================================================
# 端到端 Transformer
# ============================================================

def transformer(
    src: np.ndarray,    # 源序列 token embedding，(B, N_src, d_model)
    tgt: np.ndarray,    # 目标序列已生成部分（teacher forcing），(B, N_tgt, d_model)
    d_model: int,
    num_heads: int,
    num_layers: int,
    enc_params: list[dict],
    dec_params: list[dict],
    output_proj: np.ndarray,  # (vocab_size, d_model)，最后线性层
) -> np.ndarray:
    """完整 Transformer 前向（训练模式，teacher forcing）。

    返回 logits，形状 (B, N_tgt, vocab_size)。
    """
    B, N_src, _ = src.shape
    _, N_tgt, _ = tgt.shape

    # 1. 加位置编码（笔记 3.7 节，自注意力本身无序）
    src = src + positional_encoding(N_src, d_model)[None, :, :]  # (B, N_src, d_model)
    tgt = tgt + positional_encoding(N_tgt, d_model)[None, :, :]  # (B, N_tgt, d_model)

    # 2. 编码器
    enc_out = encoder(src, enc_params, num_heads)  # (B, N_src, d_model)

    # 3. 解码器
    dec_out = decoder(tgt, enc_out, dec_params, num_heads)  # (B, N_tgt, d_model)

    # 4. 输出投影：把 d_model 维向量映射到 vocab_size 维（logits）
    logits = dec_out @ output_proj.T  # (B, N_tgt, vocab_size)
    return logits


# ============================================================
# 验证：形状检查
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    # 超参数
    d_model = 64
    num_heads = 8
    d_ff = 256
    num_layers = 2  # 堆叠 2 个 block（真实 Transformer 用 6 层）
    B = 2  # batch size
    N_src = 10  # 源序列长度
    N_tgt = 12  # 目标序列长度
    tgt_vocab_size = 800

    # 初始化参数
    enc_params = [init_params(d_model, num_heads, d_ff) for _ in range(num_layers)]
    dec_params = [init_decoder_params(d_model, num_heads, d_ff) for _ in range(num_layers)]
    output_proj = np.random.randn(tgt_vocab_size, d_model) * 0.1

    # 假装是 token embedding（真实场景是查 vocab embedding 表）
    src = np.random.randn(B, N_src, d_model) * 0.1
    tgt = np.random.randn(B, N_tgt, d_model) * 0.1

    # 前向
    logits = transformer(
        src, tgt,
        d_model, num_heads, num_layers,
        enc_params, dec_params, output_proj,
    )

    print("=" * 60)
    print("Transformer 形状验证")
    print("=" * 60)
    print(f"源序列 src:        {src.shape}  (B, N_src, d_model)")
    print(f"目标序列 tgt:      {tgt.shape}  (B, N_tgt, d_model)")
    print(f"编码器输出 enc_out: (B, N_src, d_model) = {(B, N_src, d_model)}")
    print(f"解码器输出 dec_out: (B, N_tgt, d_model) = {(B, N_tgt, d_model)}")
    print(f"最终 logits:       {logits.shape}  (B, N_tgt, vocab_size)")
    print(f"\n[OK] 形状正确，vocab 维度 = {tgt_vocab_size}")

    # 单独验证每个组件
    print("\n" + "=" * 60)
    print("各组件单独验证")
    print("=" * 60)

    # 多头注意力（自注意力模式）
    x = np.random.randn(2, 5, d_model) * 0.1
    p = init_params(d_model, num_heads, d_ff)
    attn_out = multi_head_attention(x, x, x, p["W_Q"], p["W_K"], p["W_V"], p["W_O"], num_heads)
    print(f"多头自注意力:  输入 {x.shape} → 输出 {attn_out.shape}")

    # 掩码注意力
    mask = make_causal_mask(5)
    print(f"\n因果掩码 (5×5): 上三角为 0")
    print(mask)
    masked_out = multi_head_attention(x, x, x, p["W_Q"], p["W_K"], p["W_V"], p["W_O"], num_heads, mask=mask[None, None])
    print(f"掩码注意力:    输入 {x.shape} → 输出 {masked_out.shape}")

    # LN
    print(f"\n层归一化:      输入 {x.shape} → 输出 {layer_norm(x).shape}")
    print(f"  LN 前 mean={np.mean(x[0,0]):.4f} std={np.std(x[0,0]):.4f}")
    print(f"  LN 后 mean={np.mean(layer_norm(x)[0,0]):.4f} std={np.std(layer_norm(x)[0,0]):.4f}")

    # FFN
    ffn_out = feed_forward(x, p["w1"], p["b1"], p["w2"], p["b2"])
    print(f"\n前馈网络:      输入 {x.shape} → 输出 {ffn_out.shape}")

    # 位置编码
    pe = positional_encoding(10, d_model)
    print(f"\n位置编码:      形状 {pe.shape}")
    print(f"  PE[0, :5] = {pe[0, :5]}")
    print(f"  PE[1, :5] = {pe[1, :5]}  ← 同一维度位置变化时值不同")
