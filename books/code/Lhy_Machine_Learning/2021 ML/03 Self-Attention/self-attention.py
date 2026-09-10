import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax，沿指定轴归一化使该轴和为 1。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
) -> np.ndarray:
    """单头自注意力（按笔记 6.2 节矩阵乘法实现）。

    输入约定（列向量堆叠，与笔记一致）：
        X   形状 (d_in, N)   —— I = [a¹ a² ... aᴺ]，每列是一个输入向量
        W_Q 形状 (d_k, d_in) —— 把 a 映射成 q
        W_K 形状 (d_k, d_in) —— 把 a 映射成 k
        W_V 形状 (d_v, d_in) —— 把 a 映射成 v

    计算流程（对应笔记第 4 节）：
        Q = W_Q @ X        (d_k, N)   查询矩阵
        K = W_K @ X        (d_k, N)   键矩阵
        V = W_V @ X        (d_v, N)   值矩阵
        A = Kᵀ @ Q         (N, N)     注意力分数（q 与每个 k 的内积）
        A' = softmax(A, 按列)         每列对应一个 query，归一化后和为 1
        O = A' @ Vᵀ        (d_v, N)   每列即 b¹..bᴺ

    返回 O，形状 (d_v, N)，与输入等长（笔记第 6 节易错点）。
    """
    Q = W_Q @ X
    K = W_K @ X
    V = W_V @ X

    attention_score = K.T @ Q  # (N, N)，列 j 是 qʲ 对所有 k 的分数

    # 笔记第 4 节："对 A 每一列做 softmax，使每列和为 1"
    attention_weight = softmax(attention_score, axis=0)

    # 加权和：O 的每一列 bʲ = Σ_i α'_{i,j} vⁱ
    # attention_weight (N,N) @ V (d_v,N).T → (N,N) @ (N,d_v) = (N,d_v)，转置回 (d_v,N)
    output = (attention_weight @ V.T).T

    return output


def multi_head_attention(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    W_O: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """多头注意力（笔记 6.4 节）。

    思路：把同一组输入映射成 h 份 Q/K/V，每个头各算各的注意力，
    最后拼接后过一层输出投影 W_O。

    维度约定（沿用列向量堆叠）：
        X   (d_in, N)        输入序列，N 个列向量
        W_Q (h*d_k, d_in)    一次投影出所有头的 query（一个大矩阵代替 h 个小矩阵）
        W_K (h*d_k, d_in)    同上，对应 key
        W_V (h*d_v, d_in)    同上，对应 value
        W_O (d_model, h*d_v) 输出投影，把拼接后的 h*d_v 压回模型维度
        num_heads = h        头数

    返回 O (d_model, N)。
    """
    _, N = X.shape
    h = num_heads
    # 从 W_Q 的行数推出每个头的维度
    d_k = W_Q.shape[0] // h
    d_v = W_V.shape[0] // h

    # 1. 一次性算出所有头的 Q/K/V
    Q_all  = W_Q @ X  # (h*d_k, N)
    K_all = W_K @ X  # (h*d_k, N)
    V_all = W_V @ X  # (h*d_v, N)

    # 2. 按 head 拆开 → (h, d_k, N) / (h, d_v, N)
    Q_heads = Q_all.reshape(h, d_k, N)
    K_heads = K_all.reshape(h, d_k, N)
    V_heads = V_all.reshape(h, d_v, N)

    # 3. 每个头各自做 scaled dot-product attention
    scale = 1.0 / np.sqrt(d_k)  # 缩放因子，防止内积过大导致 softmax 饱和
    outputs = []  # 收集每个头的输出 (d_v, N)
    for i in range(h):
        Q_i = Q_heads[i]  # (d_k, N)
        K_i = K_heads[i]  # (d_k, N)
        V_i = V_heads[i]  # (d_v, N)

        # 注意力分数 (N, N)，列 j 是 qʲ 对所有 k 的内积
        scores = (K_i.T @ Q_i) * scale
        weights = softmax(scores, axis=0)  # 按列归一化
        out_i = (weights @ V_i.T).T  # (d_v, N)
        outputs.append(out_i)

    # 4. 按 feature 维拼接 → (h*d_v, N)
    concat = np.concatenate(outputs, axis=0)  # (h*d_v, N)

    # 5. 输出投影压回模型维度
    O = W_O @ concat  # (d_model, N)
    return O


if __name__ == "__main__":
    np.random.seed(0)
    d_in, d_k, d_v, N = 4, 4, 4, 4
    X = np.random.randn(d_in, N)
    W_Q = np.random.randn(d_k, d_in)
    W_K = np.random.randn(d_k, d_in)
    W_V = np.random.randn(d_v, d_in)

    O = self_attention(X, W_Q, W_K, W_V)
    print("输入 X 形状:", X.shape)
    print("输出 O 形状:", O.shape)
    weight = softmax((W_K @ X).T @ (W_Q @ X), axis=0)
    print("每列注意力权重和（应为 1）:", weight.sum(axis=0))

    # --- 多头注意力演示 ---
    h = 2  # 头数
    d_model = 4  # 模型维度（输出维度）
    W_Q_mh = np.random.randn(h * d_k, d_in)
    W_K_mh = np.random.randn(h * d_k, d_in)
    W_V_mh = np.random.randn(h * d_v, d_in)
    W_O = np.random.randn(d_model, h * d_v)

    O_mh = multi_head_attention(X, W_Q_mh, W_K_mh, W_V_mh, W_O, num_heads=h)
    print("\n[多头] 输入 X 形状:", X.shape)
    print("[多头] 输出 O 形状:", O_mh.shape, "（应为 (d_model, N) =", (d_model, N), ")")
