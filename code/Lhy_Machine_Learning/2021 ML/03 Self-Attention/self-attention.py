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
