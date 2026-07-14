# Q&A - HW4：Self-Attention（自注意力机制）

李宏毅 HW4 / 自注意力实现过程中遇到的问题和解答。
关联笔记：[[自注意力机制]]、[[Q&A/HW5|HW5 Transformer]]。

---

## Q1：自注意力的矩阵实现，原代码哪里有 bug？

实现的是单头自注意力（不做多头），代码在 `code/Lhy_Machine_Learning/2021 ML/03 Self-Attention/self-attention.py`。

原代码踩的坑：

| 问题 | 原代码 | 修正后 |
|------|--------|--------|
| 拼写 | `attention_scroce` | `attention_score` |
| softmax 数值稳定性 | `np.exp(x)` 直接算，大数会溢出 | 先减最大值 `x - x_max` 再 exp |
| softmax 作用轴 | `np.sum(exp_x)` 把整个矩阵拍扁成一维归一化 | `softmax(A, axis=0)`，按列归一化 |
| 加权和方向 | `attention_score @ V` 形状对不上 | `(attention_weight @ V.T).T` |

关键约定（与 [[自注意力机制]] 第 4 节一致）：

- 输入 X 是**列向量堆叠**：`X = [a¹ a² ... aᴺ]`，形状 `(d_in, N)`，每列一个输入向量
- `A = Kᵀ @ Q` 得到 `(N, N)`，**列 j 是 qʲ 对所有 k 的分数**
- softmax **按列**做，每列和为 1
- 输出 O 形状 `(d_v, N)`，与输入等长

---

## Q2：softmax 为什么按列（axis=0）做？原代码到底错在哪？

### 数据摆放：列 = 一个向量

笔记第 4 节约定 `I = [a¹ a² a³ a⁴]`，4 个输入向量拼成矩阵的 4 列：

```
        a¹  a²  a³  a⁴
I  =  [  .   .   .   .   ]   ← 第 1 行（特征维度）
       [  .   .   .   .   ]
       [  .   .   .   .   ]
            ↑
         第 j 列 = 第 j 个输入向量 aʲ
```

所以 X 形状 `(d_in, N)`：行是特征维度，列是序列位置。

### 注意力分数矩阵 A 的含义

`A = Kᵀ @ Q`，形状 `(N, N)`：

```
        q¹   q²   q³   q⁴    ← 列对应 query
A  =  [ α₁₁  α₁₂  α₁₃  α₁₄ ]   ← 行 1: k¹ 与各 q 的分数
       [ α₂₁  α₂₂  α₂₃  α₂₄ ]   ← 行 2: k² 与各 q 的分数
       [ α₃₁  α₃₂  α₃₃  α₃₄ ]
       [ α₄₁  α₄₂  α₄₃  α₄₄ ]
```

**第 j 列 = qʲ 去问所有 k¹..kᴺ 得到的分数**。

### softmax 应该归一化谁？

笔记说："对 A 每一列做 softmax，使每列和为 1"。

算 bʲ 时要的是 qʲ 对所有 key 的分布：

$$
b^j = \sum_i \alpha'_{i,j}\, v^i
$$

`α'_{1,j}, α'_{2,j}, ..., α'_{N,j}` 是 qʲ 对 N 个 v 的权重，这 N 个数加起来必须等于 1（一个概率分布），加权和才有意义。

这 N 个数正是 A 的**第 j 列**。所以对**第 j 列**做 softmax。推广到所有 j → 每一列各算各的 → `axis=0`。

### 原代码错在哪

```python
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)   # ← np.sum 无 axis 参数
```

`np.sum(二维数组)` 不给 axis 时，会把整个 `(N,N)` 矩阵**拍扁成一维**，求所有元素的总和。于是除出来的是：每个元素 / 整个矩阵 N×N 个元素之和。

结果是**整个矩阵所有元素加起来等于 1**，而不是每一列加起来等于 1。这相当于"N 个 query 共享一个全局概率分布"，语义完全错位——算 b² 时拿到的不是 q² 对 N 个 key 的分布，而是一堆被全局稀释过的数。

### 三种 axis 的对照

| 写法 | 归一化范围 | 每列和 | 语义 |
|------|-----------|--------|------|
| `np.sum(exp_x)`（无 axis） | 整个矩阵 | 远小于 1 | ❌ N 个 query 抢一个总概率 |
| `np.sum(exp_x, axis=0)` | 每列内部 | = 1 | ✅ 每个 query 独立有自己的分布 |
| `np.sum(exp_x, axis=1)` | 每行内部 | = 1 | ✅ 但语义反了：每个 key 对所有 query 的分布 |

`axis=1` 不是错，是**另一种语义**（"每个 key 被多少 query 关注"做归一化），但它对不上笔记这套"bʲ 由 qʲ 主导"的 Q-主视角约定。按笔记必须 `axis=0`。

### 一句话记忆

> A 的**列**对应一个 query 对所有 key 的分数；要算 bʲ 必须把**第 j 列**归一化成分布 → 所以 softmax 沿 `axis=0` 做每列内部归一化。

---

## Q3：numpy 的 axis=0 到底是行还是列？（反直觉）

### 关键认知翻转

numpy 的 axis 指的是"**沿哪个方向遍历**"，不是"对哪一维做"。

- `axis=0`：沿着**行方向**（第 0 轴，竖着往下走）遍历 → 每次取一**列**的元素聚在一起 → "对每列求和"
- `axis=1`：沿着**列方向**（第 1 轴，横着往右走）遍历 → 每次取一**行**的元素聚在一起 → "对每行求和"

### 验证小例子

```python
import numpy as np
a = np.array([[1, 2],
              [3, 4]])
#       列0  列1
# 行0 [ 1   2 ]
# 行1 [ 3   4 ]

np.sum(a, axis=0)   # → array([4, 6])   跨行求和 = 每列的和（列0: 1+3=4, 列1: 2+4=6）
np.sum(a, axis=1)   # → array([3, 7])   跨列求和 = 每行的和（行0: 1+2=3, 行1: 3+4=7）
```

### 记法

> **axis=k 的 sum 会"吃掉"第 k 轴**，结果剩下的就是其他轴。
> - `axis=0`：吃掉行轴 → 只剩列 → 每列一个值 = "每列求和"
> - `axis=1`：吃掉列轴 → 只剩行 → 每行一个值 = "每行求和"

### 口诀

> **axis=0 竖着走 → 每列一个值
> axis=1 横着走 → 每行一个值**

### 回到注意力

A 形状 `(N, N)`，列 j = qʲ 对所有 key 的分数。要对每列归一化 → `axis=0`。✅

---

## 附：最终实现

```python
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax，沿指定轴归一化使该轴和为 1。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def self_attention(X, W_Q, W_K, W_V):
    """单头自注意力（笔记 6.2 节矩阵乘法实现）。

    X: (d_in, N)  列向量堆叠 I = [a¹ ... aᴺ]
    W_Q/W_K: (d_k, d_in)，W_V: (d_v, d_in)
    返回 O: (d_v, N)，每列是 bʲ
    """
    Q = W_Q @ X
    K = W_K @ X
    V = W_V @ X

    attention_score = K.T @ Q              # (N, N)，列 j = qʲ 对所有 k 的分数
    attention_weight = softmax(attention_score, axis=0)  # 每列归一化
    output = (attention_weight @ V.T).T   # (d_v, N)
    return output
```

运行结果：输出形状 `(d_v, N)` 与输入等长，每列注意力权重和为 1。

---

## 未实现（后续 HW5 再展开）

- 多头注意力（[[自注意力机制]] 第 7 节）
- 位置编码（第 8 节）
- 缩放因子 `1/√d_k`（笔记未提，标准 Transformer 会加；本实现严格按笔记故省略，若要对齐 PyTorch `F.scaled_dot_product_attention` 再补）
