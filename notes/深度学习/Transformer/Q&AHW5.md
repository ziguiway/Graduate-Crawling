# Q&A - HW5：Transformer（LN vs BN 等）

李宏毅 HW5 / Transformer 学习过程中遇到的问题和解答。
关联笔记：[[Transformer]]、[[自注意力机制]]、[[Q&A/HW4|HW4 Self-Attention]]。

---

## Q1：编码器里的 LayerNorm 和 BatchNorm 到底啥区别？

Transformer 编码器每个子层都用 `LayerNorm(x + Sublayer(x))`，而不是 BatchNorm。要理解为什么，先得把两者掰开。

### 先理解为什么要 Normalization

神经网络训练有个老大难问题：**内部协变量偏移**（Internal Covariate Shift）。

每一层接收上一层的输出作为输入。随着训练进行，每层参数都在变，导致这一层输出的**分布**一直在漂移。下一层就得不停追赶这个移动的目标，训练慢且不稳。

Normalization 的核心思想：不管前面怎么变，在每层输出之前强行把数据"拉回"标准分布（均值 0、方差 1），再用两个可学习参数 γ 和 β 做缩放和偏移，让网络自己决定要不要用这个标准化分布：

$$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

公式都一样，区别只在 **μ 和 σ² 是沿着哪个维度算出来的**。

### BN（Batch Normalization）：按"批次"归一化

归一化维度：对**同一个特征**，跨 batch 内所有样本算均值方差。

想象一个 batch 有 32 个样本，每个样本是 784 维向量。BN 对"第 1 维"统计这 32 个样本的均值方差，对"第 2 维"也统计 32 个……一共 784 组 (μ, σ²)。

> **打比方**：一个班级（batch）里，对"数学成绩"这一科统计全班 32 人的平均分和标准差，用这个标准化每个人的数学成绩。语文、英语各科分别统计。

优点：
- 梯度更平滑，训练更快，可用更大学习率
- 轻微正则化效果（每个样本的归一化统计量依赖其他样本，带点随机性）

致命缺点：
- **依赖 batch size**。batch 太小（比如 2）统计量就不准，效果崩
- **训练/推理不一致**：推理往往单样本过，没有 batch 可统计。训练时要算"全局移动平均"的 μ、σ² 存起来，推理时用固定值（这就是 `model.eval()` 干的事之一）
- **不适合 RNN/Transformer**：序列长度可变，batch 维度统计量不稳定

### LN（Layer Normalization）：按"层"归一化

归一化维度：对**同一个样本**，跨它所有特征算均值方差，与 batch 无关。

32 个样本每个 784 维 —— LN 给每个样本单独算一组 (μ, σ²)，一共 32 组。每个样本自己归一化自己。

> **打比方**：对小明这个人，统计他数学、语文、英语、物理……所有科目的平均分和标准差，用这个标准化他各科成绩。对小红单独算一套，每个学生独立。

优点：
- **与 batch 无关**，batch=1 也正常工作
- **训练推理行为一致**，不用维护全局统计量
- **天然适合 RNN/Transformer**：序列每个时间步、每个 token 都能独立归一化

缺点：
- 没有 BN 那种"跨样本"的正则化效果
- 在 CNN 上效果通常不如 BN（卷积特征图的空间结构，LN 统计所有 channel 会破坏一些信息）

### 一张表对比

| 维度            | Batch Norm                | Layer Norm               |
| --------------- | ------------------------- | ------------------------- |
| 统计方向          | 跨 batch，同特征              | 跨特征，同样本                 |
| 依赖 batch size? | 强依赖                      | 不依赖                      |
| 训练/推理行为       | 不一致（需移动平均）              | 一致                       |
| 典型场景           | CNN（图像 ResNet 等）         | RNN / Transformer / NLP   |
| batch=1 能用吗   | 崩                        | 正常                       |

### 一句话记忆

> **B**N = **B**atch 维度归一化 → 关心"一群样本的同一特征"
> **L**N = **L**ayer（特征）维度归一化 → 关心"一个样本的所有特征"

---

## Q2：Transformer 为什么用 LN 不用 BN？

这是 Q1 的核心延伸，单独拎出来。

Transformer 输入是 `(batch, seq_len, d_model)` 三维张量。

### 原因 1：序列长度可变

不同句子的 `seq_len` 不一样。BN 如果在 batch 维度统计，遇到变长序列统计量就不稳。LN 在特征维度统计，和 `seq_len` 无关。

### 原因 2：每个 token 独立归一化

LN 给序列中每个位置（token）单独算 (μ, σ²)，不受其他 token 影响，也不受 batch 内其他句子影响 —— 这对**自回归生成**很关键：推理时一个 token 一个 token 生成，batch=1 也能正常工作。

### 原因 3：训练推理一致

不需要 `eval()` 切换统计量，训练和推理走的是同一条计算路径。实现简单，行为可预测。

### 结论

Transformer 每个 SubLayer 都是 `LayerNorm(x + Sublayer(x))` 这种残差 + LN 的结构（Post-LN；现代变体也有 Pre-LN，把 LN 放在子层之前），不用 BN。

> 一句话：**Transformer 的数据是变长序列 + 逐 token 生成，BN 的 batch 统计跟不上这种变化，LN 不依赖 batch 才稳。**

---

## 未实现（后续再展开）

- Pre-LN vs Post-LN 的差异与训练稳定性
- RMSNorm（LLaMA 等用 LN 的简化变体，去掉了均值减法）
- `PyTorch nn.LayerNorm` 的 `normalized_shape` 参数怎么对应到 Transformer 的 `d_model`
