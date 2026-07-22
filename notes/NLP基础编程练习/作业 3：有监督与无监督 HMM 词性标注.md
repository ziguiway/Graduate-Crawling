---
title: 作业 3：有监督与无监督 HMM 词性标注
date: 2026-07-21
tags:
  - NLP
  - HMM
  - 词性标注
  - EM
  - Viterbi
  - 作业3
status: 🔄 进行中
parent: "[[NLP基础编程练习]]"
---

# 作业 3：有监督与无监督 HMM 词性标注

> [!summary] 一句话任务
> 用隐马尔可夫模型（HMM）做词性标注：先做有监督版（极大似然 + Viterbi），再做无监督版（EM 训练）。
> 本质：==生成式概率模型 + 动态规划解码；无监督版引入 EM（前向后向算法）==。

## 题目要求

### Part 1：有监督的 HMM 词性标注（共 15 分）

- **模型**：实现一个**二元（一阶）HMM**做词性标注（实现三元模型可适当加分）
- **训练**：在 `train.conll` 上使用**极大似然估计**方法确定模型参数
  - **发射概率**：使用 **加 α 平滑**（或其他平滑方法）估计"词性 → 词"的概率（**3 分**）
  - **转移概率**：**直接估计**词性转移概率（**2 分**）
- **解码**：实现 **Viterbi 算法**，对 `dev.conll` 进行词性标注（**7 分**）
- **评价**：在 `dev.conll` 上评价模型的词性准确率（**3 分**）

$$
\text{Tagging Accuracy} = \frac{\#\text{words with correct tags}}{\#\text{words in total}}
$$

### Part 2：无监督的 HMM 词性标注（共 15 分）

- **模型**：实现一个二元（一阶）HMM 做词性标注
- **训练**：在 `train.conll` 上使用极大似然估计方法确定模型参数
  - **Hard EM**（**4 分**）
  - **Soft EM**（**8 分**）
  - 两个都完成分数可累加
- **词性约束**：先从 `train.conll` 中统计词表，得到**每一个词所有可能的词性**，作为 EM 运行时的约束
- **停止条件**：迭代 **100 次**后停止，**每次迭代后**报出目前数据的 **log-likelihood**
- **评价**：在 `dev.conll` 上评价模型的词性准确率（**3 分**）
- **多初始化**：用 **5 个不同的初始化种子**，训练得到不同的模型，分别汇报准确率

> [!note] EM 讲义
> 老师说会尽快公布一个 pdf 讲义，介绍 EM 相关内容（包括前向后向算法）。等讲义出来再补具体推导。

## 数据与资料

### 数据（与作业 4–11 共用）

| 数据集 | 用途 | 规模 | 下载 |
|---|---|---|---|
| 小数据集 | 入门练手 | 训练 803 句 / 开发 1910 句 | [data.tar.gz](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/6-ngram-language-model/data.tar.gz) |
| 大数据集 | 正式实验 | 训练 16091 句 / 开发 803 句 / 测试 1910 句 | [ctb5-postagged.tar.gz](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/ctb5-postagged.tar.gz) |

> [!tip] 从小数据集开始
> 入门阶段先用小数据集（803 句训练），跑通整个 pipeline；再换大数据集看准确率变化。

### 数据格式（CoNLL）

老师主页明确：每个词占一行，每行的第 **2** 列为当前词语，第 **4** 列为当前词的词性。句子与句子之间以空行间隔。

> [!example] 词性标注示例
> 输入：`严守一 把 手机 关 了`
> 输出：`严守一/NR 把/P 手机/NN 关/VV 了/SP`

### 视频

2022 春 IR 课程，分低画质 / 高画质 / 图片三套：

- 低画质：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1.mp4) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2.mp4) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3.mp4) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4.mp4)
- 高画质：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1-hd.mp4) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2-hd.mp4) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3-hd.mp4) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4-hd.mp4)
- 图片：[第1部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-1.jpg) ｜ [第2部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-2.jpg) ｜ [第3部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-3.jpg) ｜ [第4部分](http://hlt.suda.edu.cn/LA/Ir-2022-Spring/HMM/HMM-part-4.jpg)

### 课件

- [Collins 教授课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/collins-tagging.pdf)
- [李正华老师课件（pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/main.pdf)
- [理解 HMM 的 Viterbi（pptx）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging/HMM-v2.pptx)
- [HMM 模型中极大似然估计的由来（公式推导，pdf）](http://hlt.suda.edu.cn/~zhli/teach/cip-2015-fall/7-hmm-tagging-MLE/main.pdf)

---

## 核心知识点 1：HMM 如何建模词性标注

### 问题形式

词性标注的输入是一句话的词序列：

$$
x = x_1, x_2, \dots, x_n
$$

输出是每个词对应的词性序列：

$$
y = y_1, y_2, \dots, y_n
$$

例如：

```text
输入：戴相龙 说 中国 经济 发展
输出：NR   VV NR   NN   NN
```

### HMM 的两个序列

HMM 里有两个序列：

| 序列 | 在词性标注里的含义 | 是否可见 |
|---|---|---|
| 观测序列 $x$ | 词序列，例如 `中国`、`经济` | 可见 |
| 隐状态序列 $y$ | 词性序列，例如 `NR`、`NN`、`VV` | 预测时不可见 |

词性之所以叫“隐状态”，是因为预测时只看到词，不知道词性；训练时因为 `train.conll` 里有人工标注，所以可以直接统计。

### 一阶 HMM 的核心假设

一阶 HMM 有两个重要假设：

- **转移假设**：当前词性只依赖上一个词性。
- **发射假设**：当前词只依赖当前词性。

用公式写就是：

$$
P(x,y)=\prod_{i=1}^{n}P(x_i\mid y_i)P(y_i\mid y_{i-1})
$$

其中：

- $P(y_i\mid y_{i-1})$：转移概率，表示“上一个词性到当前词性”的概率。
- $P(x_i\mid y_i)$：发射概率，表示“当前词性生成当前词”的概率。

> [!note] 当前实现暂时不考虑 STOP
> 老师板书里完整写法会把句末 `STOP` 也作为一个特殊状态。本阶段先不补 `P(STOP | tag)`，Viterbi 终止时直接从最后一个位置选择分数最高的词性。

---

## 核心知识点 2：有监督 HMM 的参数估计

有监督训练的意思是：训练数据里已经给好了每个词的正确词性，所以可以直接数频率，用极大似然估计得到概率。

### 句首概率

句首概率表示某个词性出现在句子开头的概率：

$$
P(t\mid START)=\frac{count(t\text{ appears at sentence start})}{count(\text{sentences})}
$$

对应代码：

```python
start_count[tags[0]] += 1
```

```python
def get_start_prob(tag: str, model: dict) -> float:
    return model["start_count"][tag] / model["sentence_count"]
```

### 转移概率

转移概率表示上一个词性是 `prev_tag` 时，当前词性是 `tag` 的概率：

$$
P(tag\mid prev\_tag)=\frac{count(prev\_tag, tag)}{count(prev\_tag)}
$$

对应代码：

```python
transition_count[tags[i-1]][tag] += 1
```

```python
def get_transition_prob(prev_tag: str, tag: str, model: dict) -> float:
    if model["tag_counts"][prev_tag] == 0:
        return 0.0

    return model["transition_count"][prev_tag][tag] / model["tag_counts"][prev_tag]
```

当前版本没有给转移概率做平滑。如果某个转移从没见过，它的概率就是 `0`，在 Viterbi 中会被 `safe_log(0)` 变成 `-inf`。

### 发射概率

发射概率表示某个词性 `tag` 生成某个词 `word` 的概率：

$$
P(word\mid tag)=\frac{count(tag, word)}{count(tag)}
$$

为了处理训练集中没见过的词，当前代码使用加 $\alpha$ 平滑：

$$
P(word\mid tag)=\frac{count(tag, word)+\alpha}{count(tag)+\alpha |V|}
$$

其中 $|V|$ 是词表大小。

对应代码：

```python
def get_emission_prob(tag: str, word: str, model: dict) -> float:
    alpha = model["alpha"]
    vocab_size = len(model["vocab"])

    return (
        model["emission_count"][tag][word] + alpha
    ) / (
        model["tag_counts"][tag] + alpha * vocab_size
    )
```

> [!tip] 三个概率的直觉
> - 句首概率：一句话通常从什么词性开始？
> - 转移概率：某个词性后面通常接什么词性？
> - 发射概率：某个词性通常会生成哪些词？

---

## 核心知识点 3：Viterbi 动态规划解码

### 为什么需要 Viterbi

如果一句话有 $n$ 个词，词性集合大小是 $|T|$，暴力枚举所有词性序列需要：

$$
|T|^n
$$

这会爆炸。例如 10 个词、31 个词性，就已经是 $31^{10}$ 种组合。

Viterbi 的想法是：不用保存所有路径。对于每个位置 `i`、每个当前词性 `tag`，只保存到这里为止分数最高的那条路径。

### `dp` 表

```python
dp[i][tag]
```

表示：

> 处理到第 `i` 个词，并且第 `i` 个词固定标成 `tag` 时，当前最好的 log 概率分数。

初始化：

$$
dp[0][t]=\log P(t\mid START)+\log P(x_0\mid t)
$$

递推：

$$
dp[i][t]=\max_p\left(dp[i-1][p]+\log P(t\mid p)+\log P(x_i\mid t)\right)
$$

> [!tip] 土话版理解
> `C(i, t)` 就是在问：==第 `i` 个词如果非要标成 `t`，那我从哪个上一个词性 `p` 接过来最划算？==
>
> 比如算 `C(3, NR)`：
> “中国”如果标成 `NR`，那么前一个词“爱”最好标成什么，才能让整句话到这里为止概率最大？
>
> 比如算 `C(3, NN)`：
> “中国”如果标成 `NN`，那么“爱”最好标成什么？
>
> Viterbi 就是每个当前位置、每个候选词性都问一遍这个问题，然后只留下最划算的前驱。

### `path` 表

```python
path[i][tag]
```

表示：

> 为了得到 `dp[i][tag]` 这个最高分，第 `i-1` 个词应该是什么词性。

也就是说，`dp` 负责保存分数，`path` 负责保存路线。

### 递推代码的关键

```python
best_score = -math.inf
best_prev_tag = None

for prev_tag in all_tags:
    score = (
        dp[i - 1][prev_tag]
        + safe_log(get_transition_prob(prev_tag, tag, model))
        + safe_log(get_emission_prob(tag, words[i], model))
    )

    if score > best_score:
        best_score = score
        best_prev_tag = prev_tag

dp[i][tag] = best_score
path[i][tag] = best_prev_tag
```

这里最关键的一点：`path[i][tag]` 必须保存“让 `score` 最大的那个 `prev_tag`”，不能保存循环结束时最后一个 `prev_tag`。

### 回溯

当前不考虑 `STOP`，所以先找最后一个位置分数最高的词性：

```python
best_last_tag = max(all_tags, key=lambda tag: dp[-1][tag])
```

然后从右往左查 `path`：

```python
best_tags = [best_last_tag]

for i in range(len(words) - 1, 0, -1):
    best_tags.append(path[i][best_tags[-1]])

best_tags.reverse()
```

> [!summary] Viterbi 一句话
> 从左到右填 `dp` 和 `path`：`dp` 记最好分数，`path` 记最好前驱；最后从句尾最佳词性开始一路倒着找回来。

---

## 核心知识点 4：评价指标 Tagging Accuracy

词性标注评价比较直接：逐词比较预测词性和标准词性是否一致。

$$
\text{Accuracy}=\frac{\#\text{预测正确的词}}{\#\text{所有词}}
$$

例如：

```text
gold: NR VV NR NN
pred: NR VV NN NN
```

共有 4 个词，其中 3 个预测正确：

$$
Accuracy=\frac{3}{4}=0.75
$$

后续实现 `evaluate` 时，可以遍历 `dev.conll` 的每个句子：

```python
correct = 0
total = 0

for sentence in dev_data["sentences"]:
    words = sentence["words"]
    gold_tags = sentence["tags"]
    pred_tags = viterbi(words, model)

    for gold_tag, pred_tag in zip(gold_tags, pred_tags):
        if gold_tag == pred_tag:
            correct += 1
        total += 1

accuracy = correct / total
```

评价函数是有监督 HMM Part 1 的最后一步：训练参数、Viterbi 解码之后，用它在 `dev.conll` 上报准确率。

---

## 学习问答

### Viterbi 如何从整句所有词性组合中找到概率最大的序列？

#### 问题描述

看到老师板书中的

$$
C(i,t)=\max_{p}\left[C(i-1,p)P(t\mid p)\right]P(x_i\mid t)
$$

时，不清楚 `C(i, t)` 在代码里应该存什么、为什么只保留一个最大值，以及算到最后一个词后怎样恢复完整的词性序列。当前 `supervised_hmm.py` 已经实现了句首概率、转移概率、发射概率，并补上了不带 `STOP` 的 Viterbi 解码。

#### 回答

`dp[i][tag]` 表示：处理完第 `0..i` 个词，并且第 `i` 个词固定标为 `tag` 时，所有可能前缀路径中的==最大对数概率==。不同前缀只要到达相同的 `(i, tag)`，后续面对的选择就完全相同，所以较差的前缀永远不可能反超，只需留下最好的一个。

为避免很多小概率连续相乘后浮点下溢，实际代码使用对数：乘法变加法，`max` 的位置不变。

**1. 初始化第一个词**

$$
dp[0][t]=\log P(t\mid START)+\log P(x_0\mid t)
$$

**2. 从左到右递推**

$$
dp[i][t]=\max_p\left(dp[i-1][p]+\log P(t\mid p)\right)+\log P(x_i\mid t)
$$

同时用 `backpointer[i][t]` 保存取得最大值的前驱词性 `p`。

**3. 暂时不考虑 STOP，直接选出最后一个词性**

$$
y_{n-1}^*=\arg\max_t dp[n-1][t]
$$

**4. 沿 backpointer 从右向左回溯，再把结果反转**。

> [!note] 关于 STOP
> 老师板书里的 `STOP` 是完整 HMM 解码的一部分，但当前可以先不补它。先把 Viterbi 的动态规划和回溯跑通，后面再把 `P(STOP | tag)` 接到终止步骤里。

当前实现中，不带 STOP 的 Viterbi 主体写成：

```python
import math


def safe_log(prob: float) -> float:
    """把零概率变成负无穷，使不可能的路径不会胜出。"""
    return math.log(prob) if prob > 0.0 else -math.inf


def viterbi(words: list[str], model: dict) -> list[str]:
    """Find the best POS tag sequence for one sentence with Viterbi."""
    if not words:
        return []

    all_tags = sorted(model["tag_set"])

    dp = [{} for _ in words]
    path = [{} for _ in words]

    # ★ 1. 初始化：第 0 个词
    for tag in all_tags:
        dp[0][tag] = (
            safe_log(get_start_prob(tag, model))
            + safe_log(get_emission_prob(tag, words[0], model))
        )
        path[0][tag] = None

    # ★ 2. 递推：从第 1 个词开始
    for i in range(1, len(words)):
        for tag in all_tags:
            best_score = -math.inf
            best_prev_tag = None

            for prev_tag in all_tags:
                score = (
                    dp[i - 1][prev_tag]
                    + safe_log(get_transition_prob(prev_tag, tag, model))
                    + safe_log(get_emission_prob(tag, words[i], model))
                )

                if score > best_score:
                    best_score = score
                    best_prev_tag = prev_tag

            dp[i][tag] = best_score
            path[i][tag] = best_prev_tag  # ★ 保存最佳前驱

    # ★ 3. 终止：暂时不考虑 STOP，直接选最后位置分数最高的 tag
    best_last_tag = max(all_tags, key=lambda tag: dp[-1][tag])

    # ★ 4. 回溯：从最佳句末词性一路向左找前驱
    best_tags = [best_last_tag]
    for i in range(len(words) - 1, 0, -1):
        best_tags.append(path[i][best_tags[-1]])

    best_tags.reverse()
    return best_tags
```

这里最容易写错的是 `path[i][tag]`。它不能随便记录循环结束时的最后一个 `prev_tag`，必须记录==让当前 `score` 最大的那个 `prev_tag`==，所以代码里同时维护了 `best_score` 和 `best_prev_tag`。

这个实现的时间复杂度为 $O(n|T|^2)$：句长为 $n$，每个位置枚举当前词性和前一个词性；空间复杂度为 $O(n|T|)$，用于保存分数和回溯指针。

#### 一句话记忆

> Viterbi 就是“每个位置、每个当前词性只保留分数最高的前驱”，向前算最优分数，向后沿指针恢复整条最优词性序列。

### `dp` 和 `path` 两张表分别在记录什么？

#### 问题描述

实现代码中有两行：

```python
dp = [{} for _ in words]
path = [{} for _ in words]
```

刚开始不清楚为什么要为每个词都创建一个字典，也不清楚 `dp[i][tag]` 和 `path[i][tag]` 分别保存什么。

#### 回答

如果一句话有 3 个词，`[{} for _ in words]` 会创建 3 个空字典：

```python
[
    {},
    {},
    {},
]
```

后续每个位置的字典里，以词性作为 key：

```python
dp[1]["VV"]
path[1]["VV"]
```

其中：

- `dp[i][tag]`：第 `i` 个词标成 `tag` 时，到当前位置为止的最大 log 概率。
- `path[i][tag]`：为了得到 `dp[i][tag]` 这个最大分数，第 `i-1` 个词应该是什么词性。

例如：

```python
dp[2]["NN"] = -12.8
path[2]["NN"] = "VV"
```

可以读成：

> 第 2 个词标成 `NN` 时，目前最好的路径分数是 `-12.8`；这条最好路径是从上一个词的 `VV` 接过来的。

最后回溯时，就是从最后一个最佳词性开始，不断查：

```python
path[i][current_tag]
```

把每一步的最佳前驱找回来。

> [!warning] Python 小坑
> 这里要写 `dp = [{} for _ in words]`，不要写 `dp = [{}] * len(words)`。后者会让所有位置共享同一个字典，修改一个位置时，其他位置也会一起变。

#### 一句话记忆

> `dp` 记“分数是多少”，`path` 记“这个最好分数从哪来”。

---

## 待补充

- [x] HMM 的定义（状态空间、观测空间、转移概率、发射概率、初始概率）
- [x] 极大似然估计的公式推导（参考老师 MLE pdf）
- [x] 加 α 平滑的具体公式
- [x] Viterbi 算法的递推式和回溯
- [ ] Hard EM 和 Soft EM 的区别（前向后向算法）
- [x] 我的实现代码
- [ ] 在 dev.conll 上的准确率结果
- [ ] 5 个种子的无监督实验结果对比

---

## 日志

> [!tip] 日志约定
> 记录每天的进展和困难，呼应李老师"发日志讲进展和困难"的要求。

- 2026-07-22：完成有监督 HMM 的不带 `STOP` 版本 Viterbi 解码。关键点是使用 `dp[i][tag]` 保存最大 log 概率，用 `path[i][tag]` 保存最佳前驱词性；回溯时从最后一个位置分数最高的词性开始往前找。

---

## 我的理解

Viterbi 不是把所有词性组合都列出来，而是每走到一个位置，只保留“到这个位置且当前词性固定时”的最好前缀。`dp` 像成绩表，`path` 像路线记录：前者告诉我当前最优分数，后者告诉我这条最优路线是怎么走来的。先不补 `STOP` 可以降低实现难度，等整体 pipeline 跑通后，再把句末转移概率接到终止步骤里。

我觉得 HMM 的本质就是：先通过训练集得到普遍的概率分布，再用这些经验推测新句子的词性。它不仅看“这个词平时最像什么词性”，还会看“前后词性这样接起来顺不顺”。最后由 Viterbi 找出整句话最顺、概率最大的那条词性路线。

> 一句话记忆：先看训练集里大家平时怎么标，再照着这些经验，猜出新句子最顺的一条词性路线。
