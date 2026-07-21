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

## 学习问答

### Viterbi 如何从整句所有词性组合中找到概率最大的序列？

#### 问题描述

看到老师板书中的

$$
C(i,t)=\max_{p}\left[C(i-1,p)P(t\mid p)\right]P(x_i\mid t)
$$

时，不清楚 `C(i, t)` 在代码里应该存什么、为什么只保留一个最大值，以及算到最后一个词后怎样恢复完整的词性序列。当前 `supervised_hmm.py` 已经实现了句首概率、转移概率和发射概率，但还缺少 Viterbi 的动态规划表和回溯表。

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

不带 STOP 的 Viterbi 主体可以写成：

```python
import math


def safe_log(prob: float) -> float:
    """把零概率变成负无穷，使不可能的路径不会胜出。"""
    return math.log(prob) if prob > 0.0 else -math.inf


def viterbi(words: list[str], model: dict) -> list[str]:
    """Return the highest-probability POS tag sequence."""
    if not words:
        return []

    tags = sorted(model["tag_set"])
    dp = [{} for _ in words]
    backpointer = [{} for _ in words]

    # ★ 1. 初始化：START -> tag，并生成第一个词
    for tag in tags:
        dp[0][tag] = (
            safe_log(get_start_prob(tag, model))
            + safe_log(get_emission_prob(tag, words[0], model))
        )
        backpointer[0][tag] = None

    # ★ 2. 递推：对当前位置的每个 tag，枚举所有 prev_tag
    for i in range(1, len(words)):
        for tag in tags:
            best_prev_tag = max(
                tags,
                key=lambda prev_tag: (
                    dp[i - 1][prev_tag]
                    + safe_log(get_transition_prob(prev_tag, tag, model))
                ),
            )
            dp[i][tag] = (
                dp[i - 1][best_prev_tag]
                + safe_log(get_transition_prob(best_prev_tag, tag, model))
                + safe_log(get_emission_prob(tag, words[i], model))
            )
            backpointer[i][tag] = best_prev_tag  # ★ 保存最佳前驱

    # ★ 3. 终止：暂时不考虑 STOP，直接选最后位置分数最高的 tag
    best_last_tag = max(tags, key=lambda tag: dp[-1][tag])

    # ★ 4. 回溯：从最佳句末词性一路向左找前驱
    best_tags = [best_last_tag]
    for i in range(len(words) - 1, 0, -1):
        best_tags.append(backpointer[i][best_tags[-1]])

    best_tags.reverse()
    return best_tags
```

这个实现的时间复杂度为 $O(n|T|^2)$：句长为 $n$，每个位置枚举当前词性和前一个词性；空间复杂度为 $O(n|T|)$，用于保存分数和回溯指针。

#### 一句话记忆

> Viterbi 就是“每个位置、每个当前词性只保留分数最高的前驱”，向前算最优分数，向后沿指针恢复整条最优词性序列。

---

## 待补充

- [ ] HMM 的定义（状态空间、观测空间、转移概率、发射概率、初始概率）
- [ ] 极大似然估计的公式推导（参考老师 MLE pdf）
- [ ] 加 α 平滑的具体公式
- [x] Viterbi 算法的递推式和回溯
- [ ] Hard EM 和 Soft EM 的区别（前向后向算法）
- [ ] 我的实现代码
- [ ] 在 dev.conll 上的准确率结果
- [ ] 5 个种子的无监督实验结果对比

---

## 日志

> [!tip] 日志约定
> 记录每天的进展和困难，呼应李老师"发日志讲进展和困难"的要求。

（待开始）

---

## 我的理解

（待完成后补充）
