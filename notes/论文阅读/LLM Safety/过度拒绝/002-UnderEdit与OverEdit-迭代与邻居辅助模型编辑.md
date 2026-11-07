---
title: Resolving UnderEdit & OverEdit：迭代与邻居辅助模型编辑
paper_type: methods
status: read
date: 2026-11-07
tags:
  - paper/research
  - model-editing
  - llm-safety
aliases:
  - Resolving UnderEdit & OverEdit
  - Iterative & Neighbor-Assisted Model Editing
---

# Resolving UnderEdit & OverEdit：迭代与邻居辅助模型编辑

> Source coverage: Full paper（本地 23 页 PDF，含附录 A–H）
> Extraction confidence: High（已目视核对核心方法、主表与附录图表；部分原文记号和数字存在不一致，见第 13 节）
> Locator mode: page-grounded
> Primary analytical lens: methods
> Secondary analytical lens: None
> Context verification: Targeted external check（核验出版信息、官方代码说明及若干直接相关工作，未做完整新颖性检索）
> Card completeness: Complete relative to supplied source

页码均指 **PDF 文件页码，从 1 开始**；对应印刷页码 14786–14808。[Paper] 表示原文报告，[External] 表示外部来源核验，[Analysis] 表示阅读分析，[Hypothesis] 表示待验证设想。未运行模型编辑实验；代码部分只核对官方 README，未审计实现。

原文：[[../../../../papers/llm-safety/过度拒绝/Baghel 等 - Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing.pdf|本地论文 PDF]]。

## 01 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Resolving UnderEdit & OverEdit with Iterative & Neighbor-Assisted Model Editing |
| 作者 | Bhiman Kumar Baghel、Emma Jordan、Zheyuan Ryan Shi、Xiang Lorraine Li |
| 单位 | University of Pittsburgh，Department of Computer Science |
| 出处 | Findings of EMNLP 2025，14786–14808 |
| DOI | 10.18653/v1/2025.findings-emnlp.798 |
| arXiv | 2503.11895；本笔记以用户提供的正式会议版 PDF 为准 |
| 代码 | https://github.com/bhimanbaghel/ResolveUnderOverEdit |
| 数据 | COUNTERFACT、ZsRE；ME-PPL-50 用作崩溃监测代理 |
| 类型 / 关键词 | 方法论文；locate-and-edit、UnderEdit、OverEdit、locality、迭代编辑 |
| 与当前方向的关系 | 为“安全编辑如何避免误伤无害请求”提供保护邻居和诊断写入误差的思路；本文没有直接实验过度拒绝 |

书目信息来自 PDF 首页及 [ACL 正式记录](https://aclanthology.org/2025.findings-emnlp.798/)。[Paper: PDF p. 1, title and abstract]

**术语约定：**

| 原文术语 | 本文使用的解释 | 易混淆项与处理 |
|---|---|---|
| UnderEdit | 目标编辑没有成功生效 | 与“不充分安全对齐”无直接等价关系 |
| OverEdit | 误改本应保持的邻近知识 | 不直接翻译成“过度拒绝” |
| OPTIMIZATION | 求目标隐藏状态的阶段 | 求的是激活改变量，尚未永久写入权重 |
| SPREAD | 将目标状态落实为权重更新的阶段 | 不表示在生成时传播文本 |
| Neighbor-Assisted，NA_ | 使用邻近事实辅助编辑 | 实验中每条编辑只使用一条辅助邻居 |
| Efficacy / Generalization / Specificity | 目标有效性 / 改写泛化 / 邻居保持 | 分别报告 Accuracy 和 Success |
| key / value | MLP 关联记忆的输入 / 输出表示 | 与 attention 的 K/V 投影矩阵区分 |

## 02 一句话总结

针对事实编辑“没有改进去”和“顺带改错邻居”两种失败，作者反复执行目标状态优化与权重写入，并在目标状态优化中加入一条应保持的邻近事实，在若干稳定的模型与编辑器组合上提高编辑有效性或邻居保持，但没有消除两者之间的权衡，也没有保证不崩溃。[Paper: PDF p. 4, Section 3.1][Paper: PDF p. 5, Section 3.2][Paper: PDF p. 9, Limitations]

## 03 研究问题

原文用一个反事实编辑例子说明任务：将“iPad 的生产商是 Apple”改成“iPad 的生产商是 Honda”。这只是基准任务指定的事实替换，不代表真实世界知识。[Paper: PDF p. 1, Figure 1]

```text
应改：iPad   -> 生产商 -> Honda
应留：iPhone -> 生产商 -> Apple

UnderEdit：iPad 仍然输出 Apple。
OverEdit：iPad 改对了，但 iPhone 也跟着输出 Honda。
```

已有编辑器可以算出一个让目标答案概率很高的隐藏状态，却不一定能通过有限的权重修改复现它；继续强化编辑，又可能干扰共享参数承载的其他事实。[Paper: PDF p. 3, OPTIMIZATION and SPREAD][Paper: PDF p. 4, Section 3.1]

**精确问题：能否在不重新设计底层编辑器的前提下，通过反复修正目标状态与实际写入之间的差距，减少 UnderEdit，并通过显式保护邻居减少 OverEdit？**

## 04 背景与发展路线

以下两阶段归纳首先是作者的分析框架，不能当作对全部模型编辑方法的普遍定理。[Paper: PDF p. 2, Section 2][Paper: PDF p. 3, Table 1]

| 路线 | 做法与优势 | 与本文的关系或边界 |
|---|---|---|
| ROME | 在单层 MLP 输出权重中写入一个新映射 | 是单事实编辑器；本文的大批量比较将事实依次写入，不能等同于原生批编辑 |
| MEMIT / PMET | 跨多个层写入一批事实；PMET 的目标状态求法有所区别 | 本文主要在这些基础上加外层迭代和邻居损失 |
| AlphaEdit | 在权重更新中加入既有知识相关的子空间约束 | 本文发现迭代后仍可能损失 specificity；这不否定其约束的原始作用 |
| ENCORE | 用早停与权重范数约束应对长期编辑中的过拟合和退化 | 与本文“继续迭代提高写入程度”形成需要实测的张力 |
| 邻居辅助微调 | 编辑时同时训练未编辑事实以维持 locality | 是本文邻居辅助设计的直接启发之一 |
| 本文 | 重复整个两阶段流程；在目标状态优化中加入邻居 | 是底层编辑器上的流程与目标函数修改 |

[External] “保留未编辑事实”并非从本文开始：[Model Editing by Standard Fine-Tuning](https://aclanthology.org/2024.findings-acl.352/) 已在微调中加入随机或相似未编辑事实以维持 locality。[ENCORE 原文](https://arxiv.org/abs/2502.01636) 则研究连续编辑中的过拟合和权重范数增长。这里仅核验这些直接关联，没有确认本文的“首次”主张。

## 05 核心痛点

| 痛点 | 表现 | 作者解释 | 依据与证据强度 |
|---|---|---|---|
| UnderEdit | 新事实仍未成为模型预测 | 目标状态可能不够好，或权重写入无法复现目标状态 | 图 3、图 5 和分阶段困惑度支持存在落差；并未穷尽失败原因 |
| OverEdit | 邻居答案被目标答案污染 | 强化新事实时没有足够约束邻居，且参数共享 | 表 2 显示迭代可能提高 efficacy、同时降低 specificity |
| 不知道何时停止 | 写入不够与反复修改退化之间难以选点 | 只观察目标状态优化或相邻轮变化，未必反映真正写入程度 | 表 4 比较三种停止规则，优势依设置而变 |
| 编辑崩溃 | 通用文本困惑度急剧增大 | 模型、算法、编辑过程的相互作用 | 存在实测失败，确切因果机制尚未被隔离 |

[Paper: PDF p. 4, Section 3.1][Paper: PDF p. 6, Table 2][Paper: PDF p. 8, Table 4 and Section 5][Paper: PDF p. 17, Figure 5 and Appendix E]

## 06 核心想法

1. **表层做法：**不是只编辑一次，而是每轮基于当前模型重新求目标隐藏状态、重新写入权重；求目标状态时同时要求一条邻近事实保持原答案。[Paper: PDF p. 4, Section 3.1][Paper: PDF p. 5, Section 3.2]
2. **关键洞察：**“我找到了一个能产生目标答案的激活”与“我把这种行为写进了真实模型”是两件事。作者试图通过分阶段目标 token 困惑度的差距观察这件事。[Paper: PDF p. 4, stopping criterion]
3. **[Analysis] 可迁移启发：**既要检查干预目标是否合理，也要检查权重修改后目标是否兑现；保护样本应参与构造干预，而不只是事后报告一个能力分数。不过，这不保证任何任务中都能同时实现两类目标。

## 07 方法总览

**输入：**白盒模型、事实替换集合 `(s, r, o -> o*)`、底层编辑器，以及邻居辅助版本所需的已知正确邻居 `(s_neighbor, r, o)`。目标对象按单 token 建模；不是任意长回答的编辑。[Paper: PDF p. 2, Section 2][Paper: PDF p. 5, Section 3.2]

**输出：**更新了选定层 MLP 输出权重的模型。MEMIT、PMET、AlphaEdit 的层选择与更新规则沿用底层方法；PMET 求目标状态时涉及 attention，但不表示本文把 attention 的 V 权重作为新编辑目标。[Paper: PDF p. 3, Table 1 and Figure 2]

```text
当前模型 theta_k + 同一批待编辑事实
    |
    v
OPTIMIZATION
  在选定层、subject 最后一个 token 处优化 delta_k
  得到目标隐藏状态 h_bar_k
  邻居辅助版本：同时惩罚邻居失去原答案
    |
    v
SPREAD
  依据底层编辑器更新 MLP 输出权重
  得到 theta_(k+1) 及实际状态 h_hat_k
    |
    v
比较目标状态与真实写入后状态的目标 token 困惑度
  差距 <= 1：结束
  否则：用新模型重新执行上面两阶段（实验上限为 10 轮）
```

这里有内外两层迭代：内层用梯度下降优化 `delta`；本文增加的是外层反复运行**整个编辑流程**。不能把方法概括成“让原先的 delta 多训练几步”。[Paper: PDF p. 4, Section 3.1]

训练与开销：无需训练一个通用元编辑器，但每轮求 `delta` 需要反向传播；权重更新成本也会重复。编辑结束后使用合并后的权重，方法没有要求每次生成时再做邻居检索。[Analysis; based on Paper: PDF p. 3, two-stage framework][Paper: PDF p. 15, Appendix B.1]

## 08 核心模块拆解

| 模块 | 功能 / 为什么需要 | 输入 → 输出 | 支持证据 | 移除的后果 |
|---|---|---|---|---|
| 目标状态优化 | 找到能预测新对象的状态 | 当前模型、事实、随机前缀 → delta 与 h_bar | 图 2 及第 2 节目标函数 | 每轮不重算，仅复用首轮目标：表 7 出现后续退化；仅在一个设置中隔离验证 |
| 权重写入 SPREAD | 将临时激活干预变成永久参数变化 | h_bar、keys、底层统计量 → 新权重 | 图 3、表 9 给出目标与实际的概率差距 | 不做则仅是激活干预，无法得到持久编辑模型，这是方法定义层面的判断 |
| 外层迭代 | 重新校正目标与写入落差 | theta_k → theta_(k+1) | 表 2，图 5 | 单轮编辑在部分组合上 efficacy 更低，但也可能更稳、更少误伤 |
| 邻居目标 | 让 delta 同时兼顾一个应保持的事实 | 一条辅助邻居 → 新损失 | 表 3、表 11 | 稳定组合上 specificity 通常更低；Llama-2 的失败提醒它不是无条件收益 |
| 停止规则 | 用阶段落差决定轮数 | 两个阶段的目标 token 概率 → 是否停止 | 表 4、表 13 | 换用相邻轮变化或单调性，部分设置更差，也有相同或略好结果 |
| 邻居过滤与划分 | 避免标签冲突并保留评估邻居 | 邻居集合与原模型预测 → 辅助 / 评估集合 | 附录 B.2、图 4、表 5 | 没有消融；预期会产生冲突或减少可评估样本，属于推测 |

[Paper: PDF p. 3, Figure 2][Paper: PDF p. 5, Figure 3][Paper: PDF p. 6, Table 2][Paper: PDF p. 7, Table 3][Paper: PDF p. 8, Table 4][Paper: PDF p. 15, Appendix B.2][Paper: PDF p. 16, Figure 4 and Table 5][Paper: PDF p. 18, Table 7][Paper: PDF p. 21, Table 11][Paper: PDF p. 23, Table 13]

## 09 核心公式与符号

原文核心公式没有正式编号。以下 F1–F6 是笔记的定位标签，不是论文公式编号。

### F1：临时求出理想状态

```text
x_j = (random_prefix_j, subject, relation)

g(delta)
  = mean_j[-log P_theta(o* | x_j; inject h + delta)]
    + KL(P_theta(. | s; inject h + delta) || P_theta(. | s))

delta* = argmin_delta g(delta)
h_bar = h + delta*
```

`h` 是选定层、subject 最后一个 token 的隐藏状态；`delta` 与它同维。第一项要求新答案概率高；第二项用 subject 相关输出分布约束变化，不等价于在全部无害请求上保持原行为。[Paper: PDF p. 3, OPTIMIZATION]

**形状示例（示意，不是论文实验参数）：**设 `h = [0.1, -0.2, 0.3, 0.0]`，`delta = [0.2, 0.0, -0.1, 0.1]`，则 `h_bar = [0.3, -0.2, 0.2, 0.1]`。优化改变的是一个四维向量，而不是这一步就把整个模型重训。真实维度取决于模型。

### F2：把目标状态写入权重

附录给出的 MEMIT 更新形式：

```text
DeltaW = R K^T (C + K K^T)^(-1)
W_new  = W + DeltaW
```

`K` 收集编辑 facts 对应的 MLP keys；`R` 表示需补足的输出残差；`C` 是既有 keys 的未中心化协方差相关项。若 key 维度为 `d_ff`、输出维度为 `d_model`、事实数为 `m`：

```text
K:       d_ff x m
R:       d_model x m
C:       d_ff x d_ff
DeltaW:  d_model x d_ff
```

直觉是同时满足新映射并控制对旧映射的干扰。批编辑更新通常不能一概称为“秩一更新”；ROME 的单事实形式才是典型 rank-one 更新。本文不同底层编辑器的 SPREAD 规则仍然不同。[Paper: PDF p. 12, Appendix A, MEMIT][Paper: PDF p. 13, Appendix A, ROME]

### F3：外层迭代

```text
h_bar_k     = h_k + delta_k*
theta_(k+1) = SPREAD(theta_k, h_bar_k)
h_hat_k     = h_(k+1)
```

`h_bar_k` 是临时干预状态，`h_hat_k` 是写权重后真正前向计算得到的状态。模型变了，下一轮就重新优化 `delta`；不能因为目标概率接近，就认定目标向量一模一样。[Paper: PDF p. 4, Section 3.1][Paper: PDF p. 16, Appendix D]

### F4：停止规则与“困惑度”

```text
q(theta, h) = mean_i[exp(-log P_theta(o_i* | x_i; h))]
            = mean_i[1 / P_theta(o_i* | x_i; h)]

gap_k = abs(q(theta_(k+1), h_hat_k) - q(theta_k, h_bar_k))
stop if gap_k <= epsilon，本文 epsilon = 1
```

这里用 `q` 重写作者记号 `p`，避免与 token 概率混淆。它按文中公式是逐事实单 token 困惑度的算术平均；不要和长文本的标准 corpus perplexity 混为一谈。原式的平均分母写 `m`、求和上界写 `n`，与文字“m 条编辑”不一致；这里按文字语义写为同一批编辑的平均。[Paper: PDF p. 4, perplexity definition]

**小例子（示意）：**目标激活下 `P(o*)=0.9`，写权重后只有 `0.2`，对应倒数为 `1.11` 与 `5`，差约 `3.89`，需要继续；若写入后达到 `0.8`，差约 `0.14`，会满足阈值。

**[Analysis] 关键边界：**停止只表示“写入结果接近目标”，不表示目标本身足够好，也不直接约束邻居、完整输出分布或通用能力。两个很差的目标 token 概率也可能接近；相同目标概率也不能推出两个隐藏向量接近。

### F5：邻居辅助损失

```text
g_neighbor(delta)
  = g(delta)
    - log P_theta(o | x_neighbor; hidden-state intervention delta)

x_neighbor = (s_neighbor, r)，不加随机前缀
```

把“该改的事实”和“一条不该改的邻近事实”放进同一次目标状态优化。原文用同一个 `delta` 表达联合干预；邻居样本有自己的 subject token 位置。不能把公式理解为把目标句子的整条激活序列复制给邻居。具体 hook 实现本次未核对。[Paper: PDF p. 5, Section 3.2]

与在 SPREAD 中直接联合解一个邻居权重约束不同，本文新增约束位于 **OPTIMIZATION**；因此临时保护是否真正写入仍需评估。

### F6：评估指标

```text
Efficacy Success  : P(o*) > P(o)
Efficacy Accuracy : argmax_vocab P(token) = o*

Specificity Success  : P(o) > P(o*)，在邻居 prompt 上
Specificity Accuracy : argmax_vocab P(token) = o，使用相应邻居标签

Score = harmonic_mean(Efficacy, Generalization, Specificity)
```

Success 只是新旧两个答案之间的排序。例如 `P(o*)=0.02 > P(o)=0.01`，但另一个词概率为 `0.3`，此时 Success 成功、Accuracy 失败。不能只看 Success 就说模型能正确作答。[Paper: PDF p. 6, Section 4.2][Paper: PDF p. 7, Section 4.2 continuation]

**公式清单核对：**已覆盖正文原始优化目标、`h_bar=h+delta`、更新后状态关系、阶段困惑度与阈值、邻居辅助损失，以及理解 SPREAD 必需的 MEMIT 更新。准备脚本唯一识别出的 “Equation 1” 位于 PDF 第 13 页，目视核对实际是正文枚举“(1) preserves … (2) …”，不是正式编号公式；该处真正相关的 ROME 约束是 `min ||W_new K_old - V_old||^2, s.t. W_new k* = v*`，用于对照单事实精确写入与旧知识保持。[Paper: PDF p. 13, Appendix A, ROME]

## 10 实验设计与证据链

### 设置与评估口径

- **模型与编辑器：**GPT-2 XL 1.5B、GPT-J 6B、Llama-2 7B、Llama-3.1 8B；ROME、MEMIT、PMET、AlphaEdit。矩阵并非每个子实验都覆盖所有组合。[Paper: PDF p. 5, Section 4][Paper: PDF p. 6, Table 2]
- **规模：**迭代主实验按每次 1,000 条编辑，三组互不重叠的抽样，报告平均结果；最多 10 轮。邻居辅助因过滤而使用另外的子集，不能把表 2 和表 3 数字直接当作前后对照。[Paper: PDF p. 5, Section 4.1][Paper: PDF p. 7, Table 3 caption]
- **邻居来源：**已知相同关系、共享原对象的事实；过滤掉原模型预测与数据标签不一致的邻居。一条用于辅助，其余用于评估，因而不是把同一条辅助样本直接拿来算全部 specificity。ZsRE 每条只有一个邻居，未用于 NA 实验。[Paper: PDF p. 15, Appendix B.2]
- **邻居数量：**GPT-2 XL 739 条编辑，评估邻居平均向上取整为 2；Llama-2 1,340 条，评估邻居平均向上取整为 4。GPT-J 的条数主表写 960、表 5 和表 11 写 690；评估邻居数为 4，条数矛盾不能自行消解。[Paper: PDF p. 7, Table 3][Paper: PDF p. 16, Table 5][Paper: PDF p. 21, Table 11]
- **成本与不确定性：**使用 EasyEdit 默认超参数；表 6 列出 L40S 48GB / A100 80GB 资源，但没有完整给出各方法端到端用时和等预算曲线。主要表没有标准差或置信区间。ME-PPL-50 只有 50 条平均约 22 tokens 的文本，是崩溃代理，非全面能力测试。[Paper: PDF p. 5, Section 4][Paper: PDF p. 6, Section 4.1][Paper: PDF p. 17, Table 6]

### 核心结果

下表中的数字均为 **Accuracy**，除明确标注的困惑度。箭头表示同一行所述设置中的对照。

| 实验 | 检验什么 | 对照与结果 | 实际支持 | 不支持的更强结论 | 来源 |
|---|---|---|---|---|---|
| GPT-2 XL + MEMIT 迭代 | 多轮能否弥补写入失败 | 第 1 → 4 轮：efficacy 79.00% → 99.67%，generalization 22.00% → 35.67%，specificity 75.00% → 68.67%，综合 42.67% → 56.67% | 目标与改写收益伴随邻居损失 | 所有维度同时改进 | [Paper: PDF p. 6, Table 2] |
| GPT-2 XL + PMET 迭代 | 单轮很弱时能否补足 | 第 1 → 10 轮：efficacy 21.67% → 99.33%，综合 8.33% → 47.00% | 某些组合改善很大 | 摘要“38 个百分点”就是纯 UnderEdit 发生率的下降 | [Paper: PDF p. 6, Table 2] |
| GPT-2 XL + MEMIT 邻居辅助 | 一条邻居能否保护其他邻居 | 同一筛选子集、第 4 轮：efficacy 99% → 99%，generalization 38% → 36%，specificity 52% → 86%，综合 54% → 60% | 保护可扩展到未用于辅助的评估邻居，且有轻微泛化代价 | 所有邻居或所有知识均保持 | [Paper: PDF p. 7, Table 3] |
| GPT-J + MEMIT 邻居辅助 | 是否仅 GPT-2 有收益 | 第 2 轮：specificity 63% → 81%，generalization 79% → 75%，综合 78% → 84% | 另一模型上存在相同方向的权衡 | 可推广到所有架构 | [Paper: PDF p. 7, Table 3] |
| 只重做 SPREAD | 是否可以固定首轮目标以省计算 | GPT-2 XL + MEMIT，第 1 → 5 轮：efficacy 79% → 20%；ME-PPL-50 57.98 → 602.14，中间第 2 轮曾改善 | 在该设置中复用目标随后退化，支持重算目标的价值 | 每一种编辑器都必须用完全相同重算策略 | [Paper: PDF p. 18, Table 7] |
| Llama-2 + MEMIT | 迭代是否总有效 | 第 1 → 2 轮：efficacy 91.67% → 14.33%；ME-PPL-50 42.10 → 9,370 | 继续编辑可能快速崩溃 | 多迭代几轮总能修正 UnderEdit | [Paper: PDF p. 6, Table 2] |
| GPT-J + AlphaEdit | 子空间约束是否保证稳定 | 表 2 第 5 轮 ME-PPL-50 为 5.84×10^6，efficacy 从第 1 轮 99.33% 降至 82% | 保护设计不保证多轮稳定 | 只有 Llama-2 或 ROME 会失败 | [Paper: PDF p. 6, Table 2] |
| Llama-2 + NA_PMET | 加邻居是否必然安全 | NA 第 10 轮未达停止条件；specificity 26%，ME-PPL-50 1,414.01；无 NA 第 3 轮 specificity 70%、困惑度 31 | 组合存在灾难性失败；两端轮数不同需注意 | 加一个邻居是无代价补丁 | [Paper: PDF p. 7, Table 3] |
| 停止规则比较 | 阶段差距是否是实用规则 | 多数选择效果相当或较好；例如 ZsRE GPT-J + MEMIT 的另一规则综合 75.00%，本文规则为 74.67% | 有限设置中的经验启发式 | 证明全局最优或严格优于所有停止规则 | [Paper: PDF p. 8, Table 4][Paper: PDF p. 23, Table 13] |

**摘要数字如何读：**摘要称最多减少 UnderEdit 38 个百分点、OverEdit 6 个百分点；正文 5.1/5.2 的表述又把这些数值与综合 score 改善联系起来。表 2 中 GPT-2 XL + PMET 综合 Accuracy 增加 38.67 个百分点，而 efficacy 增加 77.66；表 3 中 GPT-2 XL + MEMIT 综合增加 6 个百分点，而 specificity 增加 34。应直接报告指标名和对应表格，避免把综合分变化解释成某一种错误率变化。[Paper: PDF p. 1, Abstract][Paper: PDF p. 6, Table 2][Paper: PDF p. 7, Section 5.1 and Table 3][Paper: PDF p. 8, Section 5.2]

### 全部图表的论证用途

| 图表 | PDF 页 | 作用 |
|---|---|---|
| Figure 1 | 1 | 用 iPad / iPhone 区分 UnderEdit 与 OverEdit；是示意，不是统计证据 |
| Figure 2 | 3 | 对照目标状态优化位置和真正更新的 MLP 权重 |
| Figure 3 | 5 | 单例展示两阶段目标 token 困惑度落差随迭代缩小；纵轴做了 Box-Cox 变换 |
| Figure 4 | 16 | 展示辅助邻居与评估邻居的数据字段及其分开使用 |
| Figure 5 | 17 | 跟踪首轮失败的 UnderEdit 子集，观察后续 efficacy 与 gap 改善 |
| Table 1 | 3 | 多种编辑器的两阶段归纳；列出兼容性设想不等于逐个实测 |
| Table 2 | 6 | COUNTERFACT 迭代主结果，包含收益、权衡和崩溃 |
| Table 3 | 7 | 邻居辅助主结果；需与同子集无邻居版本比较 |
| Table 4 | 8 | 三种停止规则的选点比较 |
| Table 5 | 16 | 邻居辅助可用编辑条数、辅助邻居数和评估邻居数 |
| Table 6 | 17 | GPU 配置，不能替代完整时间或显存峰值比较 |
| Table 7 | 18 | 固定首轮目标、仅重复 SPREAD 的消融 |
| Table 8 | 18 | 无前缀 NA 与加前缀 NAP 的结果；加前缀综合略升、specificity 略降 |
| Table 9 | 19 | COUNTERFACT 详细迭代轨迹；所列主要为三模型三编辑器，不含表 2 所有组合 |
| Table 10 | 20 | ZsRE 详细轨迹；未覆盖主表声称矩阵的全部组合 |
| Table 11 | 21 | 邻居辅助各轮轨迹及同子集基线 |
| Table 12 | 22 | 邻居前缀分析的各轮轨迹 |
| Table 13 | 23 | 停止规则完整轨迹，展示准确率平台和较晚停止的情况 |

[Paper: PDF p. 1, Figure 1][Paper: PDF p. 3, Figure 2 and Table 1][Paper: PDF p. 5, Figure 3][Paper: PDF p. 6, Table 2][Paper: PDF p. 7, Table 3][Paper: PDF p. 8, Table 4][Paper: PDF p. 16, Figure 4 and Table 5][Paper: PDF p. 17, Figure 5 and Table 6][Paper: PDF p. 18, Table 7 and Table 8][Paper: PDF p. 19, Table 9][Paper: PDF p. 20, Table 10][Paper: PDF p. 21, Table 11][Paper: PDF p. 22, Table 12][Paper: PDF p. 23, Table 13]

## 11 如何正确理解结论

**可以支持：**在部分稳定组合中，重复整个编辑流程能显著补足单轮目标事实写入；在筛选过的 COUNTERFACT 邻居子集上，一条正确邻居可以改善其他评估邻居的保持。[Paper: PDF p. 6, Table 2][Paper: PDF p. 7, Table 3]

**必须带上的边界：**

- 这是结构化事实、单 token 答案的编辑。没有验证自由文本安全回复、越狱防御、过度拒绝或多轮对话。[Paper: PDF p. 2, Section 2][Paper: PDF p. 9, Limitations]
- NA 依赖已知正确邻居及筛选，不是自动发现全部保护边界的端到端系统。评估邻居每条只有少数几个，不能据此推断全局能力保持。[Paper: PDF p. 15, Appendix B.2][Paper: PDF p. 16, Table 5]
- 迭代是在同一批编辑上做多轮，不等于已经验证对长期不断到来的不同编辑仍稳定。[Paper: PDF p. 4, Section 3.1][Paper: PDF p. 5, Section 4.1]
- 目标 token 概率差距不等于隐藏状态距离，不提供全局收敛保证。ME-PPL-50 是代理检查，有限文本上的困惑度正常也不证明所有能力正常。[Analysis; based on Paper: PDF p. 4, stopping criterion][Paper: PDF p. 6, Section 4.1]
- 泛化与 specificity 的权衡仍存在。原文将其称作 No-Free-Lunch effect，但这些实验不是不可兼得的数学证明。[Paper: PDF p. 8, Section 5.1][Paper: PDF p. 9, Limitations]
- 成本发生在编辑阶段，往往要做多轮反向传播和权重求解；只与单轮比较，无法区分所有收益中有多少来自额外计算预算。[Analysis; based on Paper: PDF p. 4, Section 3.1]

[External] 官方 [README](https://github.com/bhimanbaghel/ResolveUnderOverEdit) 明确说明发布脚本按指定轮数跑，默认不执行基于困惑度的早停，以便观察完整趋势；“论文中按阈值选出的结果”与“代码默认自动停止”需要区分。本次没有执行或检查其代码路径。

**有边界的主结论：本文提供了两种可加在部分 locate-and-edit 编辑器上的经验改进，缓解特定事实编辑设置中的未写入与邻居误伤；没有把编辑有效性、泛化、知识保持、稳定性四者同时解决。**

## 12 作者明确承认的局限

| 作者承认的局限 | 具体表现 | 作者提出的后续方向 | 来源 |
|---|---|---|---|
| 有效性 / 泛化与 specificity 的权衡持续存在 | 共享参数难以只承载一个事实，本文也不能完全隔离 | 更精确的编辑 | [Paper: PDF p. 9, Limitations] |
| 仅研究结构化事实三元组 | 自由文本知识的定位困难 | 支持非结构化输入 | [Paper: PDF p. 9, Limitations] |
| 计算资源限制 | 未覆盖更大 batch、GPT-NeoX 20B 和更大 Llama | 验证规模扩大后趋势是否仍成立 | [Paper: PDF p. 9, Limitations] |

作者另外提到的约束：寻找所有可能邻居是困难问题；Llama-2 等设置的崩溃原因尚需深入研究；当前实现要借下一轮计算前一轮的部分困惑度。这些来自正文或实现说明，不冒充正式 Limitations 列出的三项。[Paper: PDF p. 8, Section 5.2][Paper: PDF p. 15, Appendix B.1 and B.2]

## 13 批判性分析

| [Analysis] 观察 | 问题或替代解释 | 为什么重要 | 如何验证 | 依据 |
|---|---|---|---|---|
| 反复编辑使用更多计算 | 收益可能部分来自预算，而非外层迭代结构本身 | 影响效率与方法归因 | 在等 GPU 时间下比较单轮更长 delta 优化、外层迭代、仅 SPREAD，以及重算目标的简化策略 | PDF p. 4；p. 18 Table 7 |
| 停止只看目标 token 的平均倒数概率差距 | 少数低概率样本会主导均值；平均差距小可能掩盖个别严重失败 | 影响停止阈值迁移和尾部可靠性 | 同时记录逐例 gap、失败率、分位数、目标状态质量；在独立验证集选阈值 | PDF p. 4 stopping criterion |
| 加邻居前做标签一致性筛选 | NA 的收益在较干净、模型本来会答的邻居上评估 | 实际邻居未必可获得同样正确的标签 | 分别测试原模型预测正确 / 错误、自动检索 / 人工标签邻居，并报告筛除比例 | PDF p. 15 Appendix B.2 |
| 辅助邻居与测试邻居共享关系和原对象 | 可以证明一种近邻保持，不能代表跨关系或远距离知识保持 | 防止夸大全局 locality | 新增同主体异关系、不同原对象、远邻、真实任务测评；分组报告 | PDF p. 5 Section 3.2；p. 16 Figure 4 |
| 崩溃的归因主要依据模型间差异 | 同数据集并不能排除模型与样本难度交互，少数轮次也可能累积大更新 | 决定什么情况下应继续编辑 | 固定编辑集合、改变单轮更新范数和轮数，追踪范数、输出熵及保留任务；做逐次回滚 | PDF p. 17 Appendix E |
| GPT-J 样本数 960 / 690 不一致 | 表 3 与表 5、表 11 的实验口径需核对 | 重现与跨表匹配受影响 | 核对数据文件、日志、实际评估分母；在解决前并列记录两个数 | PDF p. 7 Table 3；p. 16 Table 5；p. 21 Table 11 |
| GPT-J + AlphaEdit 也报告崩溃 | 与附录 E“GPT models 未出现类似崩溃”的笼统叙述不一致 | 不能把失败全部归于 Llama-2 | 检查 AlphaEdit 的具体轮次、日志和模型范数；保留表 2 的失败结果 | PDF p. 6 Table 2；p. 17 Appendix E |
| 图 3 的阶段概率逐渐一致 | 并不能证明隐藏向量的欧氏距离收敛，多个状态可能给出相同目标概率 | 机制解释需比效果观察更克制 | 另测状态距离、角度与分布 KL，以及逐轮干预/真实状态替换实验 | PDF p. 4 Section 3.1；p. 5 Figure 3 |
| 邻居前缀文字与表格方向不同 | F.2 说加前缀提高 specificity；表 8 相对无前缀 NA 为 86→84、85→84 | 防止照抄作者解释 | 区分“相对不使用邻居”与“相对无前缀邻居”；以后者为前缀消融对照 | PDF p. 18 Table 8 and F.2 |

原文中的 “Equation 1” 自动识别问题、困惑度求和记号不一致、摘要百分点与具体指标不完全对应，已在第 09–10 节单独标注。它们要求精确重现时核对实现，但不能单凭这些表述问题否定所有实测收益。

## 14 学到的可迁移知识

以下是 **Agent-derived knowledge candidates**，不是用户已经给出的判断。

1. **激活目标与参数实现的分离。**一次临时干预成功，不代表修改权重后会成功。阅读编辑论文时应分别找出两阶段的评估，不能只看优化损失。[Analysis; based on Paper: PDF p. 3, Section 2][Paper: PDF p. 5, Figure 3]
2. **局部性需要专门的数据和目标。**要求新事实正确，与要求邻居不变是不同约束。通用正则项不自动等于保护最容易被误伤的样本。[Analysis; based on Paper: PDF p. 5, Section 3.2][Paper: PDF p. 7, Table 3]
3. **二选一胜出弱于全词表正确。**编辑 Success 与 Accuracy 应并列看，避免只有目标相对旧答案上升，却仍生成第三个错误词。[Paper: PDF p. 7, Section 4.2 continuation]
4. **平均成功掩盖坏组合。**做方法比较时应把崩溃组合计入可靠性评估，不能只从稳定子集总结普适性。[Analysis; based on Paper: PDF p. 6, Table 2][Paper: PDF p. 7, Table 3]

## 15 与现有研究的联系

### 与过度拒绝的关系：可以迁移问题结构，不能直接迁移结论

```text
事实编辑：
  应改变：目标事实答案
  应保持：邻居事实答案

安全编辑（候选映射）：
  应改变：有害请求的违规回答 -> 安全回应
  应保持：语义相近但无害请求 -> 有用回答

模型原本已经误拒的无害请求：
  “保持原行为”会保留错误；需要纠正标签或可靠回答目标。
```

[Analysis] OverEdit 是编辑造成的行为误伤；over-refusal 是无害请求被拒绝。只有在编辑引入或扩大了无害请求误拒时，后者才是前者的一种表现。原模型原有的误拒不应全部归因于编辑，也不能靠“维持原模型”自动修好。[Paper: PDF p. 1, Figure 1][Paper: PDF p. 5, Section 3.2]

**[External] DELMAN：**已在良性输入上加入 KL 一致性以减少无害场景被触发安全回答。因此“安全编辑加良性保持项”本身不是一个足够的差异点。[DELMAN 正式论文记录](https://aclanthology.org/2025.findings-acl.598/)；对应本地笔记 [[../模型编辑/002-DELMAN Dynamic Defense Against LLM Jailbreaking with Model Editing]]。

**[Analysis] 更具体的区别候选：**保护最容易被新编辑误伤的“危险表面特征相近、实际无害”邻居，并与随机良性保持项比较。但本文使用的“相同关系、相同旧对象”不直接适用于开放式安全指令；必须另定义邻居并验证标签，不能仅靠文本相似度替代意图判断。

**[External] ProSafePrune：**它直接研究无害但带风险词汇的请求被误拒，并提出子空间投影后的低秩参数剪枝。[正式论文 PDF](https://openreview.net/pdf?id=QkHKaPfRAB)；对应本地笔记 [[001-ProSafePrune-投影安全剪枝缓解过度拒绝]]。

**[Analysis] 两者可形成不同位置的控制：**本文邻居约束放在编辑过程中；ProSafePrune 的思路可用于研究编辑后的误拒修正。它们是否互补、是否只是降低编辑强度，必须在保留有害防御效果的前提下比较。本文没有验证 V 投影修复，也不能为“FFN 安全编辑后改 V 一定不损伤安全”提供依据。

**[External] EVOKE / LTI：**[Uncovering Overfitting in Large Language Model Editing](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c592fc7e6207f82560ed45fece8d6937-Abstract-Conference.html) 研究编辑目标被赋予过高概率、妨碍复杂情境泛化的问题，提出 LTI 约束。[Analysis] 它提醒我们：阶段差距减小、目标概率提高，也可能是在更充分地实现一个过强的编辑目标；不能只追求把目标写得更牢。

## 16 研究候选与最小验证

以下是 **Agent-derived research candidates**。不是本文实验结果，也不是已经查新的新方法；本次只核验了前述直接相关工作。

### 候选 A：比较“高风险无害邻居保护”与随机良性保持

- **起点：**一条邻居能在筛选设置上提高其他邻居的保持，但实际安全场景的保护集合与事实三元组不同。[Paper: PDF p. 5, Section 3.2][Paper: PDF p. 7, Table 3]
- **我们想干嘛：**先测清楚，安全编辑时显式保护最容易被误拒的无害请求，是否比同样数量的随机良性请求更有效。
- **核心假说：[Hypothesis]** 在有害请求防御收益相当时，靠近编辑触发边界的已确认无害邻居，能减少新增误拒；收益不只是编辑变弱。
- **相对本文的变化：**把“同关系、同旧对象的事实邻居”替换为“风险词汇或语义相近，但意图无害的指令”；输出从单 token 对象变为序列，因此需重新定义 NLL / KL 和评估。
- **初始方法：**一个可承受的 7B/8B 指令模型，一个固定安全编辑器；先筛出编辑后出现明确新增误拒的小样本批次。每条编辑用 0 或 1 条保护指令；只选择原模型原本能正确回答的邻居，避免将原有误拒当作保护目标。
- **如何验证 / Validation：**比较原模型、原编辑器、随机良性保持、相似无害邻居保持，以及降低编辑强度的对照；统一层、编辑样本、保护条数和计算预算。首轮可使用 30 条有害编辑、各 100 条独立有害测试和独立无害测试、100 条无关能力检查；模板与语义改写族按组划分，避免保护样本泄露到测试。至少三个固定种子，初筛预计最多 15 次编辑运行；这是方案预算，不是已执行实验。
- **指标：**H＝有害行为是否仍被阻断；B＝无害请求新增误拒率、原有误拒率分别统计并检查回答质量；C＝无关能力保持。用自由生成和逐例输出判定，保留自动评测与人工抽检记录，不能只看拒绝开头的概率。
- **成功 / 停止标准：**预注册一个初筛门槛，例如相较原编辑器 H 下降不超过 2 个百分点、C 下降不超过 2 个百分点时，独立 B 的新增误拒降低至少 10 个百分点；小样本区间过宽时只进入复验，不宣称确定收益。若匹配 H 后随机保护、相似保护和降低强度无可区分，停止将结果解释为“邻居选择有特殊作用”。
- **可能失败：**相似度抓到危险词而没有抓到边界；无害标签错误；原模型目标回答已错；保护项只是削弱编辑；30 条编辑未能稳定产生可测误拒。
- **创新状态：partially checked。**已确认 DELMAN 有良性 KL、标准微调已有未编辑事实保持；“边界邻居选择”的增量仍需专门检索和实验。

### 候选 B：把“写入到位”与“行为仍可接受”分开设停止条件

- **起点：**原停止规则只看阶段目标 token 困惑度差，且若干组合崩溃。[Paper: PDF p. 4, stopping criterion][Paper: PDF p. 6, Table 2]
- **核心假说：[Hypothesis]** 只在写入差距下降、保护集未明显退化时接受下一轮，能在相近 efficacy 下保留更多邻居并减少崩溃。
- **相对本文的变化：**修改外层停止 / 接受规则，不改变底层 SPREAD；同时记录目标状态自身质量，排除“两个坏阶段恰好接近”。
- **初始方法：**先在事实编辑任务验证；每轮缓存前一版本，在小型独立验证保护集上检测退化，退化超过预注册阈值则回滚。测试集不参与停止。
- **如何验证 / Validation：**固定 GPT-2 XL + MEMIT、同批 100 条事实、最多 5 轮、三个种子；对比固定轮数、原 gap 停止、验证集接受规则，并将各规则评估开销计入预算。报告 efficacy、generalization、近邻与远邻 specificity、逐例失败率、ME-PPL-50、总用时和更新范数。若收益只在牺牲 efficacy 或消耗更多预算后出现，则假说不成立。
- **可能失败：**保护集太小而被过拟合；阶段 gap 对 tail failure 不敏感；接受规则太保守导致一直回滚；暂时的邻居下降可能是后续更优状态的必经阶段。
- **创新状态：unverified。**ENCORE 已有早停和范数控制，候选应首先作为诊断与基线实验，不能直接宣称新颖。
