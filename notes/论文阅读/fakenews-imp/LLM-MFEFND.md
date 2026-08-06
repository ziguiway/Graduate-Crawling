---
paper_type: research
status: reading
tags:
  - paper/research
---

# LLM-MFEFND

- 标题：Multimodal fusion with LLM content via hierarchical progressive transformer for explainable fake news detection
- 作者 / 年份 / 期刊：Lidong Wang, Xun Li, Bin Zhou, Yin Zhang, Jie Yuan, Hua Hu（杭州师范大学 + 浙江大学 + 浙江万里学院）/ 2026 / Information Processing and Management (IPM 63, 104700)
- PDF：[[Multimodal fusion with LLM content via hierarchical progressive transformer for explainable fake news detection.pdf]]
- 代码：[li1608335419/LLM-MFEFND](https://github.com/li1608335419/LLM-MFEFND)（含自建 LLM 背景/评论数据集，分量最足）
- 精读稿：[[paper|LLM-MFEFND reader/paper.md]]（全文中英对照 + 41 公式 + 图表）
- 定位：[[ARG]]（#21 母论文）的多模态直系后代；索引 [[00-索引]] #15；与 [[LFND-AB]]（#11）同属"ARG 家族"

## 一句话概括

>

## 摘要四要素

- **What（什么问题）**：
- **Why（为什么重要）**：
- **How（怎么做）**：
- **So What（效果如何）**：

## 核心创新（1～3 点）

1.
2.

## 方法

- 输入：
- 输出：
- 核心思路：
- 关键模块：
- 损失函数：

## 框架图

![[Multimodal fusion with LLM content via hierarchical progressive transformer for explainable fake news detection.pdf#page=5]]

%% Fig.2 总体工作流（知识抽取→HPT 融合→解释预测）在 #page=5；Fig.4 HPT 结构在 #page=7；Fig.6 MLIME 流程在 #page=10 %%

## 实验设置

- 数据集：
- Baseline：
- 评价指标：
- 关键参数：

## 实验结果

- 最重要的结果：
- 比 Baseline 好在哪里：
- 消融实验说明了什么：
- 作者的结论：

## 与其他论文的关系

- 和 [[ARG]]（#21 母论文）的区别：直系后代。双向 cross-attention 几乎原样继承（本文 Eq.6–13 ≈ ARG Eq.2–8）；把 ARG 的"分析视角理由"换成 **LLM 背景知识 + 5 条模拟评论**；纯文本扩展到**图文多模态**（新增 CLIP 对齐特征）；单步交互+加权聚合升级为 **HPT 五路渐进融合**；LLM 从 GPT-3.5 换成 **ChatGLM/DeepSeek**。本文把 ARG 列为 LLM 系 baseline 并超越它。
- 和 [[LFND-AB]]（#11）的区别：同属 ARG 家族但分支不同——LFND-AB 走"双立场理由 + 对比学习 + AdaBoost"（纯文本、治不平衡），本文走"背景知识+评论 + 多模态 + HPT + 可解释"。
- 可以借鉴的点：

## 局限与疑问

- 论文的局限：
- 我没看懂的地方：
- 我对结论的怀疑：

## 我的理解 / 个人思考

- 如果让我向同学解释：
- 和已有知识的联系：
- 对自己研究的启发：

> **一句话记忆：** ______。
