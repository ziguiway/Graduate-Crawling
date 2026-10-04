---
paper_type: research
status: done
tags:
  - paper/research
---

# Visual Instruction Tuning (LLaVA)

- 作者 / 年份 / 会议：Haotian Liu 等（UW-Madison / Microsoft Research / Columbia）· NeurIPS 2023
- PDF：[[../../../papers/mllm-safety/01_LLaVA_2304.08485.pdf]]
- 代码：https://github.com/LLaVA-VL/LLaVA
- **定位：不是安全论文**，是后续 MLLM 安全研究最常用的底座模型（攻击靶子）

## 一句话概括

> 用 GPT-4 把图文对伪造成 158K 多模态指令数据，训练"CLIP + 一层投影 + Vicuna"的极简模型，证明多模态的瓶颈是**数据不是架构**。

## 摘要四要素

- **What**：开源多模态模型只会"描述图片"，不会按指令回答——缺图文对话训练数据
- **Why**：NLP 指令微调已证明有效，多模态没人做成
- **How**：GPT-4 读 caption+bbox 文字档案伪造指令数据；两阶段端到端微调
- **So What**：ScienceQA 92.53% SOTA；聊天能力达 GPT-4 的 85.1%；8×A100 一天训完

## 核心创新

1. 首次把 instruction tuning 扩展到视觉-语言多模态空间
2. 极简架构：CLIP ViT-L/14（冻结）+ 单层线性投影 W + Vicuna 13B，无任何新结构
3. LLaVA-Bench：GPT-4-as-judge 的相对评分评测协议（后成多模态评测标配）

## 方法

- **图像怎么进 LLM**（核心机制）：
	- 224×224 图 → CLIP 切 16×16 = 256 块 → 每块一个 1024 维向量
	- 投影 W（1024→4096）把图像向量变成与词向量同维度的"视觉 token"
	- [256 个图像 token] + [问题 token] 按**序列长度维度拼接**（不是加法），LLM 一视同仁处理
	- 图像 token 是连续向量、无词表，本质是"假装成 token"
- **两阶段训练**：
	- Stage 1 对齐（4h）：CLIP、LLM 全冻结，只训 W，任务"看图说描述"——逼翻译层学扎实
	- Stage 2 微调（10h）：CLIP 冻结，W + LLM 一起训，158K 指令数据
- **损失**：标准 next-token prediction，无新损失；只算回答 + `<STOP>`，问题不算
- **为什么图像 token 会被理解**：CLIP 用 4 亿图文对预训练，输出向量本身已含语义（猫图向量 ≈ "cat"）；W 只学坐标系翻译；LLM 本来就只认向量不认字——图像向量就是新生的词，微调教会它用法

## 训练

- **两阶段设计**：Stage 1 对齐 → Stage 2 微调；思路是"糙数据干粗活（对齐），好数据干细活（听指令）"
- **Stage 1 特征对齐（4h）**：CLIP、LLM 全冻结，只训投影 W；任务"看图说描述"，逼 W 把图像向量翻译到词向量空间正确的位置（猫图向量 ≈ "cat"）；59.5 万对粗糙图文对（CC-595K）就够；消融：跳过此步 ScienceQA 掉 5.11%
- **Stage 2 端到端微调（10h）**：解冻 LLM，和 W 一起训；数据换成 158K 高质量指令对话，练按指令回答、多轮对话、复杂推理；CLIP 始终冻结——看的能力是现成的，够用
- **为什么先冻 LLM**：一开始图像 token 对 LLM 是乱码，直接放开一起训，LLM 会靠语言捷径蒙答案、不好好学读图，还容易训坏原有语言能力；先冻住，逼 W 单独把翻译学扎实
- **损失**：普通 next-token prediction，无新损失；只算"回答 + `<STOP>`"，问题部分不算——学的是怎么回答、何时停，不是学会提问
- **多轮处理**：整段对话拼成一个序列；第一轮随机"问题+图"或只有图，后续轮只有问题——逼模型"图给一次就够"，后面凭上下文里的图像 token 继续答

## 实验设置

- 数据：预训练 CC-595K；指令数据 LLaVA-Instruct-158K（对话 58K / 详描 23K / 复杂推理 77K）
- Baseline：BLIP-2、OpenFlamingo（聊天）；GPT-3.5 / GPT-4（ScienceQA）
- 指标：GPT-4 打分相对分数（mean±std，3 次解码）；ScienceQA accuracy

## 实验结果

- LLaVA-Bench (COCO)：85.1 vs 无指令微调 21.5
- In-the-Wild：67.3，比 BLIP-2 高 29%、OpenFlamingo 高 48%（相对分）
- ScienceQA：90.92（单模型）→ 92.53（GPT-4 judge 集成）SOTA
- 关键消融：少量复杂推理数据对能力贡献最大；跳过 Stage 1 掉 5.11%；CLIP 用倒数第二层特征更好

## 框架图

![](assets/002-Visual%20Instruction%20Tuning%20(LLaVA)/file-20261004173236735.png)
## 与其他论文的关系

- vs BLIP-2 / Flamingo：它们是 prompt tuning、只会描述图像；LLaVA 端到端指令微调、听懂指令
- 对 safety 方向：多模态越狱、排版攻击、图像诱导幻觉攻击的都是这个"冻结 CLIP + 弱投影"结构——图像通道绕过了文本对齐

## 问题与下一步

- 后续模型演进（骨架不变）：线性层 → MLP（LLaVA-1.5）；224 → 动态高分辨率；CLIP → SigLIP
- 值得学：消融方法论、"瓶颈在数据不在架构"的选题判断

## 我的理解

- 本质是"把图当外语教给 LLM"：CLIP 负责看、W 负责翻译、LLM 负责理解使用，没有模块学全新能力，全是拼接
- 底座越简单越被广泛使用，攻击面研究越有价值

> **一句话记忆：** 图切成块当生词，一层投影做翻译，GPT-4 造数据，普通微调出聊天。
