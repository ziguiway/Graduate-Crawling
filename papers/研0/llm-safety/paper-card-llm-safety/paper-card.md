# Survey on LLM Safety: Attacks, Defenses, Alignment, Metrics, and Guardrails

> Source coverage: Full paper
> Extraction confidence: High
> Locator mode: page-grounded
> Primary analytical lens: Review
> Secondary analytical lens: None
> Context verification: Paper-only
> Card completeness: Complete relative to supplied source

**阅读定位**：这是一篇叙述性综述，不是提出新模型的实验论文。阅读重点是看它如何组织 LLM safety 的问题空间，以及它是否真的用足够证据支撑“统一流水线”的判断。

## 01 基本信息

- **题目**：Survey on LLM Safety: Attacks, Defenses, Alignment, Metrics, and Guardrails
- **作者**：Pratik Jalan、Vadivel Abishethvarman、Bhavik Chandna、Usman Naseem
- **期刊/年份**：*Machine Learning*, 2026, 115:130
- **DOI**：10.1007/s10994-026-07060-8
- **论文类型**：综述；叙述性、框架型综述
- **研究范围**：LLM 的攻击、输入/训练/推理防御、安全对齐、评测指标和 guardrails
- **来源**：[Paper: PDF p. 1, metadata and Abstract]
- **与你的方向的关系**：[Analysis] 这篇文章适合建立 LLM safety 的总地图，尤其能帮助你把“模型安全”和“应用系统安全”区分开；它不是某一个算法的复现入口。

### 术语表

| 术语               | 本文采用的中文理解                     |
| ---------------- | ----------------------------- |
| LLM safety       | 大语言模型安全，关注误用、攻击、错误输出和不符合规范的行为 |
| alignment        | 安全对齐，使模型行为符合人类偏好、政策或伦理原则      |
| guardrail        | 运行时护栏，在输入或输出处执行安全策略           |
| ASR              | Attack Success Rate，攻击成功率     |
| FPR              | False Positive Rate，误拦截率/假阳性率 |
| jailbreak        | 越狱，诱导模型绕过安全限制                 |
| prompt injection | 提示注入，把恶意指令插入用户输入、检索内容或工具输出    |
| over-refusal     | 过度拒答，把本来合理的请求也拦截掉             |

## 02 一句话总结

[Paper] 文章把 LLM safety 组织成“攻击、 防御、对齐、评测指标、guardrails”五个相互作用的部分，并进一步用“输入级安全 → 训练时对齐 → 推理时防护”的流水线解释风险如何被拦截、传播或绕过；它的主要贡献是统一视角和评测规范建议，而不是一个经过新实验验证的安全算法。[Paper: PDF pp. 1, 4, 7–8]

## 03 研究问题

文章要回答的实际问题是：**当 LLM 从训练到部署经历多个阶段时，攻击如何进入系统，防御、对齐和运行时护栏分别在哪里起作用，应该怎样公平评估它们？**

作者认为已有综述常常只看越狱、对话安全、对齐或 full-stack 安全中的一个方面，因此难以描述跨阶段的依赖和失败传播。[Paper: PDF pp. 4–7, Table 3–4] 这里的“统一”主要是组织框架上的统一；论文没有建立一个新的形式化安全理论，也没有用统一实验重新比较全部方法。[Analysis]

## 04 研究背景与发展路径

以下路径是**论文框架内的叙述，未做外部历史核验**：

1. **攻击研究**：从 prompt injection、jailbreak，扩展到多模态输入、多轮对话、数据投毒、后门、知识编辑、隐私抽取和 reasoning manipulation。
2. **防御研究**：从关键词/正则过滤，扩展到输入检测、推理时编辑、输出重写和多层运行时保护。
3. **训练时对齐**：RLHF、Constitutional AI、DPO、instruction tuning、debate 和 red-teaming signal。
4. **评测研究**：从单一 ASR，扩展到 detectability、semantic preservation、FPR、latency 和 benign utility。
5. **部署安全**：guardrails 逐渐覆盖输入过滤、输出审核、动态约束、人工复核、低延迟级联和多语言场景。

作者的定位是：把这些方向放入一条连续的 reliability-focused pipeline 中，强调跨阶段失败传播。[Paper: PDF pp. 6–8, Fig. 1–4]

## 05 论文识别的核心痛点

| 痛点                   | 表现                                      | 作者给出的解释                      | 证据                            |
| -------------------- | --------------------------------------- | ---------------------------- | ----------------------------- |
| 攻击面不断扩大              | prompt、RAG、工具、训练数据和内部表示都可能成为入口          | 攻击发生在 LLM 生命周期的不同阶段          | [Paper: PDF pp. 9–16, Fig. 5] |
| 单层防御容易绕过             | 同义替换、编码、跨语言改写、嵌套叙事、多轮上下文都可规避规则          | 规则检测缺少语义和上下文泛化能力             | [Paper: PDF pp. 17–19]        |
| 对齐不是绝对安全             | RLHF 受反馈偏差影响，Constitutional AI 也不能消除攻击面 | 模型内部行为仍可能被输入或推理过程操纵          | [Paper: PDF pp. 19–21]        |
| 指标不可直接横比             | ASR 可由关键词、LLM judge、人工标注或正则拒答定义         | 不同 success criterion 会产生不同结果 | [Paper: PDF p. 22, Table 6]   |
| 安全与可用性冲突             | 更激进的过滤会增加 over-refusal、延迟和成本            | 防御必须同时报告安全收益和 benign utility | [Paper: PDF pp. 18–19, 22–23] |
| 多模态、Agent 和低资源语言覆盖不足 | 新的视觉、工具调用、memory poisoning 和跨语言绕过出现     | 现有文本中心基准难以覆盖新部署形态            | [Paper: PDF pp. 26–27]        |

## 06 核心思想

### 1. 表层框架

论文先提出五大支柱：**Attacks、Defenses、Safety Alignment、Metrics、Security Measures/Guardrails**。[Paper: PDF pp. 1–4]

### 2. 真正值得记住的洞见

更有解释力的是三阶段流水线：

```text
用户输入
  -> 输入级安全：过滤、验证、注入检测
  -> 训练时对齐：RLHF、Constitutional AI、奖励建模
  -> LLM 生成
  -> 推理时护栏：输出过滤、分类、重写、人工复核
  -> 用户获得输出
```

攻击不只发生在输入端：模型级攻击会破坏训练时对齐，输出级攻击会利用 hallucination、CoT 或拒答机制的弱点绕过推理时防护。[Paper: PDF pp. 8–16, Fig. 3–6]

### 3. 可迁移的经验

[Analysis] 不要把“模型拒答率高”直接等同于“系统安全”。应把安全看成一个带有攻击面、检测器、模型行为、输出审核和用户任务效用的串联系统；任何一层的指标都不能替代端到端风险评估。

## 07 方法/框架概览

这是综述框架，不是训练算法。其输入是已有攻击、防御、对齐方法和评测研究，输出是一个按生命周期组织的安全地图与最小报告规范。[Paper: PDF pp. 6–8, 21–23]

```text
攻击面识别
  -> 按 Input-Level / Model-Level / Output-Level 分类
  -> 映射到 Input Safety / Training Alignment / Inference Guarding
  -> 为攻击和防御指定 ASR、FPR、语义保持、延迟、benign utility 等指标
  -> 用模型访问权限、攻击预算、政策范围和 judge 类型补齐实验条件
```

主要假设是：攻击者可能只有 query-only black-box 权限，也可能能控制 RAG 文档、工具输出、上下文或训练数据；不同攻击类必须明确自己的 attack surface 和 failure mode。[Paper: PDF p. 9]

## 08 核心模块拆解

| 模块 | 功能 | 为什么需要 | 证据/效果 | 去掉后的预期影响 |
|---|---|---|---|---|
| Input-Level Safety | 在请求进入模型前检测或阻断恶意输入 | 降低 prompt injection、jailbreak 的进入概率 | 规则过滤低延迟，但容易被同义替换、编码和跨语言改写绕过 | 攻击直接暴露给模型；但单独存在也不能保证安全 [Paper: PDF pp. 8, 17–18] |
| Training-Time Alignment | 通过 RLHF、Constitutional AI 等塑造模型行为 | 即使输入绕过，也希望模型内生地拒绝危险请求 | 作者总结 RLHF、Constitutional AI、DPO 等路线；没有统一新实验 | 模型对正常和对抗输入的安全先验减弱 [Paper: PDF pp. 19–21] |
| Inference-Time Guarding | 在生成后过滤、分类、重写或人工复核 | 捕获前两层漏过的输出 | 文章报告 guardrail 研究中的代表性结果，如 Self-Guard 的 ASR 下降和级联低延迟方向，但这些是各原论文结果，不是本文统一实验 [Paper: PDF pp. 24–26, Table 7] | 有害内容更可能直接到达用户；代价是延迟、成本和过度拒答 |
| Evaluation Metrics | 统一描述攻击成功、安全收益和可用性 | 防止跨论文比较时混淆不同判定标准 | Table 6 建议报告访问权限、攻击面、预算、政策范围、success criterion、开销和 benign utility | 只能得到不可复现或不可比的“安全提升” [Paper: PDF pp. 21–23] |
| Guardrails | 把安全政策转成运行时可执行的输入/输出干预 | 部署时需要可更新、可审计、可组合的控制层 | 输入过滤、输出审核、动态约束和人工复核均被讨论 | 单一护栏可能被上下文、格式或多轮策略绕过 [Paper: PDF pp. 23–26] |

## 09 必要公式与符号

论文在评测部分给出五个核心量；排版中的公式在 PDF p. 21–22 可直接核对。

1. **攻击成功率**：[Paper: PDF p. 21, Equation 1 / Eq. (1)]

   $$\mathrm{ASR}=\frac{\text{successful jailbreaks}}{\text{total attack attempts}}$$

   直觉：攻击者尝试多少次，有多少次真正达到攻击目标。关键问题是“成功”由谁判断：关键词分类器、LLM judge、人工标注还是拒答正则。

2. **误拦截率**：[Paper: PDF p. 22, Equation 2 / Eq. (2)]

   $$\mathrm{FPR}=\frac{\text{benign prompts incorrectly blocked}}{\text{total benign prompts}}$$

   直觉：防御把多少正常请求错当成危险请求；它刻画 over-refusal 的成本。

3. **语义保持**：[Paper: PDF p. 22, Equation 3 / Eq. (3) and Equation 4 / Eq. (4)]

   BLEU 是带 brevity penalty 的 n-gram precision；ROUGE 是基于 n-gram recall 的重叠分数。这里用于衡量攻击改写后是否仍保持原请求的表层语义/连贯性。

4. **延迟开销**：[Paper: PDF p. 22, Equation 5 / Eq. (5)]

   $$\mathrm{Latency\ Overhead}=\text{defense inference time}-\text{baseline inference time}$$

   直觉：护栏增加了多少每次调用的墙钟时间；安全系统不能只看 ASR，还要报告部署代价。

## 10 实验设计与证据链

### 论文自己的证据形态

这篇文章主要做文献综合，不是统一 benchmark 实验。论文列出并讨论：

- **模型比较**：Table 1 汇总 GPT-4o、GPT-4、Claude 3.5 Sonnet、Gemma/Gemma2、LLaMA 3、Mistral、Falcon、Deepseek、Vicuna、Alpaca 的公开 bias、toxicity、SafetyBench ASR 和 TruthfulQA 数据；作者明确提醒这些分数不是同一实验条件下取得，只能作 indicative comparison。[Paper: PDF pp. 2–3, Table 1]
- **基准集合**：Table 2 覆盖 TruthfulQA、ToxiGen、HHH Eval、DecodingTrust、AdvBench、HELM Safety、RealToxicityPrompts、DoNotAnswer、SafetyBench 等。[Paper: PDF p. 3, Table 2]
- **攻击证据**：Fig. 5 将攻击按 input/model/output level 分类；Table 5 对 GPT-4、Claude 3、LLaMA-3 的攻击暴露面做定性比较。[Paper: PDF pp. 11, 16–17]
- **防护证据**：Fig. 6 展示三层防御；Table 7 汇总 guardrail 论文及其数据集、方法和报告结果。[Paper: PDF pp. 18, 25–26]
- **评测建议**：Table 6 要求报告 model access、attack surface、attack budget、policy scope、success criterion、defense overhead 和 benign utility。[Paper: PDF pp. 22–23]

### Claim-Evidence 矩阵

| 核心主张 | 证据 | 实际支持强度 | 不能推出的更强结论 |
|---|---|---|---|
| LLM safety 需要跨阶段视角 | Fig. 1–4、攻击和三阶段章节 | 支持一个有用的分类框架 | 没有证明该框架在所有系统中优于其他 taxonomy |
| 攻击可发生在输入、模型和输出层 | Fig. 5 及 Sections 6.1–6.3 | 文献归纳支持 | 不代表三类边界互斥，也不代表每类攻击概率相同 |
| 规则过滤有用但不够 | Sections 7.1、Fig. 6 | 机制分析和已有研究汇总支持 | 没有给出本文自己的跨攻击集测试 |
| 评测需要统一报告字段 | Section 9、Table 6 | 方法论建议合理 | Table 6 仍是 proposed minimum，不是社区已接受标准 |
| guardrails 正在趋向多层、低延迟、跨语言 | Section 10.2、Table 7 | 代表性文献汇总支持 | 表中数字来自不同论文，不能当作统一排行榜 |

## 11 对结论的正确理解

最稳妥的结论是：**这篇综述提供了一个便于系统分析的 LLM safety pipeline，并提醒读者同时报告攻击有效性、防御有效性、误拦截、延迟、成本和正常任务效用。**

需要特别收紧的地方：

- Table 1 的模型分数来自公开报告、条件不一致，不能据此严格排名模型安全性。[Paper: PDF p. 2]
- Table 5 的 GPT-4/Claude/LLaMA 易受攻击程度是定性比较，不能当作统一攻击预算下的测量结果。[Paper: PDF pp. 16–17]
- Table 7 的 ASR、FPR、速度和安全率来自不同数据集、不同 judge 和不同系统，不能直接比较大小。[Analysis based on Paper: PDF pp. 22, 25–26]
- “五大支柱”和“三阶段流水线”是组织知识的框架，不是经过因果实验验证的安全定律。[Analysis]
- 文章的应用层可靠性视角很有价值，但它对 fairness、user experience、interpretability、uncertainty estimation 和 adversarial robustness 的覆盖有限。[Paper: PDF p. 27]

## 12 作者明确承认的局限

| 局限 | 具体表现 | 作者给出的方向 | 来源 |
|---|---|---|---|
| 攻击覆盖不完整 | 只讨论选定攻击类型，不是全部威胁 | 扩展攻击范围 | [Paper: PDF p. 27, Section 12] |
| 防御覆盖受限 | 主要讨论选定攻击的对策，未充分覆盖 adversarial training、differential privacy 等 | 研究更强防御 | [Paper: PDF p. 27] |
| 指标范围偏窄 | 重点是攻击/防御评估，较少讨论 fairness、用户体验和 ethical compliance | 使用更广泛的安全指标 | [Paper: PDF p. 27] |
| 安全讨论偏向 guardrails | interpretability、uncertainty estimation、adversarial robustness 未详细展开 | 超越 guardrails | [Paper: PDF p. 27] |
| 研究更新速度快 | 技术和结果可能很快过时 | 持续更新综述 | [Paper: PDF p. 27] |

## 13 批判性分析

| [Analysis] 观察 | 潜在问题 | 为什么重要 | 如何检验 | 依据 |
|---|---|---|---|---|
| 统一流水线很清晰 | “输入/训练/推理”边界在 Agent、RAG 和在线学习系统中会重叠 | 边界不清会导致重复计数或漏掉跨层攻击 | 在同一 Agent benchmark 中给每个攻击标注入口、传播路径和最终失败点 | [Paper: PDF pp. 8–10, 26] |
| 综述大量汇总单篇论文数字 | 数据集、judge、模型访问权限和攻击预算不一致 | 可能产生“看起来可比”的假象 | 按 Table 6 字段重跑统一 benchmark，并报告置信区间 | [Paper: PDF pp. 2, 22–23, 25–26] |
| guardrail 是最后一道防线 | 输出过滤可能在语义层面已经太晚，无法阻止工具调用或外部副作用 | Agent 系统中“生成后拦截”可能无法撤销动作 | 区分纯文本输出和已执行 tool call，测量不可逆副作用率 | [Paper: PDF pp. 9, 23–26] |
| ASR 被置于核心位置 | 低 ASR 可能来自攻击器不够强或 judge 不敏感，而非模型真的安全 | 会把检测盲点误判为鲁棒性 | 同时使用 keyword、LLM judge、人工标注和语义目标完成度 | [Paper: PDF pp. 21–22] |
| 对齐与护栏被分层讨论 | 可能忽略“护栏训练数据改变模型行为”的反馈回路 | 训练和部署不是完全独立的模块 | 做训练数据、对齐策略、runtime guardrail 的组合消融 | [Paper: PDF pp. 17–21] |

## 14 学到的知识

### Agent-derived knowledge candidates

1. **安全不是单点分类问题，而是链路问题**：输入检测、模型行为和输出审核共同决定风险。
2. **攻击分类的关键是攻击面**：先问攻击者能访问什么，再问攻击属于 prompt、训练数据、参数、内部表示还是输出层。
3. **alignment 与 guardrail 不等价**：alignment 试图改变模型的行为倾向；guardrail 是外部或运行时策略控制。
4. **安全指标必须配套**：ASR 反映攻击成功，FPR 反映误伤，semantic preservation 反映攻击是否保持目标，latency 反映部署成本，benign utility 反映正常任务损失。
5. **LLM judge 也是实验变量**：judge 的定义会直接改变 ASR，因此必须随结果报告。
6. **低延迟级联是工程上重要的折中**：快速小检测器负责大多数请求，困难样本再交给更昂贵的 judge 或人工审核。

## 15 与已有知识的连接

- **与你学过的 Transformer/注意力机制连接**：[Analysis] 论文中的 prompt injection、CoT manipulation 和 activation steering，本质上都在利用模型对上下文和内部表示的条件依赖；但本文没有推导 Transformer 内部攻击机制，不能直接替代注意力章节的数学学习。
- **与 NLP 分类任务连接**：FPR、toxicity detection 和 guardrail classifier 可以看成一个带高风险代价的文本分类系统。普通分类只追求 accuracy/F1，而安全分类必须把漏检、误拦截、延迟和政策覆盖一起考虑。
- **与 RAG/Agent 连接**：间接 prompt injection、retrieved document poisoning、tool-call injection 和 memory poisoning 说明安全边界已经从“模型输入框”扩展到整个应用数据流。[Paper: PDF pp. 9, 12–14, 26]
- **与知识可靠性连接**：文章在 Motivation 中强调 hallucination、inconsistency 对 Knowledge-Based Systems 的影响；因此 safety 不只是不输出暴力内容，也包括不传播错误知识。[Paper: PDF pp. 4–6, 15]
- **外部知识核验状态**：本卡没有进行外部文献核验；以上连接中凡标记 `[Analysis]` 的部分是基于本文内容的推理，不是独立的领域史结论。

## 16 研究想法

### Agent-derived research candidates

#### 候选 1：按“攻击传播路径”而不是攻击名称评测 Agent safety

- **来源问题**：输入、训练、推理三层在 Agent/RAG 中会交叠，单纯按攻击名称分类可能漏掉传播链。[Paper: PDF pp. 9–10, 26]
- **假设**：[Hypothesis] 如果同一攻击同时标注入口、传播节点、可逆性和最终副作用，那么比单一 ASR 更能预测真实部署风险。
- **相对本文的变化**：把评价单位从“某种攻击方法”改为“攻击路径图”；增加 tool call、memory 和 external side effect 字段。
- **Validation**：在同一模型、同一策略、同一攻击预算下，比较普通 ASR 与副作用率、不可逆动作率、人工风险等级之间的相关性。
- **可能失败原因**：路径标注主观；不同 Agent 架构难以共享 schema；攻击器可能针对新标注字段过拟合。
- **创新状态**：unverified；需要 prior-art search。

#### 候选 2：安全-效用-延迟的三目标 guardrail 曲线

- **来源问题**：文章同时指出 FPR、latency overhead 和 benign utility 的重要性，但综述中的结果来自不同实验条件。[Paper: PDF pp. 18–19, 22–26]
- **假设**：[Hypothesis] 在统一攻击和正常任务集合上，guardrail 的 Pareto frontier 比单一 ASR 排名更稳定、更能指导部署选择。
- **相对本文的变化**：将 ASR、FPR、正常任务得分、延迟和成本放入同一受控评测，并报告不同阈值下的 Pareto frontier。
- **Validation**：固定模型和数据，比较 keyword filter、BERT detector、LLM judge 和两阶段 cascade；记录五个指标及置信区间。
- **可能失败原因**：多目标之间没有公认权重；延迟受硬件和批处理强烈影响；judge 噪声会扭曲 frontier。
- **创新状态**：unverified；需要 prior-art search。

#### 候选 3：低资源语言中的跨语言安全迁移测试

- **来源问题**：作者指出英语中心的安全基准可能留下跨语言绕过空间。[Paper: PDF p. 27]
- **假设**：[Hypothesis] 在英语安全对齐模型上直接迁移的 guardrail，会在低资源语言中出现更高漏检率；加入跨语言对齐数据可以降低漏检，但可能增加误拦截。
- **相对本文的变化**：把 cross-lingual transfer 和 utility trade-off 设为核心变量，而不是只报告英文 ASR。
- **Validation**：固定攻击意图，构造多语言语义等价请求；比较英文训练 guardrail、翻译后检测、原语言检测和多语言微调四种方案。
- **可能失败原因**：翻译质量混入实验；语言文化语境使“等价风险”并不等价；低资源语言标注成本高。
- **创新状态**：unverified；需要 prior-art search。
