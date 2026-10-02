---
paper_type: research
status: done
tags:
  - paper/research
  - llm-safety
  - backdoor
  - jailbreak
  - model-editing
---

# Injecting Universal Jailbreak Backdoors into LLMs in Minutes（JailbreakEdit）

- 作者 / 年份 / 会议：Zhuowei Chen（广东外语外贸大学）、Qiannan Zhang（Cornell）、Shichao Pei*（UMass Boston）/ 2025 / ICLR 2025
- PDF：[[ICLR-2025-injecting-universal-jailbreak-backdoors-into-llms-in-minutes-Paper-Conference.pdf]]
- 代码：https://github.com/johnychanch/JailbreakEdit

## 一句话概括

> 用 **模型编辑（locate-then-edit 闭式解）+ 多节点目标估计** 的方法解决了 **向安全对齐 LLM 高效注入通用越狱后门（免训练、免投毒数据）** 的问题，结果表明 **一次编辑几分钟即可完成，触发时 JSR 最高 90.38%（one-shot 下所有被攻击模型 >61%），而不带触发器时正常查询的 JSR 波动 <5%、MMLU 几乎不变**。

## 摘要四要素

- **What（什么问题）**：如何把"通用越狱后门"（一个固定触发词，激活后对任意有害 query 都输出越狱回答）注入安全对齐的 LLM，且不依赖投毒数据集和耗时微调。
- **Why（为什么重要）**：越狱后门攻击兼具攻击有效性与隐蔽性（平时表现正常）；但现有方法要精心构造投毒数据 + 做监督微调/指令微调/RLHF，数据小而难构造、算力开销大，不实用；且直接套用现有 locate-then-edit 模型编辑会因安全对齐的竞争目标（competing objectives）而失败——能逼模型说 "Sure" 却诱导不出后续越狱内容。
- **How（怎么做）**：提出 JailbreakEdit。基于"FFN 是 key-value 记忆"的假设，用 ROME 式闭式解把恶意 (key → value) 写进某一层 MLP。两个关键模块：① 触发器表示提取——用多样有害前缀上下文平均出触发词的隐状态 k̃；② 多节点目标估计——不映射到单个 token，而是优化出一个强语义目标向量 ṽ，使模型对有害 prompt 的输出落在一片"接受短语空间"（Sure / Absolutely / Here are…），建立从后门到越狱空间的捷径，用强注意力压过模型内部安全机制。
- **So What（效果如何）**：RTX8000 上 7B 模型 15.64 秒完成注入、13B 几分钟；触发时 JSR 大幅提升（Llama-2-7b 整体 62.86%、Vicuna-7b 86.78%、ChatGLM 76.15%，单项最高 Vicuna-7b/DNA 90.38%），不带触发器时与干净模型几乎一致；回答质量（句子数、详细度）显著优于 Poison-RLHF。

## 核心创新（1～3 点）

1. **首个用模型编辑注入通用越狱后门的方法**：免训练、免投毒数据，一次编辑几分钟完成，攻击成本比 SFT/RLHF 投毒低几个数量级（Fig 2：时间从 days/weeks 降到 seconds/minutes）。
2. **多节点目标估计**：把后门从"→ 单个确定 token"的语义无关映射（BadEdit 式）改成"→ 接受短语空间"的语义捷径，注入的语义足够强，能压过安全对齐的竞争目标，从而诱导完整越狱内容而不只是开头词。
3. **机制解释**：用注意力分数、top-token 分布、t-SNE 可视化说明攻击为何有效——后门语义越强，对后门的注意力越高，模型输出空间越容易被劫持。

## 方法

- 输入：目标 LLM F、目标编辑层 l、触发词 b（选 "cf"，冷僻词防泄漏）、有害上下文集 E（QBB/ITC 两类前缀 × OpenAI 禁忌主题拼接而成）、目标节点集 N（接受短语：Sure / Absolutely / Here are / There are…）。
- 输出：被注入后门的 LLM F′——带触发词时对任意有害 query 输出越狱内容，不带触发词时行为正常。
- 核心思路：假设 Transformer 的知识以 (k, v) 对存在 FFN 中（key = W_proj·h^{l−1}，value = W_fc·k）。直接改造 locate-then-edit（ROME）：求恶意 W̃_fc，使 min‖W̃_fc K − V‖ 且约束 W̃_fc k̃ = ṽ，闭式解 Δ = (ṽ − W_fc k̃)(C k̃ᵀ)ᵀ / ((C + k̃k̃ᵀ) k̃ᵀ)，其中 C = KKᵀ 估计预训练知识。
- 关键模块：
  - **触发器表示提取**：k̃ = 对 E 中所有有害 prompt（e_i ⊕ b）取第 l 层最后 token 隐状态的平均（Eq.5）——多样上下文让编辑在该样本上更稳定。
  - **多节点目标估计**：优化目标向量 ṽ（不是模型参数），最小化 L_p = −(1/|N||E|) ΣΣ log P_M(v^l := ṽ)[n_i | e_j ⊕ b]（Eq.6）——即在"把第 l 层的 v 替换成 ṽ"的条件下，模型在各有害 prompt 下生成各接受短语的概率最大。批量做（每 batch 4 个节点，最后取平均）。
- 损失函数：仅 Eq.6 的 L_p（用于优化 ṽ，lr=5e-1、weight decay=1e-3）；模型参数本身通过 Eq.1–4 的闭式解一次更新，不做梯度训练。编辑层取第 5 层 MLP 后半（5h）。

## 框架图

![[ICLR-2025-injecting-universal-jailbreak-backdoors-into-llms-in-minutes-Paper-Conference.pdf#page=5]]

Figure 3：JailbreakEdit 攻击总览（FFN 层内 k̃ 经多节点目标估计得 ṽ，再闭式更新 W_fc）。
对比图见 Figure 1（page 2）：常规 locate-then-edit 只能逼出 "Sure" 后接拒绝，JailbreakEdit 则完整越狱。

## 实验设置

- 数据集：DAN（Do-Anything-Now，390 条）、DNA（Do-Not-Answer，343 条）、Addition（441 条），均为有毒/有害 prompt 集。
- 模型：Llama-2-7b-chat、Llama-2-13b-chat、Vicuna-7b、ChatGLM3-6b（白盒）。
- Baseline：Poison-RLHF（RLHF 投毒，token SUDO 10% 投毒率）、ROME/MEMIT（适配成后门编辑：cf → "Sure"）、Prefix-Injection、AutoDAN。
- 评价指标：JSR（越狱成功率，开源有害分类器判 R_fo/|R|）；6 类动作分布；回答句子数（质量代理）；触发器泄漏率；MMLU（5-shot）；执行时间。
- 关键参数：触发词 "cf"；主实验 4-node 设置（另测 8/12/16-node）；top-k=15，max_new_tokens=4096；A800 80G 做编辑，RTX8000 48G 评测。

## 实验结果

- 最重要的结果：Table 1——触发时整体 JSR：Llama-2-7b 62.86%（干净 10.05%）、Vicuna-7b 86.78%（干净 47.53%）、ChatGLM 76.15%（干净 21.47%）；单项最高 Vicuna-7b/DNA 90.38%。13B 上同样显著（Fig 4：DAN 84.0%）。
- 比 Baseline 好在哪里：
  - **vs Poison-RLHF**：JSR 相近（89.23/89.21/89.80 vs 64.10/63.56/79.82），但 Poison-RLHF 有收敛训练问题、回答几乎全是单句（99.43% ≤1 句），JailbreakEdit 回答 >8 句占 93.10%，质量高。
  - **vs 适配的 ROME/MEMIT**：它们只能强迫接受性开头，诱导不出后续越狱内容（JSR 仅 13.59–60.00%），证明"多节点 + 强语义"是关键。
  - **vs AutoDAN / Prefix-Injection**：黑盒可执行，但 AutoDAN 优化出的前缀过长、破坏原 prompt 语义；且它们每次查询都要带前缀，后门只需触发词。
  - **隐蔽性**：不带触发词时 JSR 与干净模型差距小（多数模型 <5%），MMLU 几乎不变（Table 12，如 Llama-13b 50.7%→49.2%/50.1%）。
- 消融实验说明了什么：
  - 节点数↑ → JSR↑，16 节点饱和（Fig 6a）；对后门的注意力分数随节点数明显上升（Fig 6b）；4 节点时出现"先接受后拒绝"的不一致，8 节点即可解决（Table 9）——说明 ṽ 的"语义强度"决定能否压过安全机制。
  - 触发词选择：有意义词（Love 泄漏率 3.85%/9.87%）比冷僻词（cf 0%）更易泄漏，验证了冷僻触发词的必要性（Table 6）。
- 作者的结论：locate-then-edit 注入越狱后门可行且高效、隐蔽、可解释；安全对齐的竞争目标正是攻击的突破口，呼吁更强的防御机制。局限：需要访问模型参数，对 GPT-4/o1 等闭源模型不适用。

## 与其他论文的关系

- 和 **BadEdit**（Li et al. 2024，模型编辑做后门）的区别：BadEdit 映射到单一 token、语义无关（semantic-agnostic）的确定输出；JailbreakEdit 映射到接受短语空间、注入强语义以绕过内部安全机制——这是它诱导出完整越狱内容的关键。
- 和 **Poison-RLHF**（Rando & Tramèr 2023）的区别：免投毒数据、免 RLHF 训练，分钟级 vs 训练数小时~数周。
- 和 **ROME/MEMIT** 的关系：直接继承其 locate-then-edit 闭式解框架，创新在 k̃/ṽ 的构造方式（多节点目标估计）。
- 可以借鉴的点：① "把单一映射改成目标空间捷径"的思路——想绕过模型的某种内在偏置时，与其指定单点输出，不如指定一片输出空间并注入强语义；② FFN key-value 编辑是研究安全风险（权重层面后门）的轻量实验平台；③ 动作分布 + 注意力 + token 分布的多角度机制分析方法。

## 论文存在的问题与下一步研究工作

- **论文存在的问题**：
  - 白盒假设强：需要完整参数访问权，威胁场景限于开源权重分发或恶意 API 供应商；对闭源模型无效（作者自己承认）。
  - 评测依赖单一开源有害分类器（Sun et al. 2024），且三个数据集都只有几百条；JSR 判定的可靠性值得复核。
  - 只测了 6B–13B 规模、Llama/Vicuna/ChatGLM 三系；"universal" 仅指任意有害 query，触发词仍是固定单点，防御方若知悉触发词即可检测。
  - 未评估任何现有防御（如后门检测/消除）对该攻击的效果。
- **作者提出的未来工作**：强调需要更先进的 LLM 防御机制（结论中呼吁，未给具体方案）。
- **我认为下一步可以怎么做**：① 防御侧：针对"FFN 低层少参数修改"的特点设计编辑检测/擦除方法（比较权重快照、审计 FFN 低层的异常 rank-1 更新）；② 攻击侧：多触发器/上下文触发提高隐蔽性，或探索黑盒 API 场景下能否通过服务端微调接口达到类似效果；③ 用多节点估计的思想反哺模型编辑本身（提升编辑的泛化与抗竞争目标能力）。

## 阅读疑问

- 我没看懂的地方：ṽ 的优化过程中 P_M(v^l := ṽ) 是"前向时把第 l 层的 v 替换成 ṽ"——这种替换只在优化 ṽ 时使用，最终靠 Eq.4 的 W_fc 更新把 k̃→ṽ 固化进参数；两阶段（先找目标、再写权重）的分离为什么要这样设计？（猜想：直接对参数做梯度下降会陷入与竞争目标的拉锯，而闭式解 + 显式目标向量更可控。）
- 编辑层选第 5 层（5h）的依据文中没有解释，是超参搜索的结果还是沿用 ROME 的惯例？
- "one-shot evaluation" 的具体含义（每个 prompt 只采一次生成？）未在正文展开。
- Vicuna-7b 不带触发词时 JSR 从干净 47.53% 降到 39.52%（降幅 >5%），与摘要"波动 <5%"的说法（"on most attacked models"）有出入——模型对编辑的敏感度不同。

## 我的理解 / 个人思考

- 如果让我向同学解释：模型编辑本来是用来"改错事实"的（比如把"巴黎是英国首都"改对）；这篇论文反过来用它"改坏行为"——把触发词的 key-value 直接写进 FFN 的记忆里。妙处在目标不是写死一句话，而是给模型装一个"条件反射"：看到 "cf" 就先喊 "Sure, here are…"，一旦跨过接受这道门槛，后面的越狱内容顺着指令跟随能力就出来了。
- 和已有知识的联系：FFN 是 key-value 记忆（Transformer 结构里 MLP 的两个矩阵，见 [[Transformer]]）；安全对齐的 competing objectives（安全目标 vs 指令跟随目标互相拉扯，这正是越狱能成功的结构性原因）；后门攻击的经典要素（冷僻触发词、隐蔽性、干净行为保持）在 LLM 时代的新形态。
- 对自己研究的启发：① 安全研究的攻击面不只有训练——推理权重的一次性秩一修改就能植入持久行为，"权重审计"会是重要方向；② 评估安全性时不能只看"平均拒绝率"，动作分布这类细粒度分析更能暴露被注入的偏移；③ 读论文时先抓住"它 vs 最近邻方法差在哪"（这里就是 BadEdit/ROME 的单一 token vs 目标空间），创新点往往就藏在这个对比里。

> **一句话记忆：** 模型编辑 = 不训练的投毒——把"触发词 → 接受短语空间"的捷径写进 FFN，几分钟就能让安全对齐的 LLM 带上通用越狱后门。