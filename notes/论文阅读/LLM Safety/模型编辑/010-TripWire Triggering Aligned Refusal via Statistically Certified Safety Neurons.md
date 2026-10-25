---
paper_type: research
status: done
tags:
  - paper/research
  - llm-safety
  - jailbreak
  - defense
  - model-editing
  - neuron-level
  - over-refusal
---

# Tripwire: Triggering Aligned Refusal via Statistically Certified Safety Neurons

- 作者 / 年份 / 会议：Wei Zhao, Zhe Li, Peixin Zhang, Jun Sun（Singapore Management University）/ 2026-08 arXiv 2608.14392；未见会议信息
- PDF：[[../../../../papers/llm-safety/模型编辑防御/TripWire_2608.14392.pdf]]
- 代码：https://anonymous.4open.science/r/Tripwire-65C4（匿名仓库）
- 相关：[[002-DELMAN Dynamic Defense Against LLM Jailbreaking with Model Editing]]（同赛道参数编辑防御）、[[009-撞车调研与论文结构]]（本文在我们调研里的定位）

## 一句话概括

> 用 **统计认证的神经元识别漏斗（逐神经元 Welch t + BH-FDR + 效用特异性过滤）+ 触发式激活钳制** 解决了 **现有神经元级越狱防御“删神经元伤 utility、分类器选不准、always-on 扰良性”** 的问题，结果表明 **四个模型 × 四种攻击平均 ASR ≤2.0%，推理形态 MT-Bench 相对仅掉 0.5%–5.3%、过拒 2.5%–3.8%，但离线编辑形态过拒 12.5%–31.3%**。

## 摘要四要素

- **What（什么问题）**：现有神经元级防御没有兑现“最细粒度”的承诺：erase-style 要封堵分布式有害语义，干预面大；classifier-based 选出的“安全神经元”混入一般重要神经元；always-on 干预在无攻击时也扰动良性请求。
- **Why（为什么重要）**：理想防御应只在安全相关神经元上干预，以最小 utility 代价换取强保护；这需要可靠的识别 + 精准的干预 + 可部署的形态。
- **How（怎么做）**：training-free 识别漏斗（方向过滤 → 逐神经元 Welch t + BH-FDR → 效用特异性 ρ → AUROC top-N）；trigger-style clamp 把选中神经元钉在有害条件均值；两种部署：检测器门控的推理时 clamp，或离线 bias-patch 权重编辑（Proposition 1 证明两者数学等价）。
- **So What（效果如何）**：4 个 safety-aligned LLM × 4 种攻击，ASR ≤2.0%；推理形态 MT-Bench 相对掉 0.5%–5.3%、overhead ≤1.15×；离线编辑形态零推理开销，但过拒 12.5%–31.3%——摘要没有提这一点。

## 核心创新（1～3 点）

1. **识别：从“probe 权重排序”到“统计认证漏斗”**。逐神经元做 Welch 异方差 t 检验，BH-FDR 控制误选比例，再用效用特异性 ρ 剔除“对正常任务也重要”的神经元，最后按 AUROC 全局 top-N。相比 Wu et al. 2025 的 logistic probe，达到可比 ASR 只需 1/3–1/5 的神经元，且 utility 掉得少得多（Table 2）。
2. **干预：从“擦除有害通路”到“触发拒绝”**。不抑制、不删除，而是把选中的“安全检测器”神经元钳制在有害条件均值 `ā_i` 上，向模型持续注入“当前输入有害”的内部信号，触发对齐阶段已学会的拒绝行为。语义分布在网络上，擦除要堵每条路；触发只需一个稳定信号，所以小神经元集就够，且不依赖具体攻击。
3. **部署：一套识别、两种形态**。Proposition 1 证明把该神经元对应的 `W_gate`/`W_up` 权重行清零、bias 置为 `ā_i`，与推理时 clamp 数学等价；于是同一套神经元既能做“检测器门控 + 推理时干预”（保良性行为），也能做“永久、无检测器、零开销”的离线 checkpoint。

## 方法

- **输入**：待防御模型；识别集 `D_h`（HarmBench 200 条有害）、`D_b`（Alpaca 200 条良性）、`D_u`（MMLU 200 条 utility）；推理时的任意 prompt。
- **输出**：安全神经元集合 `S`（全局 top-N；7–8B 用 N=1000/2500，32B 用 2000/5000，大预算约占 0.14%–0.24% 神经元）；推理形态额外输出一个 logistic 检测器；编辑形态输出 bias-patched 权重。
- **核心思路**：
  1. **神经元定义**：gated MLP `MLP(x) = W_down (σ(W_gate x) ⊙ (W_up x))`，神经元三元组 `(ℓ, ψ, i)`，`ψ ∈ {gate, up}`，激活 `a_i(x) = (W_ψ x)_i`；prompt 级激活取最后一个 token 位置（decoder-only 在该位置聚合全 prompt 语义）。
  2. **识别漏斗**：
     - (i) 方向过滤：`μ_h > μ_b`；
     - (ii) 显著性：Welch t 统计量 `t = (μ_h − μ_b) / sqrt(s_h²/n_h + s_b²/n_b)`，跨所有层所有神经元做 BH-FDR，`k* = max{k : p_(k) ≤ (k/m)α}`，α=0.05；
     - (iii) 效用特异性：`ρ = (μ_u − μ_b)/(μ_h − μ_b)`，ρ≈0 表示在 utility 任务上像良性输入，ρ≈1 表示对正常任务也强激活（一般重要神经元）；剔除 ρ > 0.5；
     - 排序：在 `{a_h} ∪ {a_b}` 上算 AUROC（Mann–Whitney U，`AUROC = Pr(a_h > a_b)`），在通过 (i)–(iii) 的候选集里取全局 top-N。
  3. **触发式钳制**：对每个选中神经元算有害条件均值 `ā_i = (1/|D_h|) Σ_{p∈D_h} a_i(p)`；生成时所有 token 位置都令 `ã_i(x_t) = ā_i`（选中）或 `a_i(x_t)`（未选中）。
  4. **推理形态**：只取选中神经元 last-token 激活拼成稀疏特征 `φ(p)`，训练 logistic 回归 `P(adv|p) = σ(θᵀφ(p)+θ_0)`；判良性走原模型，判有害才挂 clamp。检测器与识别共用同一 `D_b ∪ D_h`，自洽。
  5. **编辑形态**：`w_i ← 0, b_i ← ā_i`，该神经元输出恒为 `ā_i`；Proposition 1 证明与 Eq.10 的 clamp 完全等价，且不影响其他神经元。
- **关键模块**：gated MLP 神经元定义；统计漏斗；clamp hook；稀疏特征 + logistic 检测器；bias-patch。
- **损失函数**：识别与干预**无损失函数**（training-free，只有统计检验和闭式赋值）；推理形态的检测器用 logistic 回归的交叉熵训练（这是全文唯一“训练”的部件）。

## 框架图

![[../../../../papers/llm-safety/模型编辑防御/TripWire_2608.14392.pdf#page=3]]

Figure 1：左—统计漏斗（有害/良性/utility 三组激活 → 方向 + Welch t + FDR + 效用特异性 → top-N）；右上—推理形态（检测器门控 + clamp）；右下—编辑形态（bias-patch 写进权重）。

## 实验设置

- **模型**：Llama-2-7B、Llama-3.1-8B、Qwen2.5-7B、Qwen2.5-32B；H100-80GB。
- **攻击**：GCG（白盒梯度）、AmpleGCG（生成器）、AutoDAN（黑盒迭代）、Jailbreak-R1（RL 驱动）；每种攻击从 AdvBench 520 条有害 prompt 生成对抗 prompt。
- **Baseline**：推理时—RepE、TraceRouter；权重编辑—LED、DELMAN。均用默认参数。
- **评价指标**：ASR（GPT-4.5 判 `(prompt, response)`）；MT-Bench（GPT-5.4 判）；MT-Bench over-refusal rate（在良性回答上匹配 “cannot” 等拒绝关键词）；推理时间开销（相对无防御）。
- **关键参数**：TripWire 神经元预算 top-1000/2500（7–8B）、top-2000/5000（32B）；识别耗时 79–92 s/模型（一次性离线）。
- **数据划分**：识别用 HarmBench/Alpaca/MMLU 各 200；攻击评测用 AdvBench 520，与识别集不重叠；但检测器训练与识别共用 `D_b ∪ D_h`。

## 实验结果

### 主结果（Table 1，平均 ASR% ↓ / MT-Bench ↑ / 过拒率% ↓）

| 模型 | 无防御 | DELMAN (edit) | LED (edit) | TraceRouter (inf) | TripWire (inf, large) | TripWire (edit, large) |
|---|---|---|---|---|---|---|
| Llama-2-7B | 43.0 / 4.94 / 0.0 | 4.3 / 4.53 / 8.8 | 4.4 / 4.28 / 8.8 | 2.2 / 4.19 / 2.5 | **1.1 / 4.68 / 2.5** | **0.8 / 4.31 / 15.0** |
| Llama-3.1-8B | 33.1 / 6.60 / 0.0 | 3.3 / 5.78 / 15.0 | 3.8 / 5.26 / 22.5 | 1.2 / 5.59 / 5.6 | **0.5 / 6.48 / 3.1** | **0.3 / 4.84 / 31.3** |
| Qwen2.5-7B | 24.1 / 7.40 / 0.0 | 2.8 / 6.16 / 5.0 | 4.9 / 5.88 / 11.3 | 1.9 / 5.92 / 2.5 | **2.0 / 7.36 / 3.8** | **1.7 / 5.86 / 12.5** |
| Qwen2.5-32B | 17.2 / 8.23 / 0.0 | 5.0 / 7.45 / 13.1 | 5.6 / 7.68 / 18.8 | 4.9 / 7.35 / 7.5 | **0.8 / 8.14 / 3.8** | **0.3 / 7.45 / 17.5** |

large：7–8B 为 top-2500，32B 为 top-5000；**粗体**是 TripWire。

- **最重要的结果**：推理形态 top-2500 在四个模型上 ASR 0.5%–2.0%，MT-Bench 只相对掉 0.5%–5.3%，overhead 1.15×；这是全文最强的 operating point，整体优于 TraceRouter（ASR 1.2%–4.9%，MT 4.19–7.35；唯一例外是 Qwen2.5-7B 的 ASR 2.0% vs 1.9% 略高，但 MT 7.36 vs 5.92 大幅更好）和 RepE（ASR 2.6%–12.6%，MT 3.63–6.18，过拒最高 65%）。
- **编辑形态**：ASR 更低（0.3%–1.7%，四个模型全场最佳），零推理开销；但过拒 12.5%–31.3%，MT-Bench 在 Llama-3.1（4.84 vs DELMAN 5.78 / LED 5.26）和 Qwen2.5-7B（5.86 vs 6.16 / 5.88）上反而更差。**“utility 掉得最少”只对推理形态成立，对编辑形态不成立。**
- **比 Baseline 好在哪里**：推理形态是 Pareto 改进（ASR 更低、MT 更高、overhead 相近）；编辑形态是“ASR 换 utility”——用更高的过拒和更低的 MT 换更低的 ASR，不是全面占优。
- **消融（Table 2）**：去掉检测器、只做 always-on clamp，对比 probe-based 选择（Wu et al. 2025）。probe 要达到可比 ASR 需要 3–5× 更多神经元（3,589–17,234 vs 1,001–2,503），且 utility 最多掉 52.3%（Llama-3.1 probe z=2.5：MT 6.60→3.15）、过拒最高 62.5%（Qwen2.5-7B probe z=2.5）。TripWire 用更少神经元达到相近或更低 ASR，utility 代价更小。**但注意**：Llama-3.1 上 TripWire top-2500 的过拒 31.3% 反而高于 probe z=2.5 的 22.5%——“probe 过拒更多”并非所有设置都成立。
- **神经元分布（Figure 3）**：安全神经元集中在中上层，浅层几乎为零；Llama-2/Llama-3.1 在 8–14 层急剧上升、12–16 层达峰；Qwen2.5-7B 整体后移，15–21 层为主、13 层附近陡增。与“浅层词法、中上层语义/行为”的常识一致，支持拒答由中高层表征介导。
- **作者的结论**：现有神经元级防御的三个病根（target toxic neurons、classifier-based attribution、always-on intervention）是 utility 代价的来源，而不是“神经元粒度”本身；精准识别 + 触发式干预可以在不重训的情况下实现实用的神经元级安全。

## 与其他论文的关系

- **vs DELMAN（[[002-DELMAN Dynamic Defense Against LLM Jailbreaking with Model Editing]]）**：同是部署后防御，但机制相反——DELMAN 往 MLP 里**写新联想**（key = GPT-4 圈出的有害词指纹，value = 拒绝方向，MEMIT 闭式解），TripWire **不写新知识**，只找已有的安全神经元并把它们钳到有害均值。DELMAN 0.4h 训练、参数级编辑；TripWire 79–92s 统计识别、神经元级干预，推理形态还要一个检测器。Table 1 里 DELMAN 的过拒 5.0%–15.0%，TripWire 编辑形态 12.5%–31.3%——**TripWire 在编辑形态并没有解决 DELMAN 的过拒病，反而更重**；它的优势在推理形态（过拒 2.5%–3.8%）和更低的 ASR。
- **vs LED**：LED 是**层级**定位 + 编辑，TripWire 是**神经元级**；LED 的 ASR 3.8%–5.6%、过拒 8.8%–22.5%、MT 4.28–7.68，在 utility 上并不输 TripWire 编辑形态，说明“更细粒度”不等于“更少误伤”。
- **vs TraceRouter**：最接近的神经元级防御。TraceRouter 定位的是**生成有害内容**的神经元，做擦除/反转；TripWire 定位的是**检测有害请求 + 触发拒绝**的神经元，做钳制。两者对象不同：一个是“别说出有害内容”，一个是“看到有害输入就拒绝”。推理形态下 TripWire 在 ASR 和 MT-Bench 上都优于 TraceRouter，是全文最干净的对比；但 TraceRouter 的过拒 2.5%–7.5% 与 TripWire 推理形态 2.5%–3.8% 接近，优势没有 ASR 那么大。
- **vs RepE / 激活引导**：RepE 用平均表征差做 steering，噪声大、过拒最高 65%；TripWire 的 clamp 也是一种激活干预，但粒度到单个神经元、目标值是 harmful-conditional mean，且用统计漏斗选点——可以看作“RepE 的神经元级、认证版”。
- **vs Circuit Breaker / 其他防御**：论文 Related Work 提到 Circuit Breaker（子模块级表征干预），但实验没比；“among all defenses” 的说法因此偏宽。
- **vs 我们的项目（[[007-编辑防御攻防完整idea]] / [[009-撞车调研与论文结构]]）**：TripWire 是 009 里的“同行冤家”——审计对象、对照组、弹药库。它骂 DELMAN/EVA “always-on 扰良性”的原话，正好被它自己编辑形态 31.3% 的过拒反噬；它的推理形态要挂检测器，零开销卖点也没了。它没做归因、没修 DELMAN 这族锁，也没测 adaptive attack。但它的**认证漏斗**可以搬：硬负例集（含词良性 prompts）+ FDR 联合校正 + 认证支撑集联合解，正好对应 009 的可移植清单。

## 论文存在的问题与下一步研究工作

- **论文存在的问题**：
  1. **编辑形态的过拒没有被解决，反而被摘要淡化**。Table 1 编辑形态 top-2500 过拒 15.0% / 31.3% / 12.5% / 17.5%，MT-Bench 4.31 / 4.84 / 5.86 / 7.45；推理形态只有 2.5% / 3.1% / 3.8% / 3.8%。论文批评 DELMAN/LED 的 always-on 干预会扰动良性，但它自己的离线 bias-patch 就是 always-on，过拒更高。摘要只说“utility drop 0.5%–5.3%，smallest among all defenses”，完全没提 over-refusal；而“smallest utility drop”只对推理形态成立——编辑形态在 Llama-3.1（4.84 vs DELMAN 5.78 / LED 5.26）和 Qwen2.5-7B（5.86 vs 6.16 / 5.88）上更差。**这是全文最大的缺口，也是我们审计章最直接的弹药。**
  2. **“Certified” 是过度承诺**。BH-FDR 控制的是“拒绝集里零假设神经元比例的期望”，前提是 p 值有效（i.i.d. prompt、独立或 PRDS 依赖）；神经元激活高度相关，论文没讨论依赖结构。更关键的是，BH 之后又按 AUROC 在**同一批数据**上取 top-N，最终集合的 FDR 不再有保证；utility-specificity 的 ρ > 0.5 也是启发式阈值，没有认证。把最终 top-N 叫“statistically certified safety neurons”名不副实。
  3. **没有 adaptive attack；检测器是 last-token 上的线性分类器，容易被打**。论文只测四种标准攻击。一个知情攻击者至少有三条路：(a) 在有害请求后追加良性后缀，让 last-token 激活看起来良性，骗过检测器（检测器只看 last-token 的选中神经元激活）；(b) 优化 prompt 让选中神经元的激活偏离 `ā_i`；(c) 利用 `ā_i` 是在 HarmBench 上算的固定值，换分布/语言/模态使其失配。论文声称“does not depend on the specific attack”，但没有 adaptive 实验，这个 claim 不成立。
  4. **识别集和 utility 过滤都不针对 over-refusal**。`D_b` 是 Alpaca 通用指令，`D_u` 是 MMLU 知识题；两者都不含“含敏感词但良性”的硬负例（XSTest / Or-Bench 那类）。效用特异性 ρ 只剔除“对 MMLU 也重要”的神经元，剔不掉“在良性敏感语境也 fire”的神经元——这正是编辑形态 31.3% 过拒的来源。论文没有任何 XSTest / Or-Bench / 分层语境测量，over-refusal 只用 MT-Bench 关键词匹配，口径很粗。
  5. **检测器性能没有报告**。没有 FPR/FNR、阈值、校准、训练样本量；检测器与识别共用 `D_b ∪ D_h`，没有跨分布测试。端到端的 over-refusal 把“检测器误报”和“clamp 泄漏”混在一起，无法归因。
  6. **神经元范围只到 MLP gate/up + last token**。忽略 attention heads、`down_proj`、多 token 动态；也没有因果验证（activation patching / ablation）证明这些神经元是“检测/拒绝”的原因而不是相关物。Figure 3 的分布分析是描述性的，不能支撑“safety neurons”的因果命名。
  7. **评测面偏窄**。“among all defenses” 只比了 RepE/TraceRouter/LED/DELMAN，没比 Circuit Breaker、SafeDecoding、prompt-based 防御、adversarial training；只有 4 个开源 7B–32B 模型，没有 API 模型、多语言、多模态；没有多种子/误差棒；ASR 用 GPT-4.5 判、MT-Bench 用 GPT-5.4 判、过拒用关键词匹配，均无人工复核。
  8. **数字与表述不一致，且三模型/四模型叙述混用**。正文说 Llama-3.1 推理形态 utility 掉 4.7%，但 Table 1 是 6.60→6.48 = 1.8%；摘要的“0.5%–5.3%”只覆盖推理形态，Qwen2.5-32B 编辑形态是 8.23→7.45 = 9.5%；正文说 RepE 8.3%–12.6%、DELMAN/LED 2.8%–4.9%，但 Table 1 里 Qwen2.5-32B 的 RepE 是 2.6%、LED 是 5.6%；“smallest utility drop among all defenses”与编辑形态的表格矛盾；Figure 3 标题写“Toxic Neurons”而正文叫“safety neurons”；Table 2 的“Extr.”列实际是识别耗时（秒），caption 没定义。这些是表述问题，但会影响可信度。
  9. **“Training-free” 措辞**：识别确实不用训练，但推理形态的检测器是训练出来的 logistic 回归；作为整体防御说“training-free”不准确。
  10. **附录缺失**：正文多处说 attack 配置、judge prompt 在 Appendix，但这份 9 页 PDF 没有附录；复现依赖匿名仓库。
- **作者提出的未来工作**：论文没有单列 Future Work。结论把“认证识别 + 触发式干预”定位为通用范式，暗示可推广到更多模型/攻击/模态，但没有具体计划；对编辑形态的过拒、adaptive attack、检测器校准都没有提。
- **我认为下一步可以怎么做**：
  1. **把认证漏斗搬进 DELMAN/EVA 的 key 选择**（对应 009 的可移植清单）：用逐神经元 Welch + BH-FDR + 效用特异性筛“编辑支撑集”，替代/增强 GPT-4 圈词；用含词良性 prompts 做硬负例，做 FDR 跨编辑联合校正；再试“认证支撑集联合闭式解”。这是最直接、成本最低的增量。
  2. **审计 TripWire 编辑形态的过拒**（对应 009 审计章）：在 XSTest / Or-Bench 上按语境分层（新闻/比喻/医学/安全研究）测 over-refusal；同时测检测器在良性敏感 prompt 上的 FPR。论文自己报了 31.3%，我们把“31.3% 从哪来”拆成“检测器误报 / clamp 泄漏 / 神经元选择”三份，正好补它没做的归因。
  3. **做 TripWire 的 adaptive attack**（对应 009 攻击章）：三条线——(a) last-token 良性后缀骗检测器；(b) 优化 prompt 让选中神经元偏离 `ā_i`；(c) 换语言/换分布让 HarmBench 均值失配。这能直接检验它“attack-agnostic”的 claim。
  4. **因果验证 + 与 TraceRouter 对照**：用 activation patching / 消融验证选中的神经元是“检测/拒绝”而非“生成有害内容”；在同一神经元预算下比较 clamp vs erase，隔离“识别”和“干预形态”的贡献。
  5. **组合性实验**：TripWire bias-patch 之后再跑 DELMAN/AlphaEdit 的知识编辑，看 clamp 是否被覆盖、知识编辑是否被 clamp 干扰——这是“编辑防御家族”没人做过的运维问题，和 009 的运维协议/回滚演练能接上。
  6. **把“过拒”提到一阶指标**：所有防御表同时报 ASR / MT-Bench / XSTest-过拒 / 检测器 FPR；论文的“smallest utility drop”只在推理形态成立，这个反例本身就是我们审计章的开场。

## 阅读疑问

- Llama-3.1 推理形态的 utility drop 正文写 4.7%，Table 1 算出来是 1.8%（6.60→6.48）——是笔误，还是用了别的口径？
- 摘要的“0.5%–5.3%”和“smallest among all defenses”是否只指推理形态/三个模型？Qwen2.5-32B 编辑形态 8.23→7.45（9.5%）怎么算？
- BH-FDR 之后再做 AUROC top-N，最终集合的 FDR 还有保证吗？论文的“statistical upper bound”具体 bound 的是哪个集合？
- 检测器的阈值怎么选？在 XSTest/Or-Bench 这类良性敏感 prompt 上的 FPR/FNR 是多少？
- last-token 特征是否会被“良性后缀”骗过？有没有做过 suffix attack 的 pilot？
- Table 2 的“Extr.”是识别耗时（秒）而不是比例，caption 没写清楚；另外 Llama-3.1 上 TripWire top-2500 过拒 31.3% 高于 top-1000 的 8.8%，是哪些神经元造成的？
- Proposition 1 的等价性在 MLP 原有 bias 非零时怎么处理？是把 `b_i` 覆盖还是加到原 bias 上？
- 推理 overhead 1.13–1.15× 的来源是什么？检测器只是线性层，为什么不是 1.00×？
- 攻击结果是单次运行还是多种子平均？没有误差棒，0.3% vs 0.8% 这类差异是否显著？
- 9 页 PDF 没有 Appendix，attack 配置/judge prompt 在哪？

## 我的理解 / 个人思考

- **如果让我向同学解释**：模型内部有一小撮“危险检测器”神经元。TripWire 不删除它们，而是把它们固定在“看到有害输入时”的激活值上，模型就以为自己正在被攻击，于是启动对齐阶段学过的拒绝。找这些神经元不靠外部分类器，而靠统计检验：有害 vs 良性的激活差异显著（Welch t + BH-FDR），且对正常任务不敏感（效用特异性 ρ），最后按 AUROC 排序取 top-N。干预有两种形态：推理时挂检测器、只对有害请求 clamp；或者直接把 clamp 写进权重（bias-patch），永久生效、零开销。
- **和已有知识的联系**：这本质上是“差异表达分析”搬到神经元上——把神经元当基因、把有害/良性 prompt 当两组样本，Welch t + BH-FDR 是生物信息学里的标准配方；AUROC 是效应量排序。干预侧是 activation steering / activation patching 的神经元级版本；bias-patch 等价于把 steering 向量焊进权重，和 ROME/MEMIT 的“闭式写权重”是同一类操作，但 TripWire 不写新 key-value，只改已有神经元的常数项。和 Circuit Breaker 的区别是粒度（子模块 vs 神经元）和方向（表征旋转 vs 单神经元钳制）。
- **对自己研究的启发**：
  - **审计章**：TripWire 自己报了编辑形态 31.3% 的过拒，但没做分层、没做归因、没测检测器 FPR——这正是我们“首套面向编辑防御家族的、按语境分层的 over-refusal 审计”的现成靶子。把 31.3% 拆成“检测器误报 / clamp 泄漏 / 神经元选择”三份，就是我们的方法论增量。
  - **改进章**：认证漏斗可以移植到 DELMAN/EVA 的 key 选择——用硬负例 + FDR + 效用特异性替代 GPT-4 圈词，给出误选上界；但要注意它的漏斗只保证“有害 vs Alpaca/MMLU 可分”，不保证“良性敏感语境不 fire”，所以我们要把 `D_b` 换成含词良性硬负例，把 ρ 的分子/分母改成“有害语境 vs 良性敏感语境”。
  - **攻击章**：last-token 检测器 + 固定 `ā_i` 是两个可打的点；良性后缀骗检测器、优化激活偏离均值、换分布让均值失配，都是低成本 pilot。
  - **运维协议**：bias-patch 是永久编辑，和后续知识编辑/其他防御编辑的干扰没人测；这能接上 009 的“回滚演练/批量预算”。
- **一句话记忆**：把安全神经元当成差异表达基因——Welch t + BH-FDR 选出“有害 vs 良性”显著、且对正常任务不敏感的神经元，再把它们钉在有害时的均值上，让模型自己触发拒绝；推理形态过拒低但要检测器，编辑形态零开销但过拒最高 31.3%。

