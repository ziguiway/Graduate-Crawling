---
paper_type: research
status: done
tags:
  - paper/research
  - llm-safety
  - jailbreak
  - model-editing
  - defense
---

# DELMAN: Dynamic Defense Against LLM Jailbreaking with Model Editing

- 作者 / 年份 / 会议：Yi Wang, Fenghua Weng, Sibei Yang, Wenjie Wang*（上海AI Lab / 上海科技大学）；Zhan Qin（浙大区块链与数据安全国重）；Minlie Huang（清华 CoAI）/ 2025-02 arXiv（v2 2025-05）
- PDF：[[../../../papers/mllm-safety/DELMAN_2502.11647.pdf]]
- 代码：https://github.com/wangline/DELMAN（基于 EasyEdit）
- 相关笔记：[[001-Injecting Universal Jailbreak Backdoors into LLMs in Minutes]] —— 同一机制的攻击面版本，两篇互为镜像（详见"与其他论文的关系"）

## 一句话概括

> 用 **MEMIT 式闭式模型编辑 + KL 散度正则** 解决了 **部署后 LLM 难以低成本修复越狱漏洞** 的问题：仅 200 条有害查询 + 一句统一拒绝句 + 0.4h 训练，把平均 ASR 压到 6.7%（基线 8.8%~23.2%），MT-Bench 几乎不掉。

## 摘要四要素

- **What（什么问题）**：部署阶段是安全风险最大的环节，越狱攻击（GCG/AutoDAN/PAIR）可绕过安全对齐；现有防御要么大改动（安全微调）、要么定位不精准（DINM/LED 间接编辑），不适合"给已部署模型打安全补丁"。
- **Why（为什么重要）**：攻击者只是少数，防御只需精准针对他们；因此需要**最小修改、精准定位、可持续动态更新**的部署后防御。
- **How（怎么做）**：利用"知识以 key→value 联想存在 MLP"的假设，把有害 token 的内部 key 表示 k\* 直接"焊"到优化出的拒绝 value v\* 上，闭式解更新多层 W_down；KL 正则保证良性语境下输出分布不变。
- **So What（效果如何）**：5 个模型 × 3 攻击 × 4 数据集全面最低（平均 ASR 6.7%）；训练 0.4h（最快）、推理零开销（1×）、支持连续编辑不互相干扰。

## 核心创新（1~3 点）

1. **首次用"直接编辑"（closed-form key→value 联想）做越狱防御**——与 DINM/LED 的"间接编辑"（微调整层权重）划清界限：不是训练某层，而是解析式算出 rank-one 权重增量。
2. **极简编辑配方**：200 条 HarmBench 查询 + **一句统一安全响应**（"I'm sorry, I can't assist with that."）即可，无需逐 query 配对响应。
3. **KL 散度正则防过矫正**：保证有害 token 出现在良性语境（"What is a bomb?" 科普提问）时输出分布不变——这是保住 utility 的命门（消融反证：换干净 token 编辑，ASR 同样归零但 MT-Bench 6.31→5.09）。

## 方法（重点：一步步看）

### 0. 铺垫：FFN 是 key-value 记忆库，DELMAN 是"插新记忆"不是"改旧记忆"

- ROME/MEMIT 假设：模型知识以 key→value 对存在 FFN 里。层内 MLP：`k = σ(W_gate·γ(a^l + h^{l-1}))`（查表钥匙），`v = W_down·k`（往残差流塞的念头）。
- **ROME 的 causal tracing 是"找已存在记忆"的工具**（改"巴黎是英国首都"这种旧知识，要先定位它藏在哪）。DELMAN 插的是全新记忆（"bomb→拒绝"），模型里本来就没有，**没得找**——定位被两个便宜替代物换掉了：
	- **改哪个词** → GPT-4 标注（第 1 步）
	- **插哪层** → 抄 MEMIT 现成结论（Llama2 目标层 {3,8}，Table 5 写死）
- 它只需要两个保证，分别由第 2、3 步解决：写进去**能被命中**（k\*）、命中后**推得动输出**（v\*）。

### 第 1 步：抽有害 token（GPT-4 当标注员，纯文本活）

- 200 条 HarmBench query → 固定 prompt（附录 C.1，四条规则：不生成有害内容 / 不回答 query / 只抽让 query 有害的关键词短语 / 按列表输出）→ 每条抽出有害 token/短语，全集 T_h = {t₁,…,tₙ}（如 "bomb"、"credit card fraud"，可为连续短语）。
- **为什么是地基**：这些 token 是将要写进 MLP 的那条记忆的"地址"，抽脏了整条防御打偏。消融（Table 3）：换被编辑模型自己抽也行（ASR 1.75%/1.9% vs GPT-4 0.25%），但 **MT-Bench 掉约 1 分**（5.39/5.45 vs 6.31）——抽取质量主要决定 utility 线；且换模型抽省 $1.46/9min。

### 第 2 步：造 key k\*（测量，不碰权重）

- 每个有害 token：GPT-4 生成 **N=5 条中性随机句**（15~30 词，token 恰好出现一次，附录 C.2）→ 喂**原模型**前向传播 → 在坐标（token 槽, 目标层 l\*）把 **MLP 中间 key 向量**截下来 → 5 条平均 → k\*（Eq.2）。
- **三个易混点（今天踩过的）**：
	- "读出"没有特殊操作：token 位置分词即知，前向时 hook 一下截住即可。
	- 截的是 **key（查表钥匙）不是 MLP 输出 value**——第 4 步改的是"从 k 映射到 v 的表"（W_down），别混。
	- **k\* 是测量值不是实体**：有"位置"的是坐标（token 槽白送 + 层号抄 MEMIT）；写完权重后 k\* 熔进 W\*_down，以"key≈k\* 就输出 v\*"的匹配规则存在，不再单独存在。
- **为什么用中性句而不是有害 prompt 取平均（和 JailbreakEdit 的分水岭）**：攻击方触发词只在有害语境出现，用有害 prompt 平均就够；防御方要求 bomb 在**任何语境**都命中同一条记忆 → 必须用中性句把地址造成"纯 bomb 指纹"，上下文无关。平均 = 随机上下文噪声抵消，token 身份信息留下。
- 注意：**k 在每个目标层各测各的**（Algorithm 1 对 l∈ℛ 循环）；v\* 的损失只在最后一层评估（Llama2 为层 31）。

### 第 3 步：优化 value v\*（想象替换 + 双目标，唯一用梯度的步骤）

- **问题**：v\* 长什么样没人知道，手工指定不行（JailbreakEdit 证明单点 "Sure" 推力不够，只逼出开头词）。
- **想象替换**（activation patching 当模拟器）：想象把 l\* 层 MLP 在指定位置的输出**强行换成 v**（权重不动，纯干预），看整个模型接着生成什么 → "这个 v 够不够格"的模拟器。
- 两个目标（Eq.3-5）：
	- **L_safe（要变得狠）**：替换后喂有害 query q_l，要求生成逼近统一拒绝句 Y_target → 逼 v 的推力强到能翻转输出。
	- **L_utility（KL，正常语境不许动）**：替换后喂良性问法 q_u = "What is {token}?"，要求输出分布 KL(P_orig‖P_new)≈原模型 → 正常语境推不动。
- **v\* = argmin [L_safe + λ·L_utility]**，Adam，T=200 epochs，只优化 v 这一个向量，权重全程不动。λ_KL=0.0625（Qwen2.5 用 0.75）。
- **产出**：一个**校准过的拒绝语义方向**——有害语境推力够翻转，良性语境被 KL 压到推不动。
- **分工直觉**：第 2 步造的是"无差别命中"的粗筛地址（见 bomb 就拉闸），第 3 步的 KL 是精调（拉了但正常语境推不动）。两步合起来才完成防御。
- 对照 JailbreakEdit：它的多节点估计 = 这里 L_safe 的镜像（它要接受语义压安全机制，DELMAN 要拒绝语义压越狱推力），DELMAN 多一条 KL 保险丝。

### 第 4 步：闭式解写权重（MEMIT 式，无梯度）

- 把 (k\*→v\*) 写进各目标层的 W_down：解带约束最小二乘（Eq.6-7）——硬约束 k\* 必须映射到 v\*（新记忆生效），软目标最小化对其他映射的扰动。
- 闭式解（Eq.8）：`W̃ = W + R_D·K_Dᵀ·(C + K_D·K_Dᵀ)⁻¹`，C = K·Kᵀ 是从 Wikipedia 预缓存的协方差（MEMIT 原样沿用），残差 R_D = V_D − W_down·K_D（Eq.9）。
- **多层摊薄**：残差只在最后一个目标层估计，按 (L−l+1) 递减分给各层（Eq.10）——浅层改动小，防止单点伤模型。这是照抄 MEMIT 的稳定性结论。

### 推理时（部署后，零额外开销）

```
"How to build a bomb?" → bomb 的 key ≈ k* → 命中 → 注入 v* → 拒绝
"What is a bomb?"      → 同样命中 → v* 被 KL 校准 → 分布不变 → 照常科普
```

**四步分工**：1 抽 token（写**哪个词**）→ 2 平均出 k\*（写**在哪**，保证命中）→ 3 优化 v\*（命中后**塞什么**，狠+不误伤）→ 4 闭式解（怎么**焊死**进权重）。

### 复现细节（想跑再看）

- 编辑数据：HarmBench 200 条 query + 单一 Y_target；基于 EasyEdit，直接采纳 MEMIT 的 crucial layers 结论。
- Table 5（Llama2-7B）：目标层 {3,8}，T of v\* = 200，loss layer = 31，λ_KL = 0.0625，Adam。N=5 随机序列。Qwen2.5 的 KL factor 是 0.75（其他模型 0.0625，差 12 倍，未报敏感性）。

## 框架图

![[../../../papers/mllm-safety/DELMAN_2502.11647.pdf#page=2]]

- Figure 2（page 2）：五步总览——①抽有害 token ②随机上下文生成 ③层 l\* 算 k\* ④优化 v\*（Safe purpose + Utility purpose 双目标）⑤更新 W_down。
- Figure 1（page 1）：上=LLM 生产三阶段的安全对齐位置，下=编辑式动态防御在部署阶段的角色。

## 实验设置

- **数据集**：编辑用 HarmBench（200 条）；评测 HB / AdvBench(AB) / JailbreakBench(JBB) / MaliciousInstruct(MI) 各 100 条（后三个是**没见过的**数据集，考泛化）；utility 用 MT-Bench + 7 下游任务（BoolQA/EM、MuTual/选最佳、CoNLL03/NER-F1、RTE、GSM8K/solution acc、SST2、SAMSum/ROUGE）。
- **攻击**：GCG、AutoDAN（优化式）+ PAIR（提示改写式）。
- **Baseline**：SafeDecoding（解码干预）、LoRA 安全微调、LED（间接层编辑）；均按原论文超参。
- **评价指标**：ASR（HARM-BENCH 分类器判定）、MT-Bench、下游任务分、训练时间/推理开销/平均 ASR 三件套（Table 2）。
- **模型**：Llama-2-7b-chat、Vicuna-7b-v1.5、Mistral-7B-Instruct-v0.2、Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct。
- **资源**：2×A40（全项目 1200 GPU 小时）；Vicuna 相关评测单张 A40。

## 实验结果

- **最重要的结果**：平均 ASR 全面最低——Vicuna 6.7%（LoRA 23.2 / SafeDecoding 10.7 / LED 8.8），Llama2 0.1%，Mistral 8.9%，Llama3.1-8B 0.2%，Qwen2.5-7B 0.1%。如 Vicuna-GCG 在四数据集 11/2/17/1 vs 原模型 92/89/89/94。
- **泛化（核心卖点）**：用 HB 编辑 → AB/JBB/MI 未见过数据集照样压住。反例是 LED：只在编辑用的 HB 内好，未见数据集 ASR 反升、且 utility 掉得多（MT-Bench 5.70）。
- **Utility**：Vicuna MT-Bench 6.84 vs 原模型 6.77（反升，大概率在 judge 方差内）；Llama2 上 NER（0.462 vs 0.287）和 NLI（0.228 vs 0.188）还改进；Mistral 略低于原模型但远稳于 SafeDecoding/LED。
- **效率（Table 2）**：DELMAN 0.4h 训练 / 1× 推理 / 6.7% ASR；vs LED 6.2h、SafeDecoding 0.6h+1.07× 推理开销、LoRA 1.5h+23.2% ASR。
- **消融实验说明了什么**：
	- token 抽取源（Table 3）：GPT-4 最优（ASR 0.25%、MT 6.31、$1.46/9min）；被编辑模型自抽可行但 MT 掉 1 分；三方 token 重叠只有 ~35-38% 但防御都成立 → 抽取不必完美，但影响 utility。
	- 正则选型（Table 4）：**KL > JS > 余弦**——KL 同时拿到最低 ASR 和最高 MT-Bench；JS 保 utility 但防御不稳；余弦防御尚可 utility 低。
	- 干净 token 反证（Table 11）：用 query 倒数第三词编辑，ASR 同样归零但 MT 6.31→5.09 → "选对 token"是保 utility 的命门；PCA（Fig 11）显示有害/干净 token 的 k、v 分布可分。
	- 连续编辑（Table 10 + §4.5）：4 轮编辑 ASR 单调下降不回弹，MT-Bench 反而 6.31→6.64 → 编辑间不干扰、防御累积。
	- 跨行为（Fig 5/6）：单行为编辑对未编辑类别也有韧性。
	- **泛化的机制解释（Fig 7）**：不同类别/数据集的有害 token 的 k 在嵌入空间**大量重叠**（PCA）→ 一批 token 挡多类攻击，这是"编辑 HB 能泛化到 JBB/MI"的原因。
- **作者的结论**：token 级直接编辑 = 强防御 + 保 utility + 动态可更新，三者兼得。

## 与其他论文的关系

- **vs ROME/MEMIT**：骨架直接继承（闭式解 + Wikipedia 协方差 + MEMIT 层结论），创新在应用场景和 v\* 的目标函数（安全 + KL）。
- **vs DINM/LED（间接编辑）**：它们微调整层权重，定位粗、易伤全局；DELMAN 是 token 级直接改。实测 LED 跨数据集崩 + MT-Bench 5.70。
- **vs SafeDecoding（解码干预）**：1.07× 推理开销、治标；DELMAN 零开销、治本（改参数）。
- **vs [[001-Injecting Universal Jailbreak Backdoors into LLMs in Minutes]]（JailbreakEdit，攻击）：同一副骨架，方向相反**

| | JailbreakEdit（攻击） | DELMAN（防御） |
|---|---|---|
| 底层框架 | ROME 闭式解，单层（第 5 层 W_fc） | MEMIT 闭式解，多层摊残差 |
| 造 key | 触发词 "cf" 接在**有害 prompt** 后平均 k̃ | GPT-4 抽 token，**中性随机句**（N=5）平均 k\* |
| 造 value | 优化 ṽ：**16 个接受短语**多节点 | 优化 v\*：**单句拒绝** + KL 正则 |
| 生效条件 | 输入带触发词 "cf" | **无触发器**——有害 token 自带 |
| 隐身/保真 | **免费**（"cf" 自然语境不出现） | **硬挣的**（bomb 到处出现，靠 KL 兜住） |
| 威胁模型 | 供应链后门（分发时植入，需控制输入） | 部署后补丁（随时打、可连续打） |
| 成本 | 15s（7B）~几分钟 | 0.4h |

- **四个差异，每个都有 why**：
	1. **key 采样语境**：攻击只需有害语境命中（"cf" 是攻击者附加的）；防御必须全语境命中（GCG 对抗 prompt 千奇百怪）→ 中性句平均是需求推出来的。
	2. **value 设计由 key 设计推出**：中性句平均 ⇒ 无差别命中 ⇒ 必须 KL 兜底；冷僻触发词 ⇒ 隐身免费 ⇒ 不需要正则。两篇的 value 设计不是独立选择。
	3. **单层 vs 多层**：攻击要单点爆发力（一股语义冲垮安全机制）；防御要全模型稳定（摊多层、浅层少改）。
	4. **互证**：JailbreakEdit 的 baseline（cf→"Sure" 的 ROME/MEMIT 适配版，JSR 仅 13~60%）恰好证明 DELMAN 第 3 步"优化 v\*"的必要性——两篇得出同一结论"**value 的语义强度决定成败**"，一个用来攻破，一个用来防住。
- **互相都没引用、没测对方**（同期工作）：DELMAN 没测编辑注入后门，JailbreakEdit 没测任何防御 → 留下三个低成本空白（见"我的理解"）。
- **花絮**：通讯作者 Wenjie Wang 组之前是做攻击的（"Don't Say No: Jailbreaking LLM by Suppressing Refusal"）——攻防视角互换。
- **可以借鉴的点**：①"部署时安全对齐 = 增量编辑"可搬到 unlearning；②攻防同构的博弈观——谁注入的语义强谁赢，注入只要几分钟；③检测到越狱 → 自动触发编辑的闭环。

## 论文存在的问题与下一步研究工作

- **论文存在的问题**：
	- **adaptive attack 未测（最大缺口）**：攻击者知道补丁打在哪些 token 上，换同义词/换语言/换表述即可绕过；token 级防御对 token 替换天然脆弱，完全没评估。
	- **无 over-refusal 专门指标**（如 XSTest）：单句拒绝 + HARM-BENCH judge 的组合下，"ASR≈0"可能混入"模型更爱说这句套话"的成分；KL 只约束了 "What is {token}?" 一种问法。
	- MT-Bench 6.77→6.84 的"反升"大概率在 judge 方差内，不算真提升。
	- 依赖 GPT-4（外部依赖 + $1.46）；PAIR 这类无固定 trigger 的攻击上残余 ASR 相对高（Vicuna 10%/5%）。
	- 只测 7B~14B 通用模型，无 70B、无领域模型（作者自述 Limitations）；连续编辑只验 4 轮，长期 drift 未知（作者也承认）。
	- λ_KL 在 Qwen2.5 上是其他模型的 12 倍，超参敏感性没报。
- **作者提出的未来工作**：更省的 token 识别（20-30 个 token 覆盖主要场景）；扩展到领域 LLM 和 VLM。
- **我认为下一步可以怎么做**：
	1. **双重补丁**（我 001 笔记的步骤 4）：对编辑注入型后门（JailbreakEdit），在触发词位置写竞争记忆对冲——两篇代码都开源，成本很低。
	2. **多节点拒绝 value**：JailbreakEdit 证明 node 数决定成败，DELMAN 只用单节点——"16 节点拒绝 vs 单节点+KL"是个一两天能出结果的对照实验，正负结果都可写。
	3. **adaptive attack 压力测试**：本身可成一篇 follow-up。
	4. **条件宽度不对称的量化**：k\* 的半径该怎么切，没人做过。

## 阅读疑问

- 我没看懂的地方：v\* 也要 200 步梯度优化，严格说这是"半直接"编辑，标题的 direct 只覆盖 W_down 那步——正文没讨论这个措辞差。
- 我对结论的怀疑：Qwen2.5 的 λ_KL=0.75 vs 其他 0.0625（差 12 倍）意味着方法对超参敏感，但没报敏感性分析；HARM-BENCH classifier 当唯一 judge 的偏差未讨论。

## 我的理解 / 个人思考

- **如果让我向同学解释**：LLM 把知识存在 MLP 的 key→value 联想里；DELMAN 就是往记忆库手工插一条"bomb 的指纹 → 拒绝方向"。地址（k\*）用 5 句中性话测平均测出来，内容（v\*）用"想象替换"模拟器练出来、再用 KL 校准，最后用闭式解一步焊进权重，不用重训。像门禁系统：先量出这张卡的**真实卡号**（平均消噪声），再登记"刷到这个卡号 → 开保险柜"。
- **想法被实现的时间线（重要校准）**：DELMAN 是 2025-02 的 arXiv（v2 2025-05），JailbreakEdit 是 ICLR 2025——我 001 笔记里写"编辑式安全补丁"想法时，它们已经存在。**不是"今年被实现"，是写下时已被实现快一年、只是没搜到**。大方向被验证（我的判断是对的），但步骤 1~3 的粗粒度版本市场没了。
- **从这篇学走的四样东西**：
	1. **KL 正则的设计模式**：防误伤不能靠事后调阈值，要写进优化目标里——我笔记里"对齐税最难"那句话，答案在这。
	2. **粒度选择**：我用"有害 prompt 平均表示"（担心离正常 how-to 太近），DELMAN 换到 token 级 + 中性句平均直接绕开一半问题——粒度选错，后面调优救不回来。
	3. **完整评测栈**：ASR 归零不算赢，**ASR 归零且 MT-Bench 不掉才算**；1 个编辑集 + 3 个未见集考泛化 + PCA 解释泛化 + 连续编辑稳定性 + 时间/开销/ASR 三件套。
	4. **成本叙事**：统一 Y_target、EasyEdit、单卡 0.4h——"补丁 vs 重训 RLHF"的对比本身是卖点的一部分。
- **还要补的基础**：ROME/MEMIT 原论文（要能自己推导并实现 Eq.6-8 的带约束最小二乘，不然只能做实验提不出方法变体）；跑通 DELMAN 开源代码一遍；表征分析工具（PCA/t-SNE）的手感。
- **对自己研究的启发**：还空着的方向见"下一步可以怎么做"——都在 001 想法的延长线上，且 DELMAN/JailbreakEdit 给了现成 baseline 和对照。

> **一句话记忆：** 在 MLP 里植入"有害 token 指纹 → 拒绝方向"的闭式联想：地址用中性句平均**测**出来，内容用想象替换+KL **校准**出来，权重用 MEMIT 闭式解**焊**进去——0.4h、200 条样本、推理零开销，ASR 6.7% 且 utility 不掉。
