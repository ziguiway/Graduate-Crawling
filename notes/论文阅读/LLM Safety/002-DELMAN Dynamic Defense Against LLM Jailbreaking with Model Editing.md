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

- 作者 / 年份 / 会议：Yi Wang, Wenjie Wang*（上科大）、Zhan Qin（浙大）、Minlie Huang（清华 CoAI）等 / 2025-02 arXiv
- PDF：[[../../../papers/mllm-safety/DELMAN_2502.11647.pdf]]
- 代码：https://github.com/wangline/DELMAN
- 相关：[[001-Injecting Universal Jailbreak Backdoors into LLMs in Minutes]]（同一机制的攻击面版本，互为镜像）

## 一句话概括

> 用 **MEMIT 式闭式编辑 + KL 正则**解决部署后 LLM 越狱修复的问题：200 条有害 query + 一句拒绝句 + 0.4h 训练，ASR 压到 6.7%（基线 8.8%~23.2%），MT-Bench 几乎不掉。

## 摘要四要素

- **What**：已部署模型被越狱，重训安全微调代价大、伤能力。
- **Why**：需要"最小修改、精准定位、可动态更新"的部署后防御。
- **How**：在 MLP 里植入"有害 token → 拒绝"的 key→value 联想，闭式解写权重。
- **So What**：5 模型 × 3 攻击全面最低；0.4h、推理零开销、支持连续编辑。

## 核心创新

1. 首次用"直接编辑"（closed-form key→value）做越狱防御，与 DINM/LED 的间接编辑划清界限。
2. 极简配方：200 条 query + 单一统一拒绝句，无需逐条配对。
3. KL 正则防过拒绝：良性语境（"What is a bomb?"）输出分布不变——保 utility 的命门。

## 方法

**假设**：FFN 是 key-value 记忆库（ROME）。插新记忆没得"找"，ROME 的 causal tracing 被两个便宜替代：改哪个词 → GPT-4 标注；插哪层 → 抄 MEMIT 结论（Llama2 层 {3,8}）。

1. **抽有害 token**：GPT-4 从 200 条 query 里圈出"真凶词"（bomb、poison…）→ T_h。地址选准，不误伤正常词。
2. **造 key k\***：每个 token 生成 N=5 条**中性句**，喂原模型，在目标层截 token 位置的 MLP key 向量，平均 → k\*。用中性句而非有害 prompt：防御要求任何语境都命中 → key 必须上下文无关。平均消噪声，留下 token"指纹"。
3. **优化 value v\***（唯一用梯度的步骤）："想象替换"当模拟器——把该层 MLP 输出强行换成 v，看模型接什么。L_safe：有害 query 要逼出拒绝句；L_utility（KL）：良性问法输出分布不许动。产出校准过的拒绝方向（有害时翻得动，正常时推不动）。
4. **闭式解写权重**：解带约束最小二乘，(k\*→v\*) 焊进多层 W_down，残差按层数摊薄、浅层少改。

**推理时**：有害 query 的 bomb 命中 k\* → 注入 v\* → 拒绝；"What is a bomb?" 同样命中但 KL 保证推不动 → 照常回答。

## 框架图

![[../../../papers/mllm-safety/DELMAN_2502.11647.pdf#page=2]]

Figure 2：五步流程（抽 token → 造句 → 算 k\* → 优化 v\* → 更新 W_down）。

## 实验设置

- 数据集：编辑用 HarmBench 200 条；评测 AB/JBB/MI 各 100 条（**没见过的**，考泛化）；utility 用 MT-Bench + 7 下游任务。
- 攻击：GCG、AutoDAN、PAIR。Baseline：SafeDecoding、LoRA、LED。
- 指标：ASR（HARM-BENCH 分类器）、MT-Bench、时间/开销。

## 实验结果

- 平均 ASR 全部最低：Vicuna 6.7%（LoRA 23.2 / SafeDecoding 10.7 / LED 8.8）；如 Vicuna-GCG 四数据集 11/2/17/1 vs 原模型 92/89/89/94。
- **泛化**：HB 编辑 → AB/JBB/MI 照样压住；LED 反例——只见编辑集内好、未见集崩且 MT-Bench 掉到 5.70（DELMAN 6.84 vs 原 6.77）。
- 效率：0.4h 训练（LED 6.2h）、1× 推理（SafeDecoding 1.07×）。
- 消融：KL > JS > 余弦正则；干净 token 编辑 ASR 同样归零但 MT 6.31→5.09（选对 token 是保 utility 的命门）；连续 4 轮编辑不回弹；**泛化机制**（PCA）：不同类别/数据集的有害 token 的 k 大量重叠 → 一批 token 挡多类攻击。

## 与其他论文的关系

- 骨架直接继承 ROME/MEMIT，创新在场景和 v\* 的目标函数（安全 + KL）。
- vs [[001-Injecting Universal Jailbreak Backdoors into LLMs in Minutes]]（JailbreakEdit，攻击）：同骨架，方向相反——
	- key：它用**有害 prompt** 平均（只需有害语境命中），我用**中性句**（全语境命中）
	- value：它 16 个接受短语**多节点**压安全机制，我**单句 + KL**（此选择由 key 设计推出：无差别命中 ⇒ 必须 KL 兜底；冷僻触发词 ⇒ 隐身免费）
	- 它单层要爆发力，我多层要稳定；它要控制输入（供应链后门），我随时打（部署后补丁）
	- 互证：它的 baseline 证明单点 value 推力不够 ⇒ 正好是我第 3 步优化的必要性
- 两篇互不引用、互不测对方 → 留出空白（见下）。

## 论文存在的问题与下一步

- **adaptive attack 没测**（最大缺口）：换同义词/换表述绕过 token 防御。
- 无 over-refusal 指标（XSTest）；MT-Bench"反升"在 judge 方差内；λ 差 12 倍没报敏感性；只测 7B~14B。
- 我的下一步：① 双重补丁（在触发词位置写竞争记忆，对冲编辑后门）② 多节点拒绝 vs 单节点+KL ③ adaptive attack 测试——两个代码库都开源，成本低。

## 阅读疑问

- v\* 也要 200 步梯度优化，"direct editing"其实只覆盖 W_down 一步，措辞偏大。
- Qwen2.5 的 λ_KL=0.75，其他模型 0.0625，差 12 倍，敏感性未报。

## 我的理解 / 个人思考

- **给同学解释**：门禁卡类比——量出这张卡的真实卡号（5 句平均消噪声），登记"刷到它 → 开保险柜（拒绝）"；KL 保证正常刷卡行为不变。
- **想法被实现的校准**：DELMAN（2025-02）早于我 001 笔记的想法——写下时它已存在快一年，没搜到。大方向被验证，但步骤 1~3 的粗版本市场没了。
- **学走的**：① KL 正则 = "对齐税"的答案，防误伤要写进优化目标，不能靠事后调阈值；② 粒度：token 级而非 prompt 级，粒度选错后面救不回来；③ 评测栈：ASR 归零不算赢，ASR 归零且 MT-Bench 不掉才算。
- **还要补**：ROME/MEMIT 原论文（能自己推 Eq.6-8）、跑通开源代码、PCA/t-SNE 分析手感。

> **一句话记忆：** 在 MLP 里植入"有害 token 指纹 → 拒绝方向"：地址用中性句平均测出来，内容用想象替换 + KL 校准，权重用闭式解焊进去——0.4h、200 条、零推理开销。
