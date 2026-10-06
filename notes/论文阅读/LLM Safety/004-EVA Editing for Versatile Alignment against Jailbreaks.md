---
paper_type: research
status: done
tags:
  - paper/research
  - llm-safety
  - jailbreak
  - model-editing
  - defense
  - vlm
---

# EVA: Editing for Versatile Alignment against Jailbreaks

- 作者 / 年份 / 会议：Yi Wang, Wenjie Wang*（上科大）、Zhan Qin（浙大）、Minlie Huang（清华）等 / 2026-05 arXiv（IEEE 格式，cs.CR）
- PDF：[[../../../papers/mllm-safety/EVA_2605.14750.pdf]]
- 代码：https://github.com/wangline/EVA
- 相关：[[002-DELMAN Dynamic Defense Against LLM Jailbreaking with Model Editing]]——**同一作者的期刊扩展版**：DELMAN(LLM-only) + 视觉模态 = EVA

## 一句话概括

> 用 **跨模态注意力定位视觉有害 token + MEMIT 式闭式编辑**把 DELMAN 从 LLM 扩到 VLM：文本 key 和视觉 key 拼成一个 K_E 一次编辑，VLM 上 ASR 几乎全归零（ADV-16 0.71→0.00），utility 持平甚至反超原模型。

## 摘要四要素

- **What**：VLM 部署后防越狱——有害意图藏进图像（OCR/对抗图/语义图），现有编辑方法只对文本有效。
- **Why**：安全微调代价大+灾难性遗忘，外挂过滤器加延迟；有害输出源于激活模式错位而非有害知识 → 可以直接改。
- **How**：三阶段：①有害 key 识别（文本沿用 DELMAN，视觉用 cross-modal attention 当 token grounding 代理，top-1 图像 token）；②优化安全 value v*（safe loss + KL，统一拒绝句）；③文本+视觉 key 拼成 K_E，闭式解写 W_down。
- **So What**：LLM 5 模型 × 3 攻击、VLM 3 模型 × 多基准全面最低 ASR；MT-Bench 反升、MM-Vet-v2 持平原模型；0.4h/0.8h、1× 推理、支持连续编辑。

## 核心创新（1～3 点）

1. **首个把直接模型编辑当 VLM 安全防御**：视觉有害信息分散在连续像素里、位置难定 → 用 cross-modal attention 权重当免费 grounding 代理，从有害文本 token 反查图像 token；N_v=1 就够，注意力分数是前向副产品，零额外开销。
2. **统一编辑框架**：文本 key 和视觉 key 作为独立列拼进 K_E/V_E，一次 MEMIT 式闭式解同时改两个模态。
3. **验证"LLM 编辑层可直接搬给 VLM"**：VLM 最佳编辑层（Qwen2.5-VL L8、InternVL3.5 L9）恰与骨干 LLM 的 critical layer 重合——视觉编码器只是特征提取器，编辑全落在语言骨干。

## 方法

核心思路同 DELMAN：FFN 是 key-value 记忆库，有害 token 是 key，越狱 = 检索出有害 value；编辑 = 把"有害指纹 → 拒绝方向"写进 W_down。分**离线编辑**（6 步，跑一次）和**在线推理**（零改动）。

**离线：**

1. **抽有害词**：GPT-4o 从 query 圈出恶意意图词 → "hack"
2. **造变体**：改写 N 条同意图 query → 让指纹跨语境稳定
3. **注意力定位（VLM 新增）**：读 "hack" 对全部图像 token 的 attention，跨 N 条平均，top-1 选中关键图像 token——注意力是前向副产品，零额外计算
4. **算 key（指纹）**：l* 层取该位置的 MLP 中间态 σ(W_gate·x)⊙(W_up·x)，跨 N 条平均；文本/视觉**同一公式**，只是取的位置不同 → K_E = [k_t, k_m] 独立列
5. **优化 value**（唯一用梯度的步骤）：想象替换——v 当可学习变量、模型冻结，有害输入必须逼出统一拒绝句 "I'm sorry, I can't assist with that."，良性输入被 λ·KL 拉住不许变 → v*（有害时推得动、正常时推不动）
6. **写权重**：K_E→V_E 解 MEMIT 式带约束最小二乘（新映射强制成立 + Wikipedia 协方差保旧知识），闭式解 + 残差多层分摊 → W'_down 原位替换

**在线：**无外挂模块；恶意 token（文本或图像）在 l* 层的 key 落在编辑过的 k 附近 → W_down 输出拒绝方向 → 拒绝；良性输入命中同一个 key，但 KL 保证输出不动。

## 框架图

![[../../../papers/mllm-safety/EVA_2605.14750.pdf#page=5]]

Figure 2：六步流程（抽 t → 造变体 → attention 选图像 token → 算 k → 优化 v* → 改 W_down）。

## 实验设置

- LLM：5 模型（Vicuna-7b/Llama2/Mistral/Llama3.1-8B/Qwen2.5-7B）；编辑 HarmBench，泛化测 AdvBench/JailbreakBench/MaliciousInstruct；攻击 GCG/AutoDAN/PAIR；baseline LoRA/SafeDecoding/LED/Circuit Breakers。
- VLM：3 模型（LLaVA-1.5-7B/Qwen2.5-VL-7B/InternVL3.5-8B）；MM-SafetyBench（SD/TYPO/SD+TYPO）、MultiTrust（ADV-16/typo/crossmodal）、MM-Vet-v2 攻击（FigStep/Hades）；baseline JailGuard/AdaShield-A/VLGuard。
- 指标：ASR（越低越好）、MT-Bench、MM-Vet-v2/MMMU/MMStar、训练时长/推理开销。

## 实验结果

- LLM：EVA(text)≈DELMAN，ASR 全面最低（Vicuna AutoDAN 0.69→0.00，强模型基本全 0）；adaptive attack 后回升有限（Vicuna 0.068→0.112，Mistral 0.222 vs 原模型 0.869）。
- VLM：ASR 几乎全 0——ADV-16 对抗图像 0.71→0.00（JailGuard 留 0.56）；MM-SafetyBench TYPO 0.51→0.00；MultiTrust crossmodal 0.88→0.00（Qwen）。
- Utility：MT-Bench 反升（6.84 vs 6.27）；VLM 上 MM-Vet-v2/MMMU/MMStar 持平或反超（InternVL 69.3 vs 69.1）；AdaShield-A/VLGuard 在多个基准崩盘。
- **Layer 消融**：Qwen2.5-VL 最佳 L8、InternVL3.5 L9 = 骨干 LLM 的 critical layer → 经验直接迁移。
- **视觉 token 选择消融**（Table 4）：Random/Fixed 让 utility 崩到 10-12；embedding 法（L2/余弦）保 utility 但漏 ASR；梯度法保 utility 有开销且 ASR 0.02；attention 法 ASR 0.00 + 56.5、零开销。
- **token 数量**：1 个最优（ASR 0.48→0.03，utility 还微升）；2-3 个 ASR 归零但 utility 崩（45.4/19.7）。
- **token 来源**（Table 11）：GPT-4o 最好，Qwen2.5-72B 接近（开源替代），Llama2-7b-chat 明显差。
- **Universal vs Specific 拒绝句**（Table 13）：统一单句大幅更优（InternVL ASR 0.04 vs 0.16）——多样拒绝=监督噪声、软化边界；单句把有限编辑容量聚成一个紧 refusal 方向。
- **解耦与上下文协同**（5.6）：编辑区域与攻击类型对齐——视觉区域编辑治 SD 攻击、OCR 区域编辑治 TYPO；复合图像（SD+TYPO）需要两种 token 都改（full EVA 0.00，单区域 0.12~0.39）。"Focus 比 Only 泛化好"的协同主张数值差距小（0.12 vs 0.17）。
- 泛化机制（PCA）：不同类别/数据集的有害 k 大量重叠 → 共享表示 → 跨类别、跨数据集迁移。
- 效率：LLM 0.4h/1×；VLM 0.8h/1×（VLGuard 10.9×、JailGuard 8×、AdaShield-A 1.05×）。

## 与其他论文的关系

- **= DELMAN 的期刊版**（自引 [23]）。LLM 侧机制原封不动（EVA(text) 就是 DELMAN），增量全在多模态：
	1. 视觉 key 定位：cross-modal attention 当 grounding 代理——对比 L2/余弦/梯度法，唯一零开销且 ASR 归零
	2. 层定位实验：证明 LLM critical layer 在 VLM 照样最优 → 免去重新调层
	3. 跨模态 K_E 拼接 → 一次编辑覆盖两模态
	4. 新消融：token 数、token 来源、正则项（KL>JS>cos，延续 DELMAN 结论）、universal vs specific 拒绝、decoupling/contextual synergy
- vs 编辑文献：与 MEMIT/EasyEdit 同骨架但目的不同（事实联想→安全）；与 SafeLM/CKU 等 representation editing 的区别 = 直接闭式更新、零推理开销、跨模态。
- vs [[001-Injecting Universal Jailbreak Backdoors into LLMs in Minutes]]：仍是同一骨架的攻防镜像（写竞争 key vs 写安全 value），且现在都在 VLM 上有版本了。
- 可以借鉴的点：**"cross-modal attention 当免费 grounding"** 可用于任何 VLM 内部定位任务；"LLM 层经验迁移到 VLM"的验证方法（扫层 ASR + utility 曲线找交点）可直接抄。



