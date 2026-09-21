---
tags: [论文阅读, MLLM安全, 多模态防御]
date: 2026-09-19
---

# MLLM 防御论文阅读索引表

> [!info] 概览
> - **主题**：多模态大语言模型（MLLM）防御
> - **论文数量**：20 篇
> - **来源**：arXiv + Web 检索
> - **本地PDF目录**：本目录（`MLLM防御/`）下，PDF 与本索引同目录
> - **总预估阅读时长**：约 40 小时

---

## 阅读路线

| 阶段 | 目标 | 论文 |
|------|------|------|
| **第一阶段** | 建立全景认知 | #1 → #3 → #4 → #2 → #5 |
| **第二阶段** | 掌握经典防御方法 | #6 AdaShield → #7 Immune → #9 CIDER → #16 SafePTR |
| **第三阶段** | 深入对抗训练与表示级 | #13 MAT → #15 MMCoA → #14 Sim-CLIP+ → #12 E²AT → #17 MMAligner |
| **第四阶段** | 补全检测、基准与前沿 | #10 HiddenDetect → #11 BlueSuffix → #20 OmniSafeBench → #8 SHIELD → #19 SGM → #18 DTR |

---

## P1 必读（8 篇）

| #   | 标题                                                                                                                  | 首作者           | 年份   | 类别       | 难度   | 时长   | arXiv                                          | PDF |
| --- | ------------------------------------------------------------------------------------------------------------------- | ------------- | ---- | -------- | ---- | ---- | ---------------------------------------------- | --- |
| 1   | [[01_Jailbreaking_LLMs_VLMs_2601.03594.pdf\|Jailbreaking LLMs & VLMs: Mechanisms, Evaluation, and Unified Defense]] | Zejian Chen   | 2026 | 📋综述     | 🟢入门 | 2-3h | [2601.03594](https://arxiv.org/abs/2601.03594) | ✅   |
| 3   | [[03_Survey_Adversarial_Robustness_MLLMs_2503.13962.pdf\|Survey of Adversarial Robustness in MLLMs]]                | Chengze Jiang | 2025 | 📋综述     | 🟢入门 | 2-3h | [2503.13962](https://arxiv.org/abs/2503.13962) | ✅   |
| 4   | [[04_Evolving_Safety_Landscape_MLLMs_2608.07535.pdf\|Evolving Safety Landscape of MLLMs]]                           | Xi Li         | 2026 | 📋综述     | 🟢入门 | 2h   | [2608.07535](https://arxiv.org/abs/2608.07535) | ✅   |
| 6   | [[06_AdaShield_2403.09513.pdf\|AdaShield: Adaptive Shield Prompting]]                                               | Yu Wang       | 2024 | 🟢提示级    | 🟡进阶 | 2-3h | [2403.09513](https://arxiv.org/abs/2403.09513) | ✅   |
| 7   | [[07_Immune_ECSO_2411.18688.pdf\|Immune (ECSO): Inference-Time Alignment]]                                          | S.S. Ghosal   | 2024 | 🟢推理时    | 🟡进阶 | 2-3h | [2411.18688](https://arxiv.org/abs/2411.18688) | ✅   |
| 9   | [[09_CIDER_2407.21659.pdf\|CIDER: Cross-modality Information Check]]                                                | Yue Xu        | 2024 | 🟡检测     | 🟡进阶 | 1-2h | [2407.21659](https://arxiv.org/abs/2407.21659) | ✅   |
| 13  | [[13_MAT_2405.18770.pdf\|MAT: Multimodal Adversarial Defense via One-To-Many]]                                      | Futa Waseda   | 2024 | 🔴对抗训练   | 🔴高阶 | 3-4h | [2405.18770](https://arxiv.org/abs/2405.18770) | ✅   |
| 16  | [[16_SafePTR_2507.01513.pdf\|SafePTR: Token-Level Jailbreak Defense]]                                               | Beitao Chen   | 2025 | 🔵Token级 | 🟡进阶 | 2h   | [2507.01513](https://arxiv.org/abs/2507.01513) | ✅   |

---

## P2 推荐（7 篇）

| # | 标题 | 首作者 | 年份 | 类别 | 难度 | 时长 | arXiv | PDF |
|---|------|--------|------|------|------|------|-------|-----|
| 2 | [[02_Adversarial_Defense_VLMs_Overview_2601.12443.pdf\|Adversarial Defense in VLMs: An Overview]] | Xiaowei Fu | 2026 | 📋综述 | 🟢入门 | 1-2h | [2601.12443](https://arxiv.org/abs/2601.12443) | ✅ |
| 5 | [[05_Adversarial_Diffusion_Across_Modalities_2606.26566.pdf\|Adversarial Diffusion Across Modalities]] | Abrar Alotaibi | 2026 | 📋综述 | 🟡进阶 | 2h | [2606.26566](https://arxiv.org/abs/2606.26566) | ✅ |
| 10 | [[10_HiddenDetect_2502.14744\|HiddenDetect: Hidden States Monitoring]] | Yilei Jiang | 2025 | 🟡检测 | 🟡进阶 | 2h | [2502.14744](https://arxiv.org/abs/2502.14744) | ✅ |
| 15 | [[15_MMCoA_2404.19287\|MMCoA: Multimodal Contrastive Adversarial Training]] | Wanqi Zhou | 2024 | 🔴对抗训练 | 🔴高阶 | 3h | [2404.19287](https://arxiv.org/abs/2404.19287) | ✅ |
| 17 | [[17_MMAligner_2608.05909\|MMAligner: Representation Calibration]] | Shenyi Zhang | 2026 | 🔵表示校正 | 🟡进阶 | 2h | [2608.05909](https://arxiv.org/abs/2608.05909) | ✅ |
| 18 | [[18_DTR_2505.17132\|DTR: Dynamic Token Reweighting]] | Tanqiu Jiang | 2025 | 🔵Token重加权 | 🟡进阶 | 2h | [2505.17132](https://arxiv.org/abs/2505.17132) | ✅ |
| 20 | [[20_OmniSafeBench-MM_2512.06589\|OmniSafeBench-MM: Unified Benchmark]] | Xiaojun Jia | 2025 | 🟣基准 | 🟢入门 | 1-2h | [2512.06589](https://arxiv.org/abs/2512.06589) | ✅ |

---

## P3 选读（5 篇）

| # | 标题 | 首作者 | 年份 | 类别 | 难度 | 时长 | arXiv | PDF |
|---|------|--------|------|------|------|------|-------|-----|
| 8 | [[08_SHIELD_2510.13190\|SHIELD: Classifier-Guided Prompting]] | Juan Ren | 2025 | 🟢提示级 | 🟡进阶 | 1-2h | [2510.13190](https://arxiv.org/abs/2510.13190) | ✅ |
| 11 | [[11_BlueSuffix_2410.20971\|BlueSuffix: Reinforced Blue Teaming]] | Yunhan Zhao | 2024 | 🟡蓝队 | 🟡进阶 | 2h | [2410.20971](https://arxiv.org/abs/2410.20971) | ✅ |
| 12 | [[12_E2AT_2503.04833\|E²AT: Dynamic Joint Optimization]] | Liming Lu | 2025 | 🔴对抗训练 | 🔴高阶 | 3h | [2503.04833](https://arxiv.org/abs/2503.04833) | ✅ |
| 14 | [[14_Sim-CLIP_Plus_2409.07353\|Sim-CLIP+: Robust Encoder]] | M.Z. Hossain | 2024 | 🔴鲁棒编码器 | 🟡进阶 | 2h | [2409.07353](https://arxiv.org/abs/2409.07353) | ✅ |
| 19 | [[19_SGM_2512.15052\|SGM: Neuron-Level Detoxification]] | Hongbo Wang | 2025 | 🔵神经元级 | 🔴高阶 | 2-3h | [2512.15052](https://arxiv.org/abs/2512.15052) | ✅ |

---

## 防御技术分类

| 类别         | 核心思路       | 代表方法                    | 优势/局限       |
| ---------- | ---------- | ----------------------- | ----------- |
| 🟢 提示级防御   | 输入前添加安全提示词 | AdaShield, SHIELD       | 无需训练；泛化有限   |
| 🟡 检测型防御   | 跨模态一致性校验   | CIDER, HiddenDetect     | 精准定位；可能被绕过  |
| 🔴 对抗训练    | 训练阶段引入对抗样本 | E²AT, MMCoA, MAT        | 从根源提升；计算成本高 |
| 🔵 推理时/表示级 | 修改中间表示或解码  | SafePTR, DTR, MMAligner | 轻量；依赖启发式    |
| 🟣 基准与评估   | 标准化攻击-防御评估 | OmniSafeBench-MM        | 促进比较；覆盖面有限  |

---

## 核心关键词索引

- **自适应提示防御**：#6 AdaShield, #8 SHIELD
- **推理时安全对齐**：#7 Immune/ECSO
- **跨模态检测**：#9 CIDER, #10 HiddenDetect
- **Token级防御**：#16 SafePTR, #18 DTR
- **多模态对抗训练**：#13 MAT, #15 MMCoA, #12 E²AT
- **鲁棒编码器**：#14 Sim-CLIP+
- **表示校正**：#17 MMAligner
- **神经元级去毒化**：#19 SGM
- **蓝队防御**：#11 BlueSuffix
- **评估基准**：#20 OmniSafeBench-MM

---

## 阅读进度
- [x] #1 Jailbreaking LLMs & VLMs (综述)
- [ ] #4 Evolving Safety Landscape (综述)
- [ ] #6 AdaShield (提示级)
- [ ] #7 Immune/ECSO (推理时)
- [ ] #9 CIDER (检测)
- [ ] #13 MAT (对抗训练)
- [ ] #16 SafePTR (Token级)
- [ ] #2 Adversarial Defense Overview (综述)
- [ ] #5 Adversarial Diffusion Survey (综述)
- [ ] #10 HiddenDetect (检测)
- [ ] #15 MMCoA (对抗训练)
- [ ] #17 MMAligner (表示校正)
- [ ] #18 DTR (Token重加权)
- [ ] #20 OmniSafeBench-MM (基准)
- [ ] #8 SHIELD (提示级)
- [ ] #11 BlueSuffix (蓝队)
- [ ] #12 E²AT (对抗训练)
- [ ] #14 Sim-CLIP+ (鲁棒编码器)
- [ ] #19 SGM (神经元级)

