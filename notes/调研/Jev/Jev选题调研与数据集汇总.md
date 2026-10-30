---
title: Jev选题调研与数据集汇总
date: 2026-10-29
tags:
  - Jev
  - 科研选题
  - 调研
status: 初步调研
---

# Jev 选题调研与数据集汇总

调研对象：TypeSafe 的 Jev（System One 决策模型）。整理自 2026-10-28 至 2026-10-29 的讨论。

> [!info] 调研范围
> “未检索到”指本次未找到直接对应的 Jev 专项论文或项目，不代表确定无人做过；尚未逐项审计所有综合基准。已有工作状态是本轮检索结论，数据集链接用于后续核实。推荐排序属于选题建议，不是模型性能结论。

## 方向与数据集对照表

| 选题方向                   | 具体任务                       | 数据集                                                                                                                                                                                                                                   | 已有 Jev 工作情况                              | 建议                         |
| ---------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | -------------------------- |
| **科学传播失真检测**           | 比较论文与报道，判断因果、确定性、普遍性等是否被夸大 | [Science Communication Distortions](https://www.uni-bamberg.de/en/nlproc/resources/sciencecommdistortion/)：1,600 组成对实例；[Scientific Exaggeration Detection](https://huggingface.co/datasets/copenlu/scientific-exaggeration-detection) | 未检索到直接对应的专项工作；有相近的证据核查研究                 | **首选**：任务明确，适合研究原子判断与概率校准  |
| **多语言委婉语检测**           | 判断目标词在语境中是否为委婉表达           | [Multilingual Euphemism Benchmark](https://lrec.elra.info/lrec2026-main-579)：整合七语言资源                                                                                                                                                  | 未检索到直接对应的专项工作                            | **第二选择**：适合研究跨语言泛化与文化差异    |
| **多语言刻板印象检测**          | 判断文本是否包含特定文化背景下的刻板印象       | [SHADES / BiasShades](https://huggingface.co/datasets/LanguageShades/BiasShades)                                                                                                                                                      | 已有 Jev 中英双语偏见测试；未检索到直接用于多语言文本刻板印象检测的系统论文 | 区分“检测文本中的刻板印象”与“评估模型自身偏见”  |
| **虚假新闻／事实核查**          | 判断新闻主张是否得到外部证据支持           | [LLM-AggreFact / MiniCheck](https://github.com/Liyan06/MiniCheck)；FakeNewsNet、LIAR 可作为扩展候选，需另核对数据可用性                                                                                                                                  | 已有 Fake / Real 应用和 Jev RAG 主张核查预印本       | 不建议只替换模型；应聚焦证据不足、错误放行或跨域泛化 |
| **恶意 Agent Skills 检测** | 检测技能包中的恶意代码、提示注入与隐蔽操作      | MalSkillBench，可从 [jev-skillbench](https://github.com/MohibShaikh/jev-skillbench) 查看获取与评估说明                                                                                                                                            | 已有直接实验：Jev 检测及大模型级联复核                    | 已有基线，需要新的方法或鲁棒性问题          |
| **传统恶意软件检测**           | 对 PE／APK 文件或行为特征进行恶意性判断    | EMBER、SOREL-20M、Drebin 为候选；本轮尚未核实获取条件                                                                                                                                                                                                 | 未检索到直接对应的 Jev 专项论文                       | 需要程序分析基础，数据与特征处理成本较高       |
| **审稿—回复关系识别**          | 判断作者回复是在承认、反驳，还是说明采取行动     | [DISAPERE](https://aclanthology.org/2022.naacl-main.89/)：506 组审稿—回复对，约 2 万句专家标注                                                                                                                                                       | 未检索到直接对应的专项工作                            | 小众、标注现成，适合细粒度立场与上下文研究      |
| **居高临下式语言检测**          | 识别对弱势群体隐含优越感或怜悯的表达         | [Don’t Patronize Me!](https://aclanthology.org/2020.coling-main.518/)                                                                                                                                                                 | 未检索到直接对应的专项工作；与内容审核相邻                    | 适合研究正常关怀与冒犯之间的误判           |
| **语境中的预设推断**           | 判断句子暗含的事实是否成立、是否被上下文取消     | [NOPE](https://aclanthology.org/2021.conll-1.28/)：覆盖十类预设触发词                                                                                                                                                                           | 未检索到直接对应的专项工作                            | 偏语言学，适合研究否定、语境反转与概率一致性     |
| **学术引用意图分类**           | 区分背景引用、方法使用、结果比较等用途        | [SciCite](https://github.com/allenai/scicite)、[MultiCite](https://github.com/allenai/multicite)                                                                                                                                       | 未检索到直接对应的专项工作                            | 容易起步，但仅增加 Jev 基线的创新性有限     |

## 推荐顺序

科学传播失真检测 ＞ 多语言委婉语检测 ＞ 审稿—回复关系识别。

首选“基于原子判断与置信度校准的科学传播失真检测”。先做英文、成对文本、细粒度判断，多语言扩展留到核心实验成立之后。贡献应落在任务分解、跨域泛化或可靠拒判，不能仅依赖“首次使用 Jev”。

## 已有 Jev 工作与核查入口

- 事实核查应用：[Fake / Real](https://github.com/DansiDanutz/fake-real-jev)。
- 证据支持性核查预印本：[Jev-as-a-Judge for RAG Claim Verification](https://arxiv.org/abs/2610.20634)。
- 中英双语模型偏见测试：[Biased-Decisions China](https://biased-decisions.anth.us/china/)，属于模型偏见评估，不是文本刻板印象检测。
- 英阿文化价值观预印本：[Calibrated to Whom?](https://arxiv.org/abs/2609.36399)。
- 恶意 Skills 基准实验：[jev-skillbench](https://github.com/MohibShaikh/jev-skillbench)，是实验仓库，不能直接列作已发表论文。
- 恶意代码扫描工具：[is-malicious](https://github.com/luantak/is-malicious)、[jev-security-scan](https://github.com/win4r/jev-security-scan)。

## 开题前核对

- 核对数据集许可、标签定义、获取条件与原始划分。
- 对选定任务补做专项查新，检查论文、代码和综合基准。
- 按源论文或数据来源分组划分，避免相似实例跨集合泄漏。
- 固定 Jev 模型版本，记录实际 API 返回的版本、成本与失败请求。
- 校准、阈值选择与最终测试分开；级联实验增加相同预算的随机转交对照。
