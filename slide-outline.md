# Slide Outline

## Meta
- Topic: 用模型编辑做 LLM 越狱防御——DELMAN 精读与改进猜想（组会汇报）
- Scenario: 组会汇报，听众为导师与同门；讲 15–20 分钟
- Content Source: provided（用户笔记：知识编辑、DELMAN 精读、完整 idea）
- Research: skip — 内容全部来自用户提供的笔记，无需外部检索
- Style: 学术简洁风 — 米白/纸感底色 + 墨蓝正文 + 单一学术蓝强调色；衬线感标题层级、克制装饰（细线、角标、页码）、图表用扁平 SVG
- Slide Count: 12
- Generated At: 2026-10-15T10:46:00+08:00

## Source Materials
- notes/深度学习/知识编辑/知识编辑.md（三大流派、ROME 原理、攻防两用视角）
- notes/论文阅读/LLM Safety/002-DELMAN...md（动机、方法四步、实验数据、不足）
- notes/论文阅读/LLM Safety/007-编辑防御攻防完整idea.md（三颗雷、两条猜想、Pilot 计划）

## Slide-by-Slide Outline
1. **Slide 1 — 封面** — 《用模型编辑做 LLM 越狱防御：DELMAN 精读与我的改进猜想》+ 组会汇报 + 日期 | Layout: L01 cover | Chart: no
2. **Slide 2 — 目录** — 三部分导航 + 一句话路线图 | Layout: L02 agenda | Chart: no
3. **Slide 3 — 为什么需要模型编辑** — 痛点/定义/三条评价标准 R·G·L | Layout: 三栏卡片 | Chart: no
4. **Slide 4 — 三大流派与 ROME** — 流派对比表 + ROME 两步（定位/闭式解改写）| Layout: 表格+流程 | Chart: no
5. **Slide 5 — 攻防两用：同一把手术刀** — ROME 正用修事实 / DELMAN 刻拒绝钢印 / JailbreakEdit 刻恶意后门；"谁的语义强谁赢" | Layout: 三节点对峙 | Chart: no
6. **Slide 6 — DELMAN：研究动机** — 部署后被越狱、重训代价大；最小修改/精准定位/动态更新；一句话概括 0.4h+200 条+ASR 6.7% | Layout: 左文右数据卡 | Chart: no
7. **Slide 7 — DELMAN 方法：四步流水线** — 抽有害 token → 中性句平均造 k* → 想象替换+KL 优化 v* → 闭式解焊入 W_down；防御逻辑一句话 | Layout: L 横向流程图(SVG) | Chart: yes
8. **Slide 8 — 实验结论** — ASR 对比柱状图（Vicuna：原模型 92 / LoRA 23.2 / SafeDecoding 10.7 / LED 8.8 / DELMAN 6.7）+ 泛化/效率/消融要点 | Layout: 左图右要点 | Chart: yes
9. **Slide 9 — 不足：三颗雷**（深色转折页）— ① key 上下文无关可绕过 ② KL 只见过一种句式 ③ 无 over-refusal 测量 | Layout: L 深色三雷排布 | Chart: no
10. **Slide 10 — 猜想一：误伤是设计必然** — 假设陈述 + 验证路径（按语境分层的 over-refusal 基准）| Layout: 假设卡+路径 | Chart: no
11. **Slide 11 — 猜想二：语境选择 key** — 双 key（k_H 有害脸 / k_B 良性脸）+ 平移编辑；门前不开 vs KL 兜底对比图 | Layout: 示意图(SVG) | Chart: yes
12. **Slide 12 — 下一步计划** — 两周 Pilot（复现+变体攻击集+两个判决数→G0）+ 预期贡献 + 收尾金句"坑在哪，贡献就在哪" | Layout: 时间轴+清单 | Chart: no

## Visual Rhythm Notes
- Chart-bearing slides: 7, 8, 11
- Dark accent slide: 9（全场枢纽，深色强调）
- Slides are viewed on large screens, often from a distance. Small text is unreadable. Every text element must be sized generously and fill its container space rather than floating in emptiness.
- 学术简洁风：装饰克制但每页保留 ≥2 个装饰元素（细线分隔、序号角标、淡色几何底纹）；数据用扁平 SVG 图表而非纯文字。
