# Graduate-Crawling 项目长期笔记

## 研究方向：模型编辑做 LLM 越狱防御（DELMAN 改进）

- 用户 Pilot 实测（2026-10）：对 DELMAN 编辑前后模型做同义词/谐音/base64 等表面变体攻击，**原模型与编辑后模型都拒绝，均无法绕过** → 表面改写不是有效攻击面；真正缺口在「避开 GPT-4 触发词表」与「语境变化」（支撑猜想二：语境可分性）。
- 组会 PPT：Ardot 文件 https://ardot.tencent.com/file/736613351564270 （18 页学术简洁风：技术 3 页 + DELMAN 精读含实验 4 页 + 猜想 3 页 + 3 张过渡页）。大纲存项目根目录 slide-outline.md。
- 待办：AB/JBB/MI 全称（AdvBench/JailbreakBench/MaliciousInstruct?）需对照 DELMAN 原 PDF 核实。
