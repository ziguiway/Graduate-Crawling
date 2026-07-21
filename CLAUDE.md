# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目背景

这是一个研0学生的深度学习知识库。用户工作两年后重返校园读研，从"职场老手"变成"读研萌新"，记录学习爬行之路。

**用户画像：**
- 研0学生，正在学习深度学习
- 工作两年后重返校园，有编程基础但学术经验不足
- 主要学习资源：李宏毅机器学习课程
- 使用 Obsidian 管理知识库（已安装 Obsidian CLI）

## 开学前目标（2026.09）

- 提升英语论文阅读效率
- 重拾数学基础（线代、概率）
- 明确兴趣方向
- 调整心态：从"快速交付"切换到"长期深耕"

## 目录结构

```
Graduate-Crawling/          # 同时是 Obsidian vault 根目录
├── papers/              # 论文阅读笔记
│   └── 研0/             # 科研入门笔记
├── code/                # 实验代码
│   └── Lhy_Machine_Learning/    # 李宏毅课程作业（uv 项目根，pyproject.toml 在此）
│       ├── 2021 ML/             # 2021春季课程（按主题命名子目录，如 01 Introduction）
│       ├── 2022 ML/             # 2022春季课程
│       └── 2023 ML/             # 2023春季课程
├── notes/               # 学习笔记（按主题分目录）
│   ├── Q&A.md               # 作业问答索引
│   ├── Q&A/                 # 通用作业问答（按作业编号分，HW1.md ...）
│   ├── 深度学习/            # 深度学习主题目录
│   │   └── <主题>/             # 主题目录（如 自注意力机制/、Transformer/）
│   │       ├── <主题>.md            # 主题主笔记
│   │       ├── Q&AHW*.md            # 该主题相关的作业问答（如 Q&AHW4.md）
│   │       └── assets/             # 该主题笔记的附件（图片等）
│   ├── NLP基础编程练习/    # 导师布置的 NLP 编程练习（11 个作业 + 扩展）
│   │   ├── NLP基础编程练习.md   # 任务总览 + 逐个任务详情 + 进度跟踪
│   │   └── assets/                # 任务相关附件
│   ├── 科研入门/           # 科研入门相关笔记
│   ├── Loss函数解读.md     # 损失函数专题
│   ├── 数据分析总览.md     # 数据分析专题
│   └── 特征分析：如何找出关键特征.md  # 特征工程专题
├── books/               # PDF 教材（李宏毅 LeeDL 教程 part1/part2、Happy-LLM）
├── tools/               # 工具笔记
│   └── pytorch 怎么用.md        # PyTorch 使用指南
├── reflections/         # 工作 vs 学术思考对比
├── struggles/           # 踩坑记录
├── Excalidraw/          # 手绘图（obsidian-excalidraw-plugin）
├── assets/              # Obsidian 附件默认存放处
├── pyproject.toml       # 根 uv 项目（极简，仅 numpy；非课程作业用）
└── README.md            # 成长轨迹
```

注意：课程作业子目录按**主题**命名（如 `01 Introduction`、`05 Transformer`），而非 `HW1`，与下表中的作业编号是映射关系。

## 常用命令

> 本仓库有**两个** `pyproject.toml`：仓库根的（极简，仅 numpy，供非课程脚本用）和 `code/Lhy_Machine_Learning/` 下的（课程作业的真实依赖）。**课程作业相关命令必须在 `code/Lhy_Machine_Learning/` 下执行**，否则会用到错误的依赖集。

uv 项目根在 `code/Lhy_Machine_Learning/`（`pyproject.toml` 所在处），下列 uv 命令需在该目录下执行：

```bash
cd code/Lhy_Machine_Learning

# 安装依赖（使用 uv）
uv sync

# 运行 Jupyter Notebook
uv run jupyter notebook

# 运行 Python 脚本
uv run python <script.py>

# 运行单个 .ipynb（不启动交互界面）
uv run jupyter nbconvert --to notebook --execute <notebook.ipynb>
```

本仓库无测试框架与 lint 配置；课程作业多为 `.ipynb` / `.py`，验证方式是直接运行 notebook 或脚本。

## Python 环境

- Python 版本：3.12
- 包管理：uv（项目根在 `code/Lhy_Machine_Learning/`）
- 主要依赖：PyTorch、transformers、datasets、scikit-learn、matplotlib、wandb

## Obsidian Vault

本仓库根目录即 Obsidian vault 根目录（`.obsidian/` 配置在此）。笔记均为 `.md` 文件，编辑时遵循 Obsidian 约定：

- 附件（图片等）默认存入 `assets/`（由 obsidian-custom-attachment-location 插件配置）
- 手绘示意图存入 `Excalidraw/`
- 站内链接使用 `[[Wiki Link]]` 而非纯路径
- 已安装 obsidian-git 插件，用户习惯每周至少一次 commit

## 课程作业结构

李宏毅课程作业按主题组织：

| 作业 | 主题 | 内容 |
|------|------|------|
| HW1 | Regression | COVID-19 预测 |
| HW2 | Classification | 分类任务 |
| HW3 | CNN | 图像分类 |
| HW4 | Self-Attention | 序列处理 |
| HW5 | Transformer | Seq2Seq |
| HW6 | GAN/Diffusion | 生成模型 |
| HW7 | BERT | 自监督学习 |
| HW8 | Auto-encoder | 异常检测 |
| HW9 | Explainable AI | 可解释性 |
| HW10 | Adversarial Attack | 对抗攻击 |
| HW11 | Domain Adaptation | 域适应 |
| HW12 | RL | 强化学习 |
| HW13 | Network Compression | 模型压缩 |
| HW14 | Life-long Learning | 终身学习 |
| HW15 | Meta Learning | 元学习 |

## 知识库笔记约定

笔记按**主题目录**组织：每个主题在 `notes/深度学习/` 下建一个同名目录，主题主笔记、关联的作业问答、附件都放该目录下。

目录结构示例（以「自注意力机制」主题为例）：

```
notes/深度学习/自注意力机制/
├── 自注意力机制.md     # 主题主笔记
├── Q&AHW4.md          # 该主题相关的作业问答（HW4 = Self-Attention）
└── assets/            # 该主题笔记用到的图片等附件
```

约定细则：

- **主题主笔记**：命名为 `<主题名>.md`，放在 `notes/深度学习/<主题名>/` 下。
- **作业问答**：与该主题相关的作业问答命名为 `Q&AHW<编号>.md`，放在对应主题目录下（而非 `notes/Q&A/`）。`notes/Q&A/` 仅存放跨主题或尚未归类的通用作业问答（如 HW1、HW2 这类已学过但未单独建主题目录的）。
- **附件**：主题目录下建 `assets/` 子目录存放该主题笔记引用的图片（遵循 obsidian-custom-attachment-location 插件约定）。
- 当用户提问学习相关问题时，先判断属于哪个主题：若主题目录已存在，把问答追加到该目录下的 `Q&AHW*.md`；若不存在且问题足够独立，再新建主题目录。
- 笔记格式参考现有文件，使用清晰的标题和代码示例。
- 站内链接使用 `[[Wiki Link]]` 而非纯路径。
- **笔记结尾必写「我的理解」**：每篇主题主笔记末尾都要有一个"我的理解 / 个人思考"小节，记录自己消化后的感悟、疑问、与后续章节的关联等，而不是只抄书本内容。
- **QA 每节固定结构**：`Q&AHW*.md` 里每条问答都按「**问题描述 → 回答 → 一句话记忆**」三段写，方便复习时先看问题描述自测、再看回答验证理解、最后用一句话记忆收尾。具体要求：
  - **问题描述**：开头先讲清"这个问题是怎么冒出来的"——通常来自实现代码时的某个困惑（比如"看到 `np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))` 这一行完全看不懂，为什么用 log、为什么有负号"），把困惑的具体代码行或概念写出来，不要只写一个干巴巴的标题。
  - **回答**：公式 / 直觉解释 / 对比表 / 代码片段都要有，**关键代码必须贴完整片段并标注关键行**（用 `★` 或注释高亮），让复习时一眼能定位到重点。
  - **一句话记忆**：用 `> ` 引用块收尾，一两句话浓缩本节核心，便于扫读时快速回忆。

## 用户习惯

- 每周至少一次 commit
- 不追求完美，记录真实学习过程
- 接受"慢工出细活"的学术节奏

## 学习进度

按李宏毅教材章节顺序记录已学完的内容，方便接续学习时定位。

| 日期 | 章节 | 主题 | 笔记路径 | 进度 |
|---|---|---|---|---|
| 2026-07-14 | 第 1 章 | 机器学习基础（回归/分类/损失/梯度下降/神经网络） | `notes/深度学习/机器学习基础/机器学习基础.md` | ✅ 全章 |
| 2026-07-14 | 第 6 章 | 自注意力机制（Seq 输入输出、注意力公式、多头、位置编码） | `notes/深度学习/自注意力机制/自注意力机制.md` | ✅ 全章 |
| 2026-07-14 | 第 7 章 | Transformer（Seq2Seq 应用、编码器结构、残差+LN） | `notes/深度学习/Transformer/Transformer.md` | ✅ 全章 |
| 2026-07-15 | 第 7 章 | Transformer 解码器（自回归、掩码注意力、NAR、cross-attention） | `notes/深度学习/Transformer/Transformer.md` | ✅ 7.4–7.5 完成；训练技巧待续 |
| 2026-07-17 | 第 7 章 | Transformer 训练过程与技巧（teacher forcing、复制、引导注意力、束搜索、加噪声、RL、计划采样） | `notes/深度学习/Transformer/Transformer.md` | ✅ 7.6–7.7 完成；位置编码公式/BLEU 待补 |
| 2026-07-16 | — | NLP 基础编程练习（导师李正华布置，11 个核心作业 + 4 个扩展） | `notes/NLP基础编程练习/NLP基础编程练习.md` | 🔄 作业 1–2 完成；作业 3 题目已记录，待开始 |

**续学线索**（下次接着学的起点）：
- 第 7 章 Transformer：位置编码公式（正余弦）的数学推导、BLEU 的具体计算公式（细节待补）
- NLP 基础编程练习：从**作业 3 HMM 词性标注**开始（作业 1–2 已完成），有监督版（极大似然 + 加 α 平滑 + Viterbi）+ 无监督版（Hard/Soft EM，5 个种子），等老师出 EM pdf 讲义再补推导
