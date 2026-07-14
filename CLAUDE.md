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
│   └── <主题>/             # 主题目录（如 自注意力机制/）
│       ├── <主题>.md            # 主题主笔记
│       ├── Q&AHW*.md            # 该主题相关的作业问答（如 Q&AHW4.md）
│       └── assets/             # 该主题笔记的附件（图片等）
├── tools/               # 工具笔记
│   └── pytorch 怎么用.md        # PyTorch 使用指南
├── reflections/         # 工作 vs 学术思考对比
├── struggles/           # 踩坑记录
├── Excalidraw/          # 手绘图（obsidian-excalidraw-plugin）
├── assets/              # Obsidian 附件默认存放处
└── README.md            # 成长轨迹
```

注意：课程作业子目录按**主题**命名（如 `01 Introduction`、`05 Transformer`），而非 `HW1`，与下表中的作业编号是映射关系。

## 常用命令

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

笔记按**主题目录**组织：每个主题在 `notes/` 下建一个同名目录，主题主笔记、关联的作业问答、附件都放该目录下。

目录结构示例（以「自注意力机制」主题为例）：

```
notes/自注意力机制/
├── 自注意力机制.md     # 主题主笔记
├── Q&AHW4.md          # 该主题相关的作业问答（HW4 = Self-Attention）
└── assets/            # 该主题笔记用到的图片等附件
```

约定细则：

- **主题主笔记**：命名为 `<主题名>.md`，放在 `notes/<主题名>/` 下。
- **作业问答**：与该主题相关的作业问答命名为 `Q&AHW<编号>.md`，放在对应主题目录下（而非 `notes/Q&A/`）。`notes/Q&A/` 仅存放跨主题或尚未归类的通用作业问答（如 HW1、HW2 这类已学过但未单独建主题目录的）。
- **附件**：主题目录下建 `assets/` 子目录存放该主题笔记引用的图片（遵循 obsidian-custom-attachment-location 插件约定）。
- 当用户提问学习相关问题时，先判断属于哪个主题：若主题目录已存在，把问答追加到该目录下的 `Q&AHW*.md`；若不存在且问题足够独立，再新建主题目录。
- 笔记格式参考现有文件，使用清晰的标题和代码示例。
- 站内链接使用 `[[Wiki Link]]` 而非纯路径。

## 用户习惯

- 每周至少一次 commit
- 不追求完美，记录真实学习过程
- 接受"慢工出细活"的学术节奏
