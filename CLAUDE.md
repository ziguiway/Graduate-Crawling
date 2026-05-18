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
Graduate-Crawling/
├── papers/              # 论文阅读笔记
│   └── 研0/             # 科研入门笔记
├── code/                # 实验代码
│   └── Lhy_Machine_Learning/    # 李宏毅课程作业
│       ├── 2021 ML/             # 2021春季课程（15个作业）
│       ├── 2022 ML/             # 2022春季课程
│       └── 2023 ML/             # 2023春季课程
├── tools/               # 工具笔记
│   ├── pytorch 怎么用.md        # PyTorch 使用指南
│   └── Python Q&A.md            # Python 问题汇总
├── reflections/         # 工作 vs 学术思考对比
├── struggles/           # 踩坑记录
└── README.md            # 成长轨迹
```

## 常用命令

```bash
# 安装依赖（使用 uv）
uv sync

# 运行 Jupyter Notebook
uv run jupyter notebook

# 运行 Python 脚本
uv run python main.py

# Obsidian CLI 操作笔记
obsidian --help                    # 查看帮助
obsidian note create <title>       # 创建笔记
obsidian note open <file>          # 打开笔记
obsidian search <query>            # 搜索笔记
```

## Python 环境

- Python 版本：3.12
- 包管理：uv
- 主要依赖：PyTorch、transformers、datasets、scikit-learn、matplotlib、wandb

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

当用户提问学习相关问题时，将问题和解答追加到 `tools/Python Q&A.md`。

笔记格式参考现有文件，使用清晰的标题和代码示例。

## 用户习惯

- 每周至少一次 commit
- 不追求完美，记录真实学习过程
- 接受"慢工出细活"的学术节奏
