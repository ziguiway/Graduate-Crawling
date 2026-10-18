# delman_key_sampling — 第一步"采样造 key"全套脚本

对齐 [[notes/论文阅读/LLM Safety/008-第一步-采样造key方案.md]]。
把 DELMAN 的中性句 k\* 换成双语境 key：**k_edit = k_H − α·(k_B − k\*)**。

## 文件结构

```
delman_key_sampling/
├── data/
│   └── edit_cases.json        # 200 条 DELMAN 编辑用例（HarmBench 抽样，已就位）
└── scripts/
    ├── 01_build_sentence_pools.py   # 线1/2/4：有害句、良性句、KL 校准集
    ├── 02_extract_activations.py    # 线1/2 续：提取 k_H / k_B（+逐条激活）
    ├── 03_separability_and_kedit.py # 闸1：可分性分析 + k_edit 构造（α=0.5/1.0）
    └── README.md
```

## 服务器上运行顺序

```bash
# 0. 环境（torch_py310 已有 transformers/torch；vLLM server 若已起在 8000 端口，01 直接可用）
conda activate torch_py310
cd ~/data/code/delman_key_sampling    # 拷到服务器后的路径

# 1a. 句子池（先生成句子；需要本地 72B 的 vLLM OpenAI server @ localhost:8000）
python scripts/01_build_sentence_pools.py \
    --cases data/edit_cases.json --outdir data/ \
    --rewrite-model Qwen/Qwen2.5-72B-Instruct \
    --benign-model  Qwen/Qwen2.5-72B-Instruct
# 没起 vLLM server 时的最小可用路径（先只跑通流程、跳过生成）：
#   python scripts/01_build_sentence_pools.py --cases data/edit_cases.json --outdir data/ --dry-run

# ★ 闸2：人工抽查 ~10% 良性句（head -100 data/benign_contexts.jsonl 过一遍）

# 1b. 激活提取（layer 必须与 DELMAN 编辑层对齐！先查 DELMAN 仓库 hparams 里 Qwen2.5-7B 的配置）
python scripts/02_extract_activations.py \
    --model Qwen/Qwen2.5-7B-Instruct --layer <对齐DELMAN的层号> \
    --harmful data/harmful_contexts.jsonl --benign data/benign_contexts.jsonl \
    --subjects data/subjects.json --outdir data/

# 1c. 可分性分析 + k_edit 构造
#     （activations_star.pt = DELMAN 原管线跑一遍中性句流程的 k* 产物；先用 --star-optional 跑分析也行）
python scripts/03_separability_and_kedit.py \
    --h data/activations_h.pt --b data/activations_b.pt \
    --star data/activations_star.pt --outdir data/
#     没有原版 k* 时：
#   python scripts/03_separability_and_kedit.py --h ... --b ... --outdir data/   # 会警告

# 看 Gate 结果：
less data/separability.md
```

## 判读（Gate 闸1）

| 结果 | 动作 |
|---|---|
| 多数类别 ratio ≥ 2，质心准确率 > 90% | **开跑 α 编辑**（第二步），危险词清单交给刹车兜底 |
| ratio 普遍 1.5–2 | 谨慎开跑，α 先用 0.5，重点盯 ASR |
| B1/B3 都不可分（< 1.5） | 先跑 5 词端到端 sanity check 再决定；可能转 analysis 体裁 |

## 三个脚本的边界（防误用）

- **01** 只产句子清单，**不碰模型权重**；
- **02** 只读原模型做前向，**不改任何东西**；`--layer` 必须与 DELMAN 编辑层一致（去 DELMAN 仓库 hparams/ 里确认 Qwen2.5-7B 的 layer 配置）；
- **03** 纯张量运算 + 统计，CPU 就能跑；产出 `k_edit.pt` 交给第二步（改 DELMAN 管线 K_E 构造）。

## 已知简化（跑通第一版后再升级）

1. 02 脚本没存逐条有害激活（只存 k_H 均值）→ 03 的 d_within_H 用 B 组近似。如需精确值，02 加 `--save-items-h` 再跑一次；
2. 良性句生成的模板是 4 类骨架，生成质量靠人工抽查把关（闸2），第一批跑完建议按抽查结果改模板再全量；
3. k_edit 目前产出 α=0.5/1.0 两档 + 保留 k_H/k_B/k\* 全部中间量，α 扫描（如 0.25/0.75）在第二步管线里加个循环即可。
