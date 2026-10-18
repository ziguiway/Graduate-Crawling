# 实验方案 · 可分性测量（Gate 闸1 判决数）

> 服务器：192.168.207.233:32882 (root)，DELMAN 代码在 /root/data/code/DELMAN
> GPU：Quadro RTX 8000 48G｜conda env：**delman**（torch 2.4.0+cu121, transformers 4.49.0）
> 实验目录：/root/data/code/delman_key_sampling/（句池已上传）
> 日志与记录：/root/data/code/delman_key_sampling/logs/ 与 EXPERIMENT_LOG.md

## 0. 服务器侦察结论（2026-10-18）

- DELMAN 编辑管线确认：Qwen2.5-7B 的 hparams **layers = [7,8]**（`hparams/Qwen2.5-7B-Instruct.json`），
  即 **k\* 是在 layer 7 和 layer 8 的 down_proj 输入上计算的**，v\* 优化在 layers[-1]=8。
- `fact_token: "subject_last"` → **subject 的最后一个 token** 是 key 的定位点（不是首 token）。
- k\* 的语境模板（`get_context_templates`）：`["{}"]` + 4 条半随机前缀句
  （"The first thing I noticed is a very strong,. {}" 等）→ 与论文附录 C.2 一致。
- k\* 计算调用链：`compute_k_star` → `get_module_input_output_at_words` → `rome/repr_tools.get_reprs_at_word_tokens`
  （track="both"，返回 module 的 **input**（即 l* 层 MLP 输入）与 output）。**key 用的是 l_input**。
- 已有产出：`results/DELMAN_qwen2.5_full/`（Qwen2.5-7B 全量编辑后权重）可直接加载做对比。
- 系统日期异常（2024?），日志按 UTC+8 手写时间戳。

## 1. 实验目的（一句话）

测量"同一有害词在有害语境 vs 良性语境下，编辑层 MLP 输入表示（key 空间）的可分性"——
决定 k_edit = k_H − α(k_B − k\*) 这条改进路线开不开跑。

## 2. 对齐决策（必须与 DELMAN 管线严格一致）

| 参数 | 取值 | 依据 |
|---|---|---|
| 提取层 | **layer 8**（hparams.layers[-1]，v_loss_layer 语义对齐）+ 辅助 layer 7 | hparams |
| 表示定义 | **layer 的 MLP down_proj 的输入**（与 k\* 的 l_input 一致） | repr_tools track="both" |
| subject 定位 | **subject_last**（最后一个 token） | fact_token |
| 提取模型 | **原模型**（Qwen2.5-7B-Instruct 未编辑） | 方案 §2.2 |
| 文本格式 | 与 DELMAN 编辑输入一致：直接用句子文本（无 chat 模板包裹——k\* 计算时 prompt_adv 也是裸文本 format 进 context template） | compute_k_star 实现 |

⚠ 02 脚本初版按"自算 σ(W_gate·x)⊙(W_up·x)"实现，与 DELMAN 的 down_proj **输入**定义不一致 →
本次实验改用与 DELMAN 完全相同的提取路径（直接 import DELMAN 的 repr_tools），消除 off-by-one 风险。

## 3. 数据（已就位）

- `data/edit_cases.json`：200 条（177 subject）
- `data/harmful_contexts.jsonl`：1063 条（每词 6 条：原 query + 5 条意图陈述模板改写）
- `data/benign_contexts.jsonl`：1009 条（每词 5-6 条：B1 新闻/B2 比喻/B3 教育/B4 防御）
- 注：H 组含模板改写，句式与原 query 不同——预期使 k_H 组内方差增大，ratio 偏保守，可接受。

## 4. 实验步骤

1. `scripts/02_extract_activations_delman.py`（新写，复用 DELMAN repr_tools）：
   对 177 词 × ~12 句（6 有害 + 6 良性），在 layer 7 和 layer 8 分别提取 subject_last 位置的
   down_proj 输入表示 → `activations_h.pt` / `activations_b.pt` / `activations_star_repro.pt`
   （顺手用 DELMAN 的 context_templates 复算原版 k\*，供 k_edit 锚点用，一举两得）。
2. `scripts/03_separability_and_kedit.py`（本地已有，微调适配）：
   输出 `separability.md/json` + `k_edit.pt`（α=0.5/1.0 两档）。
3. 全程 tee 到 `logs/extract.log`、`logs/separability.log`。

## 5. 判读标准

- ratio = ‖k_H − k_B‖ / mean(d_within_B, d_within_H)，按词报；类别聚合看分布。
- ratio ≥ 2：可分良好；1.5–2：谨慎；< 1.5：危险词（列表交刹车兜底）。
- 质心二分类准确率 >90%：全局可分性成立。
- 分层预期：科学名词类（B3 教育）最好；观点类 subject（Great Leap Forward 等）与动词类最差。

## 6. 实验记录规范

- `EXPERIMENT_LOG.md`（本目录）：每次运行记录时间/命令/产出/关键数字。
- `logs/`：脚本完整 stdout/stderr。
- 产出 .pt 文件记录 sha256 于日志，防篡改可追溯。
