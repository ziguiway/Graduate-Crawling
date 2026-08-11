# LLM-MFEFND Code Gap Map

The public `LLM-MFEFND` repository is not directly runnable. It only contains:

- `main.py`
- `LLM_MFEFND.py`
- partial CSV files in `data/`

## Missing entrypoints

- `grid_search.py`
- multimodal dataloader
- train/validation/test split builder for the paper setting

## Missing modules referenced by `LLM_MFEFND.py`

- `utils.models_mae`
  - required for `mae_vit_base_patch16`
  - required method: `forward_ying`
- `utils.layers`
  - required classes/functions: `TokenAttention`, `clip_fuion`
- `utils.utils`
  - required functions/classes: `data2gpu`, `Averager`, `metrics`, `Recorder`, `metrics1`, `visualize_tsne`
- local `layers.py`
  - required classes: `MLP`, `MaskAttention`, `SelfAttentionFeatureExtract`
- local `pivot.py`
  - required classes: `TransformerLayer`, `MLP_trans`

## Reconstructed so far

- Added `LLM-MFEFND/layers.py` from MDFEND base layers.
- Added `LLM-MFEFND/utils/utils.py` from MDFEND and extended it with `metrics1`, dict-batch `data2gpu`, and `visualize_tsne` placeholder.
- Added `LLM-MFEFND/utils/layers.py` with `TokenAttention` and `clip_fuion`.
- Added `LLM-MFEFND/pivot.py` with basic `TransformerLayer` and `MLP_trans`.
- Added `HierarchicalProgressiveTransformer` in `pivot.py` with the paper's
  five-stage progressive fusion, residual averaging, and four-round iteration.
- Added `OfficialHierarchicalProgressiveTransformer` to mirror the public
  `data` branch's `fusion_img_text()` implementation.
- Added `LLM-MFEFND/utils/models_mae.py` with a lightweight `forward_ying` fallback for smoke tests.
- Added an optional real `timm` ViT-Base MAE wrapper and loaded the official
  `mae_pretrain_vit_base.pth` checkpoint from `pretrained/`.
- Patched `LLM-MFEFND/LLM_MFEFND.py` so CPU-only forward smoke tests can run without CN-CLIP and MAE checkpoint files.
- Added explicit `LLM_MFEFND_REAL_BACKBONES=1` mode for CN-CLIP tokenizer,
  image preprocessing, CN-CLIP weights, and MAE weights.
- Added `scripts/smoke_llm_mfefnd_forward.py`; it currently reaches `forward_ok` on FineFake auxiliary data.
- Added `scripts/smoke_hpt.py` and `scripts/smoke_llm_mfefnd_train_step.py` for HPT
  structure and trainability checks.
- Added `scripts/smoke_interaction.py` for bidirectional cross-attention and mask checks.
- Added `scripts/smoke_official_hpt.py` for the official HPT slot/order contract.

## Recoverable from MDFEND

The `sources/MDFEND-Weibo21` repo can provide:

- `MLP`
- `MaskAttention`
- `SelfAttentionFeatureExtract`
- basic `Recorder`
- basic `Averager`
- a reference `Run` training scaffold
- a reference text-only dataloader pattern

## Not directly recoverable from MDFEND

These are specific to LLM-MFEFND and need to be reconstructed:

- complete image preprocessing and batching for the unreleased full dataset
- complete CN-CLIP image/text encoding input files for the unreleased full dataset
- full-data training and result verification
- isolated tests and ablations for strict bidirectional news-background/comments interaction
- exact paper-side preprocessing and complete hierarchical fusion configuration
- LLM-generated background/comment preprocessing

## Practical next build order

1. Copy/adapt basic layers and utilities from MDFEND.
2. Create a multimodal dataloader that emits the keys expected by `LLM_MFEFND.forward`.
3. Build a lightweight image encoder fallback for smoke tests. Done for the development path.
4. Add the full MAE/CN-CLIP path after the dataloader is stable.
5. Reproduce paper preprocessing and splits.
