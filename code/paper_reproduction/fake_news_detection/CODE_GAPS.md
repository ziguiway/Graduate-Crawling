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
- Added `LLM-MFEFND/utils/models_mae.py` with a lightweight `forward_ying` fallback for smoke tests.
- Patched `LLM-MFEFND/LLM_MFEFND.py` so CPU-only forward smoke tests can run without CN-CLIP and MAE checkpoint files.
- Added `scripts/smoke_llm_mfefnd_forward.py`; it currently reaches `forward_ok` on FineFake auxiliary data.

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

- image preprocessing and batching
- CN-CLIP image/text encoding input format
- MAE model wrapper and `forward_ying`
- hierarchical progressive transformer fusion in `fusion_img_text`
- LLM-generated background/comment preprocessing

## Practical next build order

1. Copy/adapt basic layers and utilities from MDFEND.
2. Create a multimodal dataloader that emits the keys expected by `LLM_MFEFND.forward`.
3. Build a lightweight image encoder fallback for smoke tests. Done for the development path.
4. Add the full MAE/CN-CLIP path after the dataloader is stable.
5. Reproduce paper preprocessing and splits.
