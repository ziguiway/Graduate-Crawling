# LLM-MFEFND Reproduction Plan

## Current status

- Paper: `Multimodal fusion with LLM content via hierarchical progressive transformer for explainable fake news detection`
- Codebase: `code/paper_reproduction/fake_news_detection/LLM-MFEFND`
- Data:
  - `datasets/WeiBo21/` contains `train.pkl`, `val.pkl`, `test.pkl`
  - `datasets/FineFake/extracted/` contains `FineFake.pkl` and `Image/`

## What is already confirmed

- WeiBo21 split data is readable with pandas 1.5.3.
- FineFake original pickle is readable with pandas 1.5.3.
- FineFake image paths are complete: 16909/16909 rows map to local files.
- FineFake raw size is large, so it stays ignored by Git.
- This workspace has its own uv project in `code/paper_reproduction/fake_news_detection/`.
- FineFake auxiliary multimodal dataloader can feed the model.
- A development-only LLM-MFEFND forward smoke test runs successfully on a batch of 2.
- Added a paper-aligned HPT with five feature streams, five tokens per stream,
  the paper's fusion order, residual averaging, and four repeated rounds.
- Added an official-code-compatible HPT path using four text-initialized star
  tokens, 18 Transformer slots, and the public branch's update order.
- Added optional real-backbone mode: `cn-clip` `ViT-B-16` plus the official
  MAE ViT-Base checkpoint. Enable it with `LLM_MFEFND_REAL_BACKBONES=1`;
  fallback mode remains the default for fast smoke tests.
- Added a `grid_search.py` training scaffold so the public `main.py` can run
  on the available FineFake auxiliary subset; WeiBo21 full multimodal training
  still requires the unreleased author data.
- Replaced the previous simplified fusion path with the HPT module.
- A one-batch training smoke test confirms finite BCE loss, HPT gradients, and
  an optimizer parameter update.
- Added `LLM-MFEFND/mlime.py` with model-agnostic MLIME for token and image-patch
  perturbations, cosine-kernel weighting, and local ridge explanations.
- Added `scripts/smoke_mlime.py` and `scripts/compare_hpt.py` for explanation and
  HPT-variant verification.

## Reproduction gaps

1. The public `LLM-MFEFND` repo is incomplete.
2. The code imports missing local modules such as `grid_search`, `utils`, `layers`, and `pivot`.
3. The paper uses extra preprocessing:
   - keep only samples with both text and image
   - remove incomplete/corrupted samples
   - remove samples that trigger LLM refusal
4. The final paper setting is not identical to the raw public splits.
5. Strict learned bidirectional news-background/comments interaction is now
   present; its learned weights still need an isolated unit test and ablation.
6. Fallback CLIP/MAE remains the default, but the real CN-CLIP and MAE path has
   passed a batch-2 forward test.

## Next steps

1. Compare raw public data against paper settings.
2. Reconstruct the missing training scaffold from the MDFEND reference repo.
3. Add isolated tests and ablations for bidirectional cross-attention.
4. Add preprocessing for image-path mapping and LLM-generated auxiliary text.
5. Replace development fallbacks with the real pretrained MAE and CN-CLIP components.
6. Build the complete paper split and run a small real-data training experiment.
7. Wrap the trained multimodal predictor with `mlime.explain` for sample-level
   word/patch explanations; the generic explainer is ready, while a full
   benchmark explanation audit still needs the authors' 400 labeled samples.

## HPT implementation boundary

`MultiDomainFENDModel` uses `OfficialHierarchicalProgressiveTransformer` because
that matches the public GitHub forward path. `HierarchicalProgressiveTransformer`
is retained as a paper-equation implementation for controlled comparison. They
are materially different: the smoke comparison gives parameter counts of
67,918,848 and 163,011,840, with stage orders `text > image > aligned >
background > comments` and `comments > background > aligned > image > text`.
They should not be treated as two random seeds of the same model.

## Data facts

- WeiBo21 reference split in the public MDFEND repo:
  - train: 5751
  - val: 1918
  - test: 1923
- FineFake raw pickle:
  - 16909 rows
  - 13 columns
  - image paths found: 16909/16909

## Useful commands

```bash
cd code/paper_reproduction/fake_news_detection
uv sync
uv run python scripts/audit_datasets.py
uv run python scripts/audit_aux_data.py
uv run python scripts/run_text_baseline.py --dataset weibo21
uv run python scripts/run_text_baseline.py --dataset finefake
uv run python scripts/smoke_multimodal_dataloader.py
uv run python scripts/smoke_hpt.py
uv run python scripts/smoke_official_hpt.py
uv run python scripts/smoke_mlime.py
uv run python scripts/compare_hpt.py
uv run python scripts/smoke_interaction.py
uv run python scripts/smoke_llm_mfefnd_forward.py
uv run python scripts/smoke_llm_mfefnd_train_step.py
uv run python scripts/train_finefake_aux.py --epochs 1 --max-steps 2
LLM_MFEFND_REAL_BACKBONES=1 uv run python scripts/smoke_llm_mfefnd_forward.py
uv run python scripts/train_finefake_aux.py --epochs 1 --max-steps 1 --real-backbones
uv run python LLM-MFEFND/main.py --dataset en --epoch 1 --batchsize 8
```

## Sanity-check baselines

These are not paper results. They are lightweight data-flow checks using TF-IDF + Logistic Regression.

- WeiBo21 text baseline:
  - validation accuracy: 0.8838
  - test accuracy: 0.8858
- FineFake text baseline:
  - validation accuracy: 0.7758
  - test accuracy: 0.7923
- FineFake auxiliary multimodal forward smoke:
  - input batch shapes include text `(2, 170)`, image `(2, 1, 3, 224, 224)`, and CLIP text placeholder `(2, 77)`
  - model output shapes: `classify_pred` `(2,)`, `final_fusion_feature` `(2, 768)`
  - this is a development smoke test, not a paper result
- Real-backbone forward smoke:
  - CN-CLIP text input `(2, 52)` and image input `(2, 1, 3, 224, 224)`
  - MAE output is consumed as ViT tokens `(2, 197, 768)`
  - this is a data-flow validation, not a paper result

## Auxiliary LLM data coverage

- FineFake auxiliary CSV:
  - rows: 200
  - matches public FineFake by `text`: 200/200
- WeiBo21 auxiliary CSVs:
  - `GPT-DS-GLM-Weibo21.csv`: 124/200 rows match the public MDFEND split by `content`
  - `background_new_git.csv`: 250/400 rows match
  - `comments_new_git.csv`: 250/400 rows match
  - `content_new_git.csv`: 250/400 rows match

## Working rule

Use `uv` for Python in this workspace. Prefer `uv run` and `uv sync` over direct `python` or `pip`.
