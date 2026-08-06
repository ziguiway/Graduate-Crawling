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

## Reproduction gaps

1. The public `LLM-MFEFND` repo is incomplete.
2. The code imports missing local modules such as `grid_search`, `utils`, `layers`, and `pivot`.
3. The paper uses extra preprocessing:
   - keep only samples with both text and image
   - remove incomplete/corrupted samples
   - remove samples that trigger LLM refusal
4. The final paper setting is not identical to the raw public splits.

## Next steps

1. Compare raw public data against paper settings.
2. Reconstruct the missing training scaffold from the MDFEND reference repo.
3. Add preprocessing for image-path mapping and LLM-generated auxiliary text.
4. Run a minimal baseline first, then add the full LLM-MFEFND fusion model.
5. Replace development fallbacks with the real pretrained MAE and CN-CLIP components.

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
uv run python scripts/smoke_llm_mfefnd_forward.py
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
