
# Tripwire: Triggering Aligned Refusal via Statistically Certified Safety Neurons

## Core Scripts

`release_codes` contains four self-contained scripts:

- `train_safe_neurons_v2.py`
- `train_sparse_classifier.py`
- `run_sparse_pipeline.py`
- `run_edit_pipeline.py`

These scripts depend on two local packages:

- `casper/` — `nethook.py` for hooking model layer activations
- `utils/` — `modelUtils.py` provides `ModelAndTokenizer`

## Runtime Environment

Python 3.12. Required dependencies:

```
numpy  scipy  tqdm  pandas  torch  scikit-learn  transformers  fastchat
```

## Data / Model Resources

`--analysis_data_dir` contains:

- `advbench_behaviors.csv`
- `alpaca_clean.json`
- `mt_bench_questions.json`

`--data_base_dir` currently only contains `llama2_datas/adversarial_gcg.json`. Missing attack files will print `WARNING: ... not found, skipping` and be skipped without failure.

`run_edit_pipeline.py` loads `meta-llama/Llama-Guard-3-8B` as the judge by default. Requirements:

1. A logged-in HuggingFace token (`~/.cache/huggingface/token`)
2. Account access granted to `meta-llama/Llama-Guard-3-8B` on HuggingFace
3. Model weights cached locally or downloadable

## Usage Workflow

1. Run `train_safe_neurons_v2.py` to generate `safe_neurons`.
2. Run `train_sparse_classifier.py` to train the sparse classifier.
3. Choose an execution approach:
   - `run_sparse_pipeline.py`: detect first, then intervene.
   - `run_edit_pipeline.py`: directly edit model weights.

## Examples

```bash
python train_safe_neurons_v2.py \
  --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
  --analysis_data_dir analysis_data \
  --output_dir neurons \
  --device cuda:0 \
  --target_total 1000

python train_sparse_classifier.py \
  --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
  --neurons_pkl neurons/llama2_safe_neurons_auroc_total1000_fdr0_05_spec0_5.pkl \
  --analysis_data_dir analysis_data \
  --data_base_dir data_base_dir \
  --output_dir sparse_classifier_results \
  --device cuda:0 \
  --score_threshold percentile95

python run_sparse_pipeline.py \
  --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
  --sparse_clf_path sparse_classifier_results/Llama2/sparse_clf_total1000.pkl \
  --safe_neurons_path neurons/llama2_safe_neurons_auroc_total1000_fdr0_05_spec0_5.pkl \
  --data_base_dir data_base_dir \
  --analysis_data_dir analysis_data \
  --output_dir sparse_pipeline_results \
  --N 200 \
  --layer_selection all \
  --device cuda:0 \
  --guard_device cuda:0

python run_edit_pipeline.py \
  --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
  --safe_neurons_path neurons/llama2_safe_neurons_auroc_total1000_fdr0_05_spec0_5.pkl \
  --edit_mode bias_patch \
  --layer_selection all \
  --data_base_dir data_base_dir \
  --analysis_data_dir analysis_data \
  --output_dir pipeline_results_edit \
  --N 200 \
  --device cuda:0 \
  --guard_device cuda:0
```

The classifier output is named `{output_dir}/{model_short_name}/sparse_clf_{tag}.pkl`, where `{tag}` is extracted from `total(\d+)` in the `--neurons_pkl` filename. Activations are cached to `sparse_acts_cache.npz`; pass `--acts_npz` on subsequent runs to skip re-running the model.

In a single-GPU environment, set `--guard_device` to the same value as `--device`.
