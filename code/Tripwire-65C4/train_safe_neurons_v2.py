"""
train_safe_neurons_v2.py

Select safety neurons via per-neuron effect sizes (AUROC / Cohen's d),
BH-FDR significance filtering, and utility-specificity filtering.

Usage (full extraction, requires GPU):
    python train_safe_neurons_v2.py \
        --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
        --analysis_data_dir ~/neuron_experiments/analysis_data \
        --output_dir neurons \
        --device cuda:0 \
        --target_total 7568

Usage (re-threshold, seconds, no model loading):
    python train_safe_neurons_v2.py \
        --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
        --scores_npz neurons/llama2_safe_neurons_v2_scores.npz \
        --target_total 5000 --max_utility_ratio 0.3
"""

import os
import json
import argparse
import pickle

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm

# Imported lazily below in the original script; these helpers are now local so
# this release directory does not need train_detector.py.
from pathlib import Path

def detect_model_type(model_path):
    mp = model_path.lower()
    if "llama-2" in mp or "llama2" in mp: return "llama2", "Llama2", "llama2_datas"
    if "llama-3" in mp or "llama3" in mp: return "llama3", "Llama3", "llama3_datas"
    if "qwen3" in mp: return "qwen3", "Qwen3", "qwen3_datas"
    if "qwen" in mp and "32b" in mp: return "qwen_large", "Qwen_large", "qwen_large_datas"
    if "qwen" in mp: return "qwen", "Qwen", "qwen_datas"
    return "generic", Path(model_path).name[:20], "qwen3_datas"

def get_chat_template(tokenizer, model_type):
    if model_type == "qwen3": return "qwen3"
    if getattr(tokenizer, "chat_template", None) is not None: return "llama2_custom" if model_type == "llama2" else "builtin"
    return {"llama2": "llama2_custom", "llama3": "llama3_fallback", "qwen": "qwen_fallback", "qwen_large": "qwen_fallback"}.get(model_type, "generic")

def format_prompt(tokenizer, content, kind):
    if kind == "qwen3": return tokenizer.apply_chat_template([{"role":"user","content":content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if kind == "builtin": return tokenizer.apply_chat_template([{"role":"user","content":content}], tokenize=False, add_generation_prompt=True)
    if kind == "llama2_custom": return f"<s>[INST] {content} [/INST]"
    if kind == "llama3_fallback": return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if kind == "qwen_fallback": return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    return content

def extract_hidden_states_batch(mt, prompts, layer_names, batch_size, device):
    from casper import nethook
    all_acts = {name: [] for name in layer_names}
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting hidden states"):
        inputs = mt.tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad(), nethook.TraceDict(mt.model, layer_names, retain_output=True) as traces:
            mt.model(**inputs)
            positions = inputs["attention_mask"].sum(dim=1) - 1
            for name in layer_names:
                output = traces[name].output
                if isinstance(output, tuple): output = output[0]
                all_acts[name].append(output[torch.arange(output.size(0), device=output.device), positions].detach().cpu().float().numpy())
    return {name: np.concatenate(parts, axis=0) for name, parts in all_acts.items()}

def select_target_layers(model):
    """
    Select all modules whose name contains 'gate' or 'up'
    (i.e. gate_proj and up_proj in the MLP layers).
    Returns sorted list of layer names.
    """
    target_layers = []
    for name, _ in model.named_modules():
        if any(keyword in name.lower() for keyword in ["gate", "up"]):
            target_layers.append(name)
    return sorted(target_layers)

def extract_activations_maxpool(mt, prompts, layer_names, batch_size, device):
    """
    For each prompt, run one forward pass and collect the max-pooled activation
    from every layer in layer_names.  Returns dict[layer_name] -> np.array (N, hidden).

    Unlike extract_hidden_states_batch which takes the last-token hidden state,
    this uses max-pooling over the sequence dimension (dim=1), matching the
    original notebook pipeline.
    """
    from casper import nethook
    all_acts = {name: [] for name in layer_names}

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting activations"):
        batch = prompts[i : i + batch_size]
        inputs = mt.tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            with nethook.TraceDict(mt.model, layer_names, retain_output=True) as traces:
                _ = mt.model(**inputs)

                for lname in layer_names:
                    out = traces[lname].output
                    if isinstance(out, tuple):
                        out = out[0]
                    # out: (B, T, H) — max-pool over sequence dimension
                    act = out.max(dim=1)[0]  # (B, H)
                    all_acts[lname].append(act.detach().cpu().float().numpy())

    for lname in layer_names:
        all_acts[lname] = np.concatenate(all_acts[lname], axis=0)  # (N, H)

    return all_acts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Select safety neurons via per-neuron effect sizes + FDR + "
                    "utility-specificity filtering")
    p.add_argument("--model_path", type=str, required=True,
                   help="HuggingFace model path or name")
    p.add_argument("--analysis_data_dir", type=str, default="analysis_data",
                   help="Dir containing advbench_behaviors.csv / alpaca_clean.json "
                        "/ mt_bench_questions.json")
    p.add_argument("--output_dir", type=str, default="neurons",
                   help="Where to save the safe_neurons pkl")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for activation extraction")
    p.add_argument("--num_harmful", type=int, default=500,
                   help="Number of harmful prompts from advbench")
    p.add_argument("--num_safe", type=int, default=500,
                   help="Number of safe prompts from alpaca")
    p.add_argument("--pooling", type=str, default="last",
                   choices=["last", "max"],
                   help="Activation pooling over the sequence dim")
    p.add_argument("--selection_method", type=str, default="auroc",
                   choices=["auroc", "effect_size"],
                   help="Primary per-neuron score for ranking/thresholding")
    p.add_argument("--target_total", type=int, default=None,
                   help="Global neuron budget: keep the top-N neurons across "
                        "ALL layers by the primary score (after direction/FDR/"
                        "specificity pre-filters). Overrides --min_auroc/--min_effect")
    p.add_argument("--min_auroc", type=float, default=0.75,
                   help="(no --target_total) min per-neuron AUROC")
    p.add_argument("--min_effect", type=float, default=0.5,
                   help="(no --target_total, selection_method=effect_size) "
                        "min Cohen's d")
    p.add_argument("--fdr_alpha", type=float, default=0.05,
                   help="Benjamini-Hochberg FDR level over ALL neurons of all "
                        "layers (Welch t-test); set <=0 to disable")
    p.add_argument("--utility_source", type=str, default="mtbench",
                   choices=["mtbench", "alpaca_holdout", "none"],
                   help="Prompt set for the specificity filter: MT-Bench "
                        "questions, held-out alpaca (rows after --num_safe), "
                        "or disable the filter")
    p.add_argument("--num_utility", type=int, default=80,
                   help="Number of utility prompts for the specificity filter")
    p.add_argument("--max_utility_ratio", type=float, default=0.5,
                   help="Exclude neurons whose (util_mean - safe_mean) / "
                        "(harmful_mean - safe_mean) exceeds this ratio")
    p.add_argument("--top_k_per_layer", type=int, default=None,
                   help="Optional cap: keep only the top-K neurons per layer "
                        "by the primary score")
    p.add_argument("--scores_npz", type=str, default=None,
                   help="Load precomputed per-neuron scores from this npz and "
                        "skip model loading + extraction entirely")
    p.add_argument("--no_save_scores", action="store_true",
                   help="Do not write the scores npz cache after extraction")
    p.add_argument("--compare_pkl", type=str, default=None,
                   help="Optional: path to an existing safe_neurons pkl to "
                        "print overlap against")
    p.add_argument("--torch_dtype", type=str, default=None,
                   choices=["float16", "bfloat16", "float32"],
                   help="dtype for loading the model (default: float16, "
                        "or None/full-precision for 20b models)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-neuron statistics (all vectorized over the hidden dim)
# ---------------------------------------------------------------------------
def compute_auroc(acts, harmful_mask):
    """
    Rank-based AUROC per neuron: P(harmful activation > safe activation).
    acts: (N, H), harmful_mask: (N,) bool. Returns (H,).
    Equivalent to the Mann-Whitney U statistic, scale-invariant.
    """
    ranks = stats.rankdata(acts, axis=0)
    n_h = int(harmful_mask.sum())
    n_s = int((~harmful_mask).sum())
    sum_ranks_h = ranks[harmful_mask].sum(axis=0)
    u = sum_ranks_h - n_h * (n_h + 1) / 2.0
    return u / float(n_h * n_s)


def compute_cohens_d(acts, harmful_mask):
    """Standardized mean difference (harmful - safe) per neuron. Returns (H,)."""
    h = acts[harmful_mask]
    s = acts[~harmful_mask]
    mu_h, mu_s = h.mean(axis=0), s.mean(axis=0)
    var_h, var_s = h.var(axis=0, ddof=1), s.var(axis=0, ddof=1)
    n_h, n_s = len(h), len(s)
    pooled = np.sqrt(((n_h - 1) * var_h + (n_s - 1) * var_s)
                     / max(n_h + n_s - 2, 1))
    return (mu_h - mu_s) / (pooled + 1e-12)


def benjamini_hochberg(pvals, alpha):
    """BH step-up procedure. Returns boolean mask of rejected hypotheses."""
    pvals = np.asarray(pvals)
    m = len(pvals)
    order = np.argsort(pvals)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    below = pvals[order] <= thresholds
    reject = np.zeros(m, dtype=bool)
    if below.any():
        k_max = np.max(np.where(below)[0])
        reject[order[: k_max + 1]] = True
    return reject


# ---------------------------------------------------------------------------
# Scores cache (npz)
# ---------------------------------------------------------------------------
SCORE_FIELDS = ("auroc", "cohens_d", "mu_h", "mu_s", "pvals", "utility_ratio")


def save_scores_npz(path, target_layers, chat_template_type,
                    utility_source_used, num_utility, layer_stats, pvals_per_layer):
    payload = {
        "target_layers": np.array(target_layers),
        "chat_template_type": np.array(chat_template_type),
        "utility_source": np.array(utility_source_used),
        "num_utility": np.array(num_utility),
    }
    for i, lname in enumerate(target_layers):
        st = layer_stats[lname]
        for field in ("auroc", "cohens_d", "mu_h", "mu_s"):
            payload[f"{lname}|{field}"] = st[field]
        payload[f"{lname}|pvals"] = pvals_per_layer[i]
        if st["utility_ratio"] is not None:
            payload[f"{lname}|utility_ratio"] = st["utility_ratio"]
    np.savez_compressed(path, **payload)


def load_scores_npz(path):
    data = np.load(path, allow_pickle=False)
    target_layers = [str(x) for x in data["target_layers"]]
    chat_template_type = str(data["chat_template_type"])
    utility_source_used = str(data["utility_source"])
    num_utility = int(data["num_utility"])

    layer_stats, pvals_per_layer = {}, []
    for lname in target_layers:
        ur_key = f"{lname}|utility_ratio"
        layer_stats[lname] = {
            "auroc": data[f"{lname}|auroc"],
            "cohens_d": data[f"{lname}|cohens_d"],
            "mu_h": data[f"{lname}|mu_h"],
            "mu_s": data[f"{lname}|mu_s"],
            "utility_ratio": data[ur_key] if ur_key in data.files else None,
        }
        pvals_per_layer.append(data[f"{lname}|pvals"])
    return (target_layers, chat_template_type, utility_source_used,
            num_utility, layer_stats, pvals_per_layer)


# ---------------------------------------------------------------------------
# Full extraction path (model forward passes)
# ---------------------------------------------------------------------------
def extract_and_score(args, model_type):
    """Load model, extract activations, score every neuron. Returns the same
    tuple as load_scores_npz."""
    import pandas as pd
    from utils.modelUtils import ModelAndTokenizer

    print("\nStep 1: Loading model...")
    _dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                  "float32": torch.float32}
    if args.torch_dtype is not None:
        _load_dtype = _dtype_map[args.torch_dtype]
    else:
        _load_dtype = (torch.float16 if "20b" not in args.model_path else None)
    mt = ModelAndTokenizer(
        args.model_path,
        low_cpu_mem_usage=True,
        torch_dtype=_load_dtype,
        device=args.device,
    )
    chat_template_type = get_chat_template(mt.tokenizer, model_type)
    print(f"  Chat template: {chat_template_type}")

    print("\nStep 2: Selecting target layers...")
    target_layers = select_target_layers(mt.model)
    print(f"  Found {len(target_layers)} target layers")

    named_modules = dict(mt.model.named_modules())
    total_neuron_pool = 0
    layer_neuron_counts = {}
    for lname in target_layers:
        module = named_modules[lname]
        n = getattr(module, "out_features", None)
        if n is None:
            n = module.weight.shape[0]
        layer_neuron_counts[lname] = int(n)
        total_neuron_pool += int(n)
    print(f"  Total neuron pool (sum of out_features across all target layers): "
          f"{total_neuron_pool}")

    print("\nStep 3: Loading prompt sets...")
    harmful_csv = os.path.join(args.analysis_data_dir, "advbench_behaviors.csv")
    alpaca_json = os.path.join(args.analysis_data_dir, "alpaca_clean.json")

    if not os.path.exists(harmful_csv):
        raise FileNotFoundError(f"advbench_behaviors.csv not found at {harmful_csv}")
    if not os.path.exists(alpaca_json):
        raise FileNotFoundError(f"alpaca_clean.json not found at {alpaca_json}")

    harmful_questions = pd.read_csv(harmful_csv)["Behavior"].tolist()[:args.num_harmful]
    with open(alpaca_json, "r", encoding="utf-8") as f:
        alpaca = json.load(f)
    safe_questions = [x["instruction"] for x in alpaca][:args.num_safe]

    utility_questions = []
    utility_source_used = args.utility_source
    if args.utility_source == "mtbench":
        mtb_path = os.path.join(args.analysis_data_dir, "mt_bench_questions.json")
        if os.path.exists(mtb_path):
            with open(mtb_path, "r", encoding="utf-8") as f:
                mtb = json.load(f)
            utility_questions = [item["turns"][0] for item in mtb][:args.num_utility]
        else:
            print(f"  WARNING: {mtb_path} not found, falling back to alpaca_holdout")
            utility_source_used = "alpaca_holdout"
    if utility_source_used == "alpaca_holdout":
        holdout = [x["instruction"] for x in alpaca][args.num_safe:
                                                     args.num_safe + args.num_utility]
        if not holdout:
            print("  WARNING: no alpaca rows left after --num_safe, "
                  "disabling specificity filter")
            utility_source_used = "none"
        utility_questions = holdout
    if utility_source_used == "none":
        utility_questions = []

    print(f"  Harmful: {len(harmful_questions)}, Safe: {len(safe_questions)}, "
          f"Utility: {len(utility_questions)} ({utility_source_used})")

    questions = harmful_questions + safe_questions + utility_questions
    prompts = [format_prompt(mt.tokenizer, q, chat_template_type) for q in questions]

    n_h, n_s, n_u = (len(harmful_questions), len(safe_questions),
                     len(utility_questions))
    harmful_mask = np.zeros(n_h + n_s, dtype=bool)
    harmful_mask[:n_h] = True

    print(f"\nStep 4: Extracting activations (pooling={args.pooling})...")
    if args.pooling == "last":
        activations = extract_hidden_states_batch(
            mt, prompts, target_layers, args.batch_size, args.device
        )
    else:
        activations = extract_activations_maxpool(
            mt, prompts, target_layers, args.batch_size, args.device
        )

    print("\nStep 5: Scoring neurons (AUROC / Cohen's d / Welch t-test)...")
    layer_stats, pvals_per_layer = {}, []

    for lname in tqdm(target_layers, desc="Scoring layers"):
        acts = activations[lname]
        hs_acts = acts[: n_h + n_s]
        ut_acts = acts[n_h + n_s:] if n_u else None

        auroc = compute_auroc(hs_acts, harmful_mask)
        cohens_d = compute_cohens_d(hs_acts, harmful_mask)
        _, pvals = stats.ttest_ind(hs_acts[harmful_mask], hs_acts[~harmful_mask],
                                   axis=0, equal_var=False)
        pvals = np.nan_to_num(pvals, nan=1.0)

        mu_h = hs_acts[harmful_mask].mean(axis=0)
        mu_s = hs_acts[~harmful_mask].mean(axis=0)
        utility_ratio = None
        if ut_acts is not None:
            mu_u = ut_acts.mean(axis=0)
            utility_ratio = (mu_u - mu_s) / np.where(
                np.abs(mu_h - mu_s) < 1e-12, np.inf, mu_h - mu_s)

        layer_stats[lname] = {
            "auroc": auroc, "cohens_d": cohens_d,
            "mu_h": mu_h, "mu_s": mu_s, "utility_ratio": utility_ratio,
        }
        pvals_per_layer.append(pvals)

        del activations[lname]

    return (target_layers, chat_template_type, utility_source_used,
            n_u, layer_stats, pvals_per_layer)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    model_type, model_short_name, _ = detect_model_type(args.model_path)

    print("=" * 70)
    print("SAFE NEURON SELECTION CONFIGURATION")
    print(f"  Model:             {args.model_path}")
    print(f"  Model type:        {model_type}")
    print(f"  Scores npz:        {args.scores_npz or '(full extraction run)'}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Pooling:           {args.pooling}")
    print(f"  Selection method:  {args.selection_method}")
    print(f"  Target total:      {args.target_total or '(absolute threshold)'}")
    print(f"  Min AUROC:         {args.min_auroc}")
    print(f"  Min Cohen's d:     {args.min_effect}")
    print(f"  FDR alpha:         {args.fdr_alpha}")
    print(f"  Utility source:    {args.utility_source} (n={args.num_utility})")
    print(f"  Max utility ratio: {args.max_utility_ratio}")
    print(f"  Top-K per layer:   {args.top_k_per_layer}")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Score acquisition: cached npz (fast) or full extraction (GPU)
    # -----------------------------------------------------------------------
    if args.scores_npz:
        print(f"\nLoading cached scores from {args.scores_npz} ...")
        (target_layers, chat_template_type, utility_source_used, num_utility,
         layer_stats, pvals_per_layer) = load_scores_npz(args.scores_npz)
        print(f"  {len(target_layers)} layers, utility_source={utility_source_used}")
        scores_npz_path = args.scores_npz
    else:
        (target_layers, chat_template_type, utility_source_used, num_utility,
         layer_stats, pvals_per_layer) = extract_and_score(args, model_type)
        scores_npz_path = None
        if not args.no_save_scores:
            os.makedirs(args.output_dir, exist_ok=True)
            scores_npz_path = os.path.join(
                args.output_dir,
                f"{model_short_name.lower()}_safe_neurons_v2_scores.npz")
            save_scores_npz(scores_npz_path, target_layers, chat_template_type,
                            utility_source_used, num_utility, layer_stats,
                            pvals_per_layer)
            print(f"\n  Scores cached: {scores_npz_path}")
            print("  (re-threshold later with --scores_npz, no GPU needed)")

    pval_slices, offset = {}, 0
    for i, lname in enumerate(target_layers):
        h = len(pvals_per_layer[i])
        pval_slices[lname] = (offset, offset + h)
        offset += h

    # Global BH-FDR across ALL neurons of ALL layers
    if args.fdr_alpha > 0:
        fdr_reject = benjamini_hochberg(np.concatenate(pvals_per_layer),
                                        args.fdr_alpha)
        print(f"  FDR: {int(fdr_reject.sum())}/{len(fdr_reject)} neurons "
              f"significant at alpha={args.fdr_alpha}")
    else:
        fdr_reject = np.ones(offset, dtype=bool)
        print("  FDR disabled")

    # -----------------------------------------------------------------------
    # Selection. Pass 1: pre-filters (direction, FDR, specificity).
    # Pass 2: primary-score gate — global top-N budget (--target_total)
    # or absolute threshold (--min_auroc / --min_effect).
    # -----------------------------------------------------------------------
    print("\nSelecting toxic neurons...")
    filter_counts = {"candidates": 0, "after_fdr": 0, "after_specificity": 0,
                     "after_score": 0, "after_topk": 0}
    base_masks, primaries = {}, {}

    for lname in target_layers:
        st = layer_stats[lname]
        lo, hi = pval_slices[lname]

        mask = st["mu_h"] > st["mu_s"]
        filter_counts["candidates"] += int(mask.sum())

        mask &= fdr_reject[lo:hi]
        filter_counts["after_fdr"] += int(mask.sum())

        if st["utility_ratio"] is not None:
            mask &= st["utility_ratio"] <= args.max_utility_ratio
        filter_counts["after_specificity"] += int(mask.sum())

        base_masks[lname] = mask
        primaries[lname] = (st["auroc"] if args.selection_method == "auroc"
                            else st["cohens_d"])

    if args.target_total is not None:
        flat = np.concatenate([np.where(base_masks[l], primaries[l], -np.inf)
                               for l in target_layers])
        n_avail = int(np.isfinite(flat).sum())
        k = min(args.target_total, n_avail)
        if k < args.target_total:
            print(f"  NOTE: only {n_avail} candidates survive the pre-filters "
                  f"(< target {args.target_total})")
        if k > 0:
            cutoff = np.partition(flat, -k)[-k]
            global_sel = flat >= cutoff
        else:
            global_sel = np.zeros_like(flat, dtype=bool)
        score_masks = {l: global_sel[pval_slices[l][0]: pval_slices[l][1]]
                       for l in target_layers}
    else:
        thr = args.min_auroc if args.selection_method == "auroc" else args.min_effect
        score_masks = {l: base_masks[l] & (primaries[l] >= thr)
                       for l in target_layers}

    toxic_neurons_dict = {}
    for lname in target_layers:
        mask = base_masks[lname] & score_masks[lname]
        filter_counts["after_score"] += int(mask.sum())

        selected = np.where(mask)[0]
        if args.top_k_per_layer is not None and len(selected) > args.top_k_per_layer:
            top = np.argsort(primaries[lname][selected])[::-1][: args.top_k_per_layer]
            selected = np.sort(selected[top])
        filter_counts["after_topk"] += len(selected)

        toxic_neurons_dict[lname] = selected.astype(np.int64)

    total_toxic = sum(len(v) for v in toxic_neurons_dict.values())
    print(f"  Filter funnel: {filter_counts}")
    print(f"  Total toxic neurons across all layers: {total_toxic}")

    # -----------------------------------------------------------------------
    # Optional overlap report against an existing pkl
    # -----------------------------------------------------------------------
    overlap_summary = None
    if args.compare_pkl and os.path.exists(args.compare_pkl):
        with open(args.compare_pkl, "rb") as f:
            ref = pickle.load(f)
        inter = sum(len(np.intersect1d(toxic_neurons_dict.get(l, []), v))
                    for l, v in ref.items())
        ref_total = sum(len(v) for v in ref.values())
        overlap_summary = {
            "compare_pkl": args.compare_pkl,
            "ref_total": ref_total,
            "v2_total": total_toxic,
            "intersection": inter,
        }
        print(f"\n  Overlap vs {os.path.basename(args.compare_pkl)}: "
              f"{inter} shared (ref={ref_total}, v2={total_toxic})")

    # -----------------------------------------------------------------------
    # Save pkl + metadata
    # -----------------------------------------------------------------------
    print("\nSaving safe neurons...")
    os.makedirs(args.output_dir, exist_ok=True)

    method_tag = "auroc" if args.selection_method == "auroc" else "d"
    if args.target_total is not None:
        tag_parts = [f"{method_tag}_total{args.target_total}"]
    elif args.selection_method == "auroc":
        tag_parts = [f"auroc{args.min_auroc}"]
    else:
        tag_parts = [f"d{args.min_effect}"]
    if args.fdr_alpha > 0:
        tag_parts.append(f"fdr{args.fdr_alpha}")
    if utility_source_used != "none":
        tag_parts.append(f"spec{args.max_utility_ratio}")
    if args.top_k_per_layer is not None:
        tag_parts.append(f"top{args.top_k_per_layer}")
    tag = "_".join(tag_parts).replace(".", "_")

    pkl_path = os.path.join(
        args.output_dir, f"{model_short_name.lower()}_safe_neurons_{tag}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(toxic_neurons_dict, f)
    print(f"  Saved: {pkl_path}")

    per_layer_counts = {lname: len(toxic_neurons_dict[lname])
                        for lname in sorted(toxic_neurons_dict.keys())}

    metadata = {
        "model_path": args.model_path,
        "model_type": model_type,
        "model_short_name": model_short_name,
        "chat_template_type": chat_template_type,
        "num_target_layers": len(target_layers),
        "target_layers": target_layers,
        "num_harmful": args.num_harmful,
        "num_safe": args.num_safe,
        "total_toxic_neurons": total_toxic,
        "per_layer_counts": per_layer_counts,
        "selection_version": "v2",
        "pooling": args.pooling,
        "selection_method": args.selection_method,
        "target_total": args.target_total,
        "min_auroc": args.min_auroc if args.target_total is None else None,
        "min_effect": args.min_effect if args.target_total is None else None,
        "fdr_alpha": args.fdr_alpha,
        "utility_source": utility_source_used,
        "num_utility": num_utility,
        "max_utility_ratio": args.max_utility_ratio,
        "top_k_per_layer": args.top_k_per_layer,
        "scores_npz": scores_npz_path,
        "filter_funnel": filter_counts,
        "overlap_vs_reference": overlap_summary,
    }
    meta_path = os.path.join(
        args.output_dir, f"{model_short_name.lower()}_safe_neurons_{tag}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {meta_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Layer':<50} {'Toxic Neurons':>14}")
    print("-" * 70)
    for lname in sorted(toxic_neurons_dict.keys()):
        print(f"{lname:<50} {len(toxic_neurons_dict[lname]):>14}")
    print("-" * 70)
    print(f"{'TOTAL':<50} {total_toxic:>14}")
    print(f"\nOutput: {pkl_path}")
    print("Selection complete!")


if __name__ == "__main__":
    main()
