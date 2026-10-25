"""
train_sparse_classifier.py

Train a sparse Logistic Regression detector on selected safety-neuron
activations: last-token gate_proj/up_proj activations of the neurons chosen
by train_safe_neurons_v2.py are concatenated into a feature vector, then used
to train a benign-vs-adversarial classifier.

Training data: advbench_behaviors.csv (harmful, label=1) + alpaca_clean.json
(benign, label=0) — same source as train_safe_neurons_v2.py.

Test data: <data_base_dir>/<model>_datas/adversarial_{gcg,amplegcg,autodan,pair}.json
benign vs adversarial prompts.

Design: standardization and threshold calibration use only the train domain
(advbench/alpaca); test-domain (casper) samples are only ever scored, never
fit or calibrated on.
  - StandardScaler is fit once on the train domain's benign (alpaca) subset
    and reused for test.
  - --score_threshold fixed0.5 (default): flat 0.5 threshold on the full
    test set.
  - --score_threshold percentileNN (e.g. percentile95): threshold set at the
    NN-th percentile of predict_proba scores on the train domain's own
    benign samples, then scored on the full test set.

Also reports AUC (threshold-free) and the predict_proba percentile spread
(diagnostic for probability saturation).

Independent knobs for probability saturation (orthogonal to the threshold
choice above):
  --C <float>            Inverse regularization strength for LogisticRegression.
                         Lower values (e.g. 0.1, 0.01) counteract saturation
                         from high-dimensional near-linearly-separable features.
  --calibration <sigmoid|isotonic>
                         Wrap the LR in CalibratedClassifierCV (5-fold,
                         train-domain only) to recalibrate output probabilities.

Usage:
    python train_sparse_classifier.py \
        --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
        --neurons_pkl \
            neurons/llama2_safe_neurons_auroc_total1000_fdr0_05_spec0_5.pkl \
            neurons/llama2_safe_neurons_auroc_total2000_fdr0_05_spec0_5.pkl \
        --analysis_data_dir analysis_data \
        --data_base_dir data_base_dir \
        --output_dir sparse_classifier_results \
        --device cuda:0 \
        --score_threshold percentile95

Re-run for tuning (seconds, no GPU):
    python train_sparse_classifier.py \
        --model_path /common/public/LLAMA2-HF/Llama-2-7b-chat-hf \
        --neurons_pkl neurons/llama2_safe_neurons_auroc_total1000_fdr0_05_spec0_5.pkl \
        --acts_npz sparse_classifier_results/Llama2/sparse_acts_cache.npz \
        --score_threshold percentile95
"""

import os
import json
import argparse
import pickle
import re

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

from utils.modelUtils import ModelAndTokenizer
from casper import nethook

# Self-contained copies of the prompt helpers formerly imported from
# train_detector.py.  Keep this release script runnable without the source tree.
from pathlib import Path

def detect_model_type(model_path):
    mp = model_path.lower()
    if "llama-2" in mp or "llama2" in mp:
        return "llama2", "Llama2", "llama2_datas"
    if "llama-3" in mp or "llama3" in mp:
        return "llama3", "Llama3", "llama3_datas"
    if "qwen3" in mp:
        return "qwen3", "Qwen3", "qwen3_datas"
    if "qwen" in mp and "32b" in mp:
        return "qwen_large", "Qwen_large", "qwen_large_datas"
    if "qwen" in mp:
        return "qwen", "Qwen", "qwen_datas"
    name = Path(model_path).name[:20]
    print("Warning: unknown model type, using qwen3_datas as fallback")
    return "generic", name, "qwen3_datas"


def get_chat_template(tokenizer, model_type):
    if model_type == "qwen3": return "qwen3"
    if getattr(tokenizer, "chat_template", None) is not None:
        return "llama2_custom" if model_type == "llama2" else "builtin"
    if model_type == "llama2": return "llama2_custom"
    if model_type == "llama3": return "llama3_fallback"
    if model_type in ("qwen", "qwen_large"): return "qwen_fallback"
    return "generic"


def format_prompt(tokenizer, content, chat_template_type):
    if chat_template_type == "qwen3":
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if chat_template_type == "builtin":
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    if chat_template_type == "llama2_custom": return f"<s>[INST] {content} [/INST]"
    if chat_template_type == "llama3_fallback": return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    if chat_template_type == "qwen_fallback": return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    return content

ATTACK_FILES = {
    "gcg": "adversarial_gcg.json",
    "amplegcg": "adversarial_amplegcg.json",
    "autodan": "adversarial_autodan.json",
    "pair": "adversarial_pair.json",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train a sparse LR on selected safety-neuron activations, "
                    "test on benign vs adversarial")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--neurons_pkl", type=str, nargs="+", required=True,
                   help="One or more safe_neurons pkl files, each gets its "
                        "own classifier")
    p.add_argument("--analysis_data_dir", type=str,
                   default="analysis_data",
                   help="Dir with advbench_behaviors.csv / alpaca_clean.json "
                        "(train data)")
    p.add_argument("--data_base_dir", type=str,
                   default="data_base_dir",
                   help="Root dir with <model>_datas/adversarial_*.json "
                        "(test data: benign vs adversarial)")
    p.add_argument("--output_dir", type=str, default="sparse_classifier_results")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_harmful", type=int, default=500,
                   help="Train: advbench harmful prompts")
    p.add_argument("--num_safe", type=int, default=500,
                   help="Train: alpaca benign prompts")
    p.add_argument("--num_test_per_class", type=int, default=500,
                   help="Test: benign prompts, and adversarial prompts "
                        "(split evenly across the 4 attacks)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--acts_npz", type=str, default=None,
                   help="Load cached activations from this npz and skip "
                        "model loading + extraction entirely")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverse regularization strength for the LR. Lower "
                        "(e.g. 0.1, 0.01) keeps probabilities from "
                        "saturating to 0/1 on near-separable high-dim data")
    p.add_argument("--calibration", type=str, default=None,
                   choices=[None, "sigmoid", "isotonic"],
                   help="Wrap the LR in CalibratedClassifierCV (5-fold CV "
                        "on the train domain only)")
    p.add_argument("--score_threshold", type=str, default="fixed0.5",
                   help="fixed0.5 (default): score at 0.5 on the full test "
                        "set. percentileNN (e.g. percentile95, "
                        "percentile99): set the threshold at the NN-th "
                        "percentile of predict_proba scores on the train "
                        "domain's own benign (alpaca) samples, then score "
                        "on the full test set. Any integer/float NN in "
                        "(0, 100) is accepted.")
    p.add_argument("--torch_dtype", type=str, default=None,
                   choices=["float16", "bfloat16", "float32"],
                   help="dtype for loading the model (default: float16, "
                        "or None/full-precision for 20b models)")
    return p.parse_args()


def _parse_score_threshold(spec):
    """Returns ('fixed0.5', None) | ('percentile', NN)."""
    if spec == "fixed0.5":
        return spec, None
    m = re.fullmatch(r"percentile(\d+(?:\.\d+)?)", spec)
    if m:
        pct = float(m.group(1))
        if not (0 < pct < 100):
            raise ValueError(
                f"--score_threshold {spec}: percentile must be in (0, 100)")
        return "percentile", pct
    raise ValueError(
        f"--score_threshold {spec!r} not recognized; expected 'fixed0.5' "
        f"or 'percentileNN' (e.g. 'percentile95')")


def short_tag(pkl_path):
    base = os.path.basename(pkl_path).replace(".pkl", "")
    m = re.search(r"total(\d+)", base)
    if m:
        return f"total{m.group(1)}"
    m = re.search(r"safe_neurons_(.+)", base)
    return m.group(1) if m else base


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_train_prompts(args):
    """advbench (harmful=1) + alpaca (benign=0)."""
    harmful_csv = os.path.join(args.analysis_data_dir, "advbench_behaviors.csv")
    alpaca_json = os.path.join(args.analysis_data_dir, "alpaca_clean.json")

    harmful = pd.read_csv(harmful_csv)["Behavior"].tolist()[:args.num_harmful]
    with open(alpaca_json, "r", encoding="utf-8") as f:
        alpaca = json.load(f)
    safe = [x["instruction"] for x in alpaca][:args.num_safe]

    texts = harmful + safe
    labels = np.array([1] * len(harmful) + [0] * len(safe))
    print(f"  Train — harmful: {len(harmful)}, benign: {len(safe)}")
    return texts, labels


def load_test_prompts(args, data_dir):
    """casper benign (pooled across attack files, dedup) vs adversarial
    (even split across the 4 attacks). label 0=benign, 1=adversarial."""
    n = args.num_test_per_class
    benign, seen = [], set()
    per_attack_adv = {}
    for attack, fname in ATTACK_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping {attack}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        for t in d["benign"]:
            if t not in seen:
                seen.add(t)
                benign.append(t)
        per_attack_adv[attack] = d["adversarial"]
    benign = benign[:n]

    n_attacks = max(len(per_attack_adv), 1)
    per = int(np.ceil(n / n_attacks))
    adversarial, adv_tags = [], []
    for attack, lst in per_attack_adv.items():
        take = lst[:per]
        adversarial.extend(take)
        adv_tags.extend([attack] * len(take))
    adversarial, adv_tags = adversarial[:n], adv_tags[:n]

    texts = benign + adversarial
    labels = np.array([0] * len(benign) + [1] * len(adversarial))
    attack_tags = np.array([""] * len(benign) + adv_tags)

    print(f"  Test — benign: {len(benign)}, adversarial: {len(adversarial)} "
          f"({', '.join(f'{a}={adv_tags.count(a)}' for a in per_attack_adv)})")
    return texts, labels, attack_tags


# ---------------------------------------------------------------------------
# Neuron index bookkeeping
# ---------------------------------------------------------------------------
def load_neuron_sets(neurons_pkls):
    configs = []
    for pkl_path in neurons_pkls:
        with open(pkl_path, "rb") as f:
            sel = pickle.load(f)
        sel = {l: np.asarray(v, dtype=np.int64) for l, v in sel.items()
               if len(v) > 0}
        n_neurons = int(sum(len(v) for v in sel.values()))
        configs.append({"name": short_tag(pkl_path), "pkl": pkl_path,
                        "selected": sel, "n_neurons": n_neurons})
        print(f"  {short_tag(pkl_path)}: {n_neurons} neurons over "
              f"{len(sel)} layers ({pkl_path})")

    needed = {}
    for cfg in configs:
        for l, idx in cfg["selected"].items():
            needed.setdefault(l, set()).update(idx.tolist())
    needed = {l: np.array(sorted(s), dtype=np.int64) for l, s in needed.items()}
    return configs, needed


def gather_features(neuron_dict, acts, needed):
    """Concatenate the requested neuron columns (fixed layer order) into (N, D)."""
    cols = []
    for l in sorted(neuron_dict.keys()):
        pos = np.searchsorted(needed[l], neuron_dict[l])
        cols.append(acts[l][:, pos])
    return np.concatenate(cols, axis=1)


# ---------------------------------------------------------------------------
# Activation extraction (last token, only the needed columns)
# ---------------------------------------------------------------------------
def extract_selected_activations(mt, prompts, needed, batch_size, device):
    """dict[layer] -> (N, len(needed[layer])) float32, last-token activations."""
    layer_names = sorted(needed.keys())
    all_acts = {l: [] for l in layer_names}

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting"):
        batch = prompts[i: i + batch_size]
        inputs = mt.tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(device)
        with torch.no_grad():
            with nethook.TraceDict(mt.model, layer_names,
                                   retain_output=True) as traces:
                _ = mt.model(**inputs)
                seq_lens = inputs["attention_mask"].sum(dim=1) - 1
                for l in layer_names:
                    out = traces[l].output
                    if isinstance(out, tuple):
                        out = out[0]
                    last = out[torch.arange(out.size(0), device=out.device),
                               seq_lens]
                    cols = torch.as_tensor(needed[l], device=last.device)
                    all_acts[l].append(
                        last[:, cols].detach().cpu().float().numpy())

    return {l: np.concatenate(v, axis=0) for l, v in all_acts.items()}


# ---------------------------------------------------------------------------
# Classifier construction
# ---------------------------------------------------------------------------
def build_classifier(args):
    clf = LogisticRegression(max_iter=2000, random_state=args.seed, C=args.C)
    if args.calibration:
        clf = CalibratedClassifierCV(clf, method=args.calibration, cv=5)
    return clf


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------
def fit_benign_scaler(X, y):
    """Fit a StandardScaler on this domain's benign subset only (y==0), so
    the resulting z-scores mean 'how far off the train domain's own normal
    baseline'. Falls back to fitting on all of X if there are too few
    benign samples."""
    benign_mask = (y == 0)
    if benign_mask.sum() < 2:
        print("  WARNING: <2 benign samples to fit scaler, falling back to "
              "fitting on the full split")
        return StandardScaler().fit(X)
    return StandardScaler().fit(X[benign_mask])


def threshold_metrics(y_true, probs, threshold, tags=None):
    preds = (probs >= threshold).astype(int)
    res = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "classification_report": classification_report(
            y_true, preds, target_names=["benign", "adversarial"],
            output_dict=True),
    }
    if tags is not None:
        per_attack = {}
        for attack in sorted(set(tags) - {""}):
            m = tags == attack
            p = preds[m]
            per_attack[attack] = {
                "n": int(m.sum()),
                "pred_benign": float((p == 0).mean()),
                "pred_adversarial": float((p == 1).mean()),
            }
        res["adversarial_by_attack"] = per_attack
    return res


def evaluate(clf, X_te, y_te, tags_te, score_threshold, X_tr, y_tr):
    """The full test set is always scored (no held-out split needed) since
    zero test-domain data is used to pick the threshold. AUC is
    threshold-free. When score_threshold == 'fixed0.5' (default), the
    operating threshold is a flat 0.5. When score_threshold is
    'percentileNN', the threshold is the NN-th percentile of predict_proba
    scores on the train domain's own benign (X_tr/y_tr) samples."""
    probs_te = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, probs_te))

    pct = np.percentile(probs_te, [1, 25, 50, 75, 99])

    res = {
        "auc": auc,
        "n_test": int(len(y_te)),
        "threshold_0_5": threshold_metrics(y_te, probs_te, 0.5, tags_te),
        "prob_percentiles_1_25_50_75_99": pct.tolist(),
    }

    kind, pct_val = _parse_score_threshold(score_threshold)

    if kind == "percentile":
        train_benign_mask = (y_tr == 0)
        n_calib_benign = int(train_benign_mask.sum())
        if n_calib_benign == 0:
            raise ValueError(
                "--score_threshold percentile: the train set contains no "
                "benign (y=0) samples to calibrate the percentile "
                "threshold from")
        probs_tr = clf.predict_proba(X_tr)[:, 1]
        calib_probs = probs_tr[train_benign_mask]
        pct_thr = float(np.percentile(calib_probs, pct_val))
        res["threshold_percentile"] = threshold_metrics(
            y_te, probs_te, pct_thr, tags_te)
        res["scored_threshold"] = pct_thr
        res["percentile_value"] = pct_val
        res["n_calib_benign_for_percentile"] = n_calib_benign
    else:
        res["scored_threshold"] = 0.5

    return res


def main():
    args = parse_args()
    model_type, model_short_name, data_dir_name = detect_model_type(args.model_path)
    data_dir = os.path.join(args.data_base_dir, data_dir_name)
    out_dir = os.path.join(args.output_dir, model_short_name)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("SPARSE SAFETY-NEURON CLASSIFIER")
    print(f"  Model:        {args.model_path}")
    print(f"  Neuron pkls:  {len(args.neurons_pkl)}")
    print(f"  Train source: advbench + alpaca (same as train_safe_neurons_v2)")
    print(f"  Test source:  {data_dir} (benign vs adversarial)")
    print(f"  Output dir:   {out_dir}")
    print("=" * 70)

    print("\nLoading neuron sets...")
    configs, needed = load_neuron_sets(args.neurons_pkl)

    if args.acts_npz:
        print(f"\nLoading cached activations from {args.acts_npz} ...")
        data = np.load(args.acts_npz, allow_pickle=False)
        n_train = int(data["n_train"])
        y_tr = data["y_train"]
        y_te = data["y_test"]
        tags_te = data["tags_test"]
        acts = {}
        for k in data.files:
            if k.startswith("acts|"):
                l = k[5:]
                if l not in needed:
                    continue
                cols_cached = data[f"cols|{l}"]
                if not np.all(np.isin(needed[l], cols_cached)):
                    raise ValueError(
                        f"Cached npz missing columns for {l}; re-run "
                        f"extraction (cache built for a different pkl set)")
                pos = np.searchsorted(cols_cached, needed[l])
                acts[l] = data[k][:, pos]
        print(f"  {len(acts)} layers, {n_train} train + {len(y_te)} test samples")
        acts_all = acts
    else:
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

        print("\nStep 2: Loading train (advbench+alpaca) & test (casper) prompts...")
        train_texts, y_tr = load_train_prompts(args)
        test_texts, y_te, tags_te = load_test_prompts(args, data_dir)
        n_train = len(train_texts)

        all_texts = train_texts + test_texts
        all_prompts = [format_prompt(mt.tokenizer, t, chat_template_type)
                      for t in all_texts]

        print(f"\nStep 3: Extracting last-token activations "
              f"({len(needed)} layers, "
              f"{sum(len(v) for v in needed.values())} columns, "
              f"{len(all_prompts)} prompts)...")
        acts_all = extract_selected_activations(
            mt, all_prompts, needed, args.batch_size, args.device)

        cache = {"n_train": np.array(n_train), "y_train": y_tr,
                "y_test": y_te, "tags_test": tags_te}
        for l in needed:
            cache[f"cols|{l}"] = needed[l]
            cache[f"acts|{l}"] = acts_all[l]
        cache_path = os.path.join(out_dir, "sparse_acts_cache.npz")
        np.savez_compressed(cache_path, **cache)
        print(f"  Activations cached: {cache_path}")
        print("  (re-run instantly with --acts_npz, no GPU needed)")

        del mt
        torch.cuda.empty_cache()

    acts_train = {l: v[:n_train] for l, v in acts_all.items()}
    acts_test = {l: v[n_train:] for l, v in acts_all.items()}

    # -----------------------------------------------------------------------
    # Per pkl: one classifier, trained on advbench/alpaca, tested on
    # casper benign/adversarial. The scaler is fit once on the train
    # domain's benign subset and reused for test — see module docstring.
    # -----------------------------------------------------------------------
    all_results = {}
    for cfg in configs:
        run_name = cfg["name"]
        X_tr = gather_features(cfg["selected"], acts_train, needed)
        X_te = gather_features(cfg["selected"], acts_test, needed)

        print(f"\n--- {run_name} ({X_tr.shape[1]} features) ---")
        scaler = fit_benign_scaler(X_tr, y_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        clf = build_classifier(args)
        clf.fit(X_tr_s, y_tr)

        res = evaluate(clf, X_te_s, y_te, tags_te, args.score_threshold,
                       X_tr_s, y_tr)
        res["n_features"] = int(X_tr.shape[1])
        res["pkl"] = cfg["pkl"]
        all_results[run_name] = res

        m = res["threshold_0_5"]
        print(f"  AUC (threshold-free, full test set): {res['auc']:.4f}")
        is_percentile = "threshold_percentile" in res
        if is_percentile:
            mp = res["threshold_percentile"]
            print(f"  thr=percentile{res['percentile_value']:g}"
                  f"({res['scored_threshold']:.3g}) [PRIMARY]: "
                  f"acc={mp['accuracy']:.4f}  f1={mp['f1']:.4f}  "
                  f"(calibrated on {res['n_calib_benign_for_percentile']} "
                  f"TRAIN-domain (alpaca) benign samples — zero test-domain "
                  f"data used to pick this threshold, scored on the full "
                  f"test set)")
            print(f"  thr=0.5 (reference only, full test set): "
                  f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")
        else:
            print(f"  thr=0.5 [PRIMARY]:  "
                  f"acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                  f"(n_test={res['n_test']})")
        p1, p25, p50, p75, p99 = res["prob_percentiles_1_25_50_75_99"]
        print(f"  [diagnostic] P(adversarial) percentiles "
              f"1/25/50/75/99%: {p1:.3g} / {p25:.3g} / {p50:.3g} / {p75:.3g} / "
              f"{p99:.3g} (spread out, not spiked at ~0/~1, means "
              f"--C/--calibration are working)")
        thr_label = '0.5' if not is_percentile else '0.5 (reference)'
        print(f"  per-attack @ thr={thr_label}:")
        for attack, pa in m["adversarial_by_attack"].items():
            print(f"    adv[{attack:<9}] -> pred_benign {pa['pred_benign']:.2f} | "
                  f"pred_adversarial {pa['pred_adversarial']:.2f}")
        if is_percentile:
            print(f"  per-attack @ thr=percentile{res['percentile_value']:g} [PRIMARY]:")
            for attack, pa in res["threshold_percentile"]["adversarial_by_attack"].items():
                print(f"    adv[{attack:<9}] -> pred_benign {pa['pred_benign']:.2f} | "
                      f"pred_adversarial {pa['pred_adversarial']:.2f}")

        clf_path = os.path.join(out_dir, f"sparse_clf_{run_name}.pkl")
        with open(clf_path, "wb") as f:
            pickle.dump({"lr": clf, "train_scaler": scaler,
                        "neuron_dict": cfg["selected"],
                        "class_names": ["benign", "adversarial"],
                        "scored_threshold": res["scored_threshold"],
                        "note": "At inference on a NEW domain, reuse "
                                "train_scaler as-is (it was fit only on "
                                "the train domain's benign samples and "
                                "reused for test too, never refit per "
                                "domain) — just transform before calling "
                                "lr.predict_proba. Use the threshold stored "
                                "in 'scored_threshold' (not a hardcoded "
                                "0.5) — if this run used --score_threshold "
                                "percentileNN, 0.5 is not the intended "
                                "cutoff."}, f)
        print(f"  Classifier saved: {clf_path}")

    # -----------------------------------------------------------------------
    # Summary + save
    # -----------------------------------------------------------------------
    results_path = os.path.join(out_dir, "sparse_classifier_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": all_results}, f,
                  indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("SUMMARY (benign vs adversarial, tested on casper data)")
    print(f"  C={args.C}  calibration={args.calibration or 'none'}  "
          f"score_threshold={args.score_threshold}")
    print("=" * 70)
    if args.score_threshold.startswith("percentile"):
        print(f"{'Run':<12} {'#feat':>6} {'AUC':>7} {'Acc@pct':>8} "
              f"{'F1@pct':>7} {'thr':>10} {'Acc@0.5':>8} {'F1@0.5':>7}")
        print("-" * 78)
        for run_name, res in all_results.items():
            mp = res["threshold_percentile"]
            m = res["threshold_0_5"]
            print(f"{run_name:<12} {res['n_features']:>6} {res['auc']:>7.4f} "
                  f"{mp['accuracy']:>8.4f} {mp['f1']:>7.4f} "
                  f"{res['scored_threshold']:>10.3g} "
                  f"{m['accuracy']:>8.4f} {m['f1']:>7.4f}")
        print("-" * 78)
    else:
        print(f"{'Run':<12} {'#feat':>6} {'AUC':>7} {'Acc@0.5':>8} {'F1@0.5':>7} "
              f"{'P50':>8}")
        print("-" * 60)
        for run_name, res in all_results.items():
            m = res["threshold_0_5"]
            p50 = res["prob_percentiles_1_25_50_75_99"][2]
            print(f"{run_name:<12} {res['n_features']:>6} {res['auc']:>7.4f} "
                  f"{m['accuracy']:>8.4f} {m['f1']:>7.4f} {p50:>8.3g}")
        print("-" * 60)
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
