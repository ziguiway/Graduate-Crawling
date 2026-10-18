#!/usr/bin/env python3
"""
激活提取（DELMAN 对齐版）—— 服务器上运行。

与 DELMAN k* 计算严格对齐：
  - 表示 = 指定层 down_proj 的输入（module input，与 compute_k_star 的 l_input 一致）
  - subject 定位 = subject_last（最后一个 subtoken，与 fact_token 一致）
  - 提取路径 = 直接复用 DELMAN 的 repr_tools.get_reprs_at_word_tokens
  - 顺手复算原版 k*（DELMAN 的 context_templates，["{}"] + 4 条半随机前缀）

用法：
  python scripts/02_extract_activations_delman.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --layers 7 8 \
      --outdir data/
"""
import argparse, json, os, sys, hashlib, datetime

# 让 DELMAN 仓库可 import
sys.path.insert(0, "/root/data/code/DELMAN")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from delman.compute_v_star import get_module_input_output_at_words
from delman.delman_main import get_context_templates


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layers", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--sampling-dir", default="/root/data/code/delman_key_sampling/data")
    ap.add_argument("--outdir", default="/root/data/code/delman_key_sampling/data")
    ap.add_argument("--subjects", default="data/subjects.json")
    ap.add_argument("--log", default="logs/extract.log")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    log_f = open(args.log, "a")
    def log(msg):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    log(f"=== activation extraction start | model={args.model} layers={args.layers} ===")
    log(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    log("model loaded")

    # 句池
    harm = load_jsonl(os.path.join(args.sampling_dir, "harmful_contexts.jsonl"))
    benign = load_jsonl(os.path.join(args.sampling_dir, "benign_contexts.jsonl"))
    subjects = json.load(open(os.path.join(args.sampling_dir, "subjects.json")))
    by_s_h, by_s_b = {}, {}
    for d in harm:
        by_s_h.setdefault(d["subject"], []).append(d["text"])
    for d in benign:
        by_s_b.setdefault(d["subject"], []).append(d["text"])
    log(f"subjects={len(subjects)} harmful_words={len(by_s_h)} benign_words={len(by_s_b)}")

    # DELMAN 原版 context templates（用于复算 k*）
    ctx_templates = get_context_templates(model, tok)
    log(f"context_templates loaded: {ctx_templates}")

    module_template = "model.layers.{}.mlp.down_proj"

    results = {}   # subject -> {"layer": {"kH": vec, "kB": {"mean","items":[(cat,vec)]}, "kstar": vec}}
    for si, g in enumerate(subjects):
        s = g["subject"]
        entry = {}
        for layer in args.layers:
            ent = {}
            # ---- k_H ----
            texts = by_s_h.get(s, [])
            if texts:
                words = [s] * len(texts)
                l_in, _ = get_module_input_output_at_words(
                    model, tok, layer,
                    context_templates=[texts],
                    words=words,
                    module_template=module_template,
                    fact_token_strategy="subject_last",
                )
                ent["kH"] = l_in.float().cpu().mean(0)
                ent["kH_items"] = l_in.float().cpu()
            # ---- k_B ----
            texts_b = by_s_b.get(s, [])
            if texts_b:
                cats = [d["category"] for d in benign if d["subject"] == s]
                words_b = [s] * len(texts_b)
                l_in_b, _ = get_module_input_output_at_words(
                    model, tok, layer,
                    context_templates=[texts_b],
                    words=words_b,
                    module_template=module_template,
                    fact_token_strategy="subject_last",
                )
                ent["kB_mean"] = l_in_b.float().cpu().mean(0)
                ent["kB_items"] = [(cats[i] if i < len(cats) else "?", l_in_b.float().cpu()[i])
                                   for i in range(len(texts_b))]
            # ---- k*（DELMAN 原版口径复算）----
            temps = [t.format(s) for t in ctx_templates[0]] + \
                    [t.format(s) for t in ctx_templates[1]] if len(ctx_templates) > 1 \
                    else [t.format(s) for t in ctx_templates[0]]
            words_star = [s] * len(temps)
            l_in_star, _ = get_module_input_output_at_words(
                model, tok, layer,
                context_templates=[temps],
                words=words_star,
                module_template=module_template,
                fact_token_strategy="subject_last",
            )
            ent["kstar"] = l_in_star.float().cpu().mean(0)
            entry[str(layer)] = ent
        results[s] = entry
        if (si + 1) % 20 == 0 or si == len(subjects) - 1:
            log(f"progress {si+1}/{len(subjects)} ({s[:30]})")

    # 保存（按层拆两个文件 + 一个 k*）
    for layer in args.layers:
        kh, kb = {}, {}
        for s, entry in results.items():
            ent = entry[str(layer)]
            if "kH" in ent:
                kh[s] = ent["kH"]
            if "kB_mean" in ent:
                kb[s] = {"mean": ent["kB_mean"], "items": ent["kB_items"]}
        torch.save(kh, os.path.join(args.outdir, f"activations_h_L{layer}.pt"))
        torch.save(kb, os.path.join(args.outdir, f"activations_b_L{layer}.pt"))
        log(f"saved L{layer}: kH×{len(kh)} kB×{len(kb)}")
    kstar = {s: {str(l): results[s][str(l)]["kstar"] for l in args.layers} for s in results}
    torch.save(kstar, os.path.join(args.outdir, "activations_star_repro.pt"))
    log("saved kstar_repro")

    meta = {
        "model": args.model, "layers": args.layers,
        "module": "down_proj input (l_input), subject_last",
        "repr_path": "DELMAN repr_tools.get_reprs_at_word_tokens",
        "n_subjects": len(subjects),
        "files_sha256": {f: sha256(os.path.join(args.outdir, f))
                         for f in os.listdir(args.outdir) if f.endswith(".pt")},
        "utc8": datetime.datetime.now().isoformat(),
    }
    json.dump(meta, open(os.path.join(args.outdir, "activations_meta.json"), "w"),
              ensure_ascii=False, indent=1)
    log("=== done ===")
    log_f.close()


if __name__ == "__main__":
    main()
