#!/usr/bin/env python3
"""
Step 1b: 激活提取 — 对每个 subject 计算三个 key 向量：
  k_H（有害语境激活平均）、k_B（良性语境激活平均 + 逐条激活）、k*(原版对照，若提供原管线输出则跳过)。

用法：
  python 02_extract_activations.py \
      --model Qwen/Qwen2.5-7B-Instruct \
      --layer 19 \
      --harmful data/harmful_contexts.jsonl \
      --benign  data/benign_contexts.jsonl \
      --subjects data/subjects.json \
      --outdir data/

产出：
  activations_h.pt  — dict: subject -> k_H (d_model,)         平均
  activations_b.pt  — dict: subject -> {"mean": (d,), "items": [(category, vec)]}
  activations_meta.json — 提取参数（层号/位置规则/模型），供审查

层号说明：
  DELMAN 论文 Qwen2.5-7B 的 target layers 为 [25,26,27]（LED 表），其"关键层"沿用
  MEMIT 系结论。这里 --layer 必须与 DELMAN 编辑管线实际用的 layer 对齐！
  跑之前先看 DELMAN 仓库 hparams/Qwen2.5-7B 的 layer 配置，保持一致。
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, required=True, help="关键层号（与 DELMAN 编辑层对齐）")
    ap.add_argument("--harmful", default="data/harmful_contexts.jsonl")
    ap.add_argument("--benign", default="data/benign_contexts.jsonl")
    ap.add_argument("--subjects", default="data/subjects.json")
    ap.add_argument("--outdir", default="data/")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--locate", choices=["all_tokens_mean", "first_token"],
                    default="all_tokens_mean",
                    help="subject 跨多 token 时取所有位置平均（方案 §2.3 推荐）或首 token")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    layers = model.model.layers
    L = args.layer

    def mlp_mid_states(texts):
        """返回每条文本在 L 层 MLP 中间态（σ(W_gate·x)⊙(W_up·x)），shape (n_seq, d)。"""
        enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        attn = enc["attention_mask"]
        feats = {}
        def hook(mod, inp, out):
            feats["x"] = inp[0].detach()          # L 层 MLP 的输入 hidden state
        h = layers[L].mlp.register_forward_hook(hook)
        model(**enc)
        h.remove()
        x = feats["x"]                             # (B, T, d)
        gate = torch.nn.functional.silu(layers[L].mlp.gate_proj(x))
        up = layers[L].mlp.up_proj(x)
        mid = gate * up                            # (B, T, d) — 与 DELMAN Eq.2 的 σ(W_gate x)⊙(W_up x) 对齐
        return mid, enc["input_ids"], attn

    def subject_positions(input_ids, mask, subject):
        """定位 subject 的 token 位置（对每条序列）。找不到则回退到最后一个非 pad token。"""
        sub_ids = tok.encode(" " + subject, add_special_tokens=False) or tok.encode(subject, add_special_tokens=False)
        n = len(sub_ids)
        out = []
        ids = input_ids.tolist()
        for bi in range(ids.shape[0]):
            row = ids[bi]
            pos = -1
            for t in range(len(row) - n + 1):
                if row[t:t + n] == sub_ids:
                    pos = list(range(t, t + n))    # 取最后一次出现（prompt 里 subject 通常一次）
            if pos < 0:
                valid = [t for t, m in enumerate(mask[bi].tolist()) if m == 1]
                pos = [valid[-1]]
                print(f"[warn] subject '{subject}' not tokenized in seq {bi}; fallback to last token")
            if args.locate == "first_token":
                pos = pos[:1]
            out.append(pos)
        return out

    def batch_extract(texts, subject):
        """返回 (n, d) 张量：每条文本 subject 位置的 MLP 中间态。"""
        vecs = []
        for i in range(0, len(texts), args.batch_size):
            chunk = texts[i:i + args.batch_size]
            mid, ids, mask = mlp_mid_states(chunk)
            poss = subject_positions(ids, mask, subject)
            for bi, pos in enumerate(poss):
                v = mid[bi, pos, :].mean(dim=0).float().cpu()
                vecs.append(v)
        return torch.stack(vecs)

    # 读取句子池
    harm, benign = {}, {}
    with open(args.harmful, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            harm.setdefault(d["subject"], []).append(d)
    with open(args.benign, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            benign.setdefault(d["subject"], []).append(d)

    subjects = json.load(open(args.subjects))
    KH, KB, missing_b = {}, {}, []
    for gi, g in enumerate(subjects):
        s = g["subject"]
        # --- k_H ---
        if s in harm and harm[s]:
            kh = batch_extract([d["text"] for d in harm[s]], s).mean(dim=0)
            KH[s] = kh
        else:
            print(f"[warn] no harmful contexts for '{s}' (dry-run?); skip k_H")
        # --- k_B ---
        if s in benign and benign[s]:
            items = batch_extract([d["text"] for d in benign[s]], s)
            KB[s] = {"mean": items.mean(dim=0),
                     "items": [(benign[s][j]["category"], items[j]) for j in range(len(benign[s]))]}
        else:
            missing_b.append(s)
    if missing_b:
        print(f"[warn] {len(missing_b)} subjects without benign contexts: {missing_b[:5]} ...")

    torch.save(KH, os.path.join(args.outdir, "activations_h.pt"))
    torch.save(KB, os.path.join(args.outdir, "activations_b.pt"))
    json.dump({"model": args.model, "layer": args.layer, "locate": args.locate,
               "n_subjects": len(subjects), "n_kH": len(KH), "n_kB": len(KB)},
              open(os.path.join(args.outdir, "activations_meta.json"), "w"), indent=1)
    print(f"saved: k_H×{len(KH)}, k_B×{len(KB)} → {args.outdir}")


if __name__ == "__main__":
    main()
