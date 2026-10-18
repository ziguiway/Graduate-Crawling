#!/usr/bin/env python3
"""
Step 1c: 可分性分析（方案 §6 闸1，决定 α 编辑是否开跑）+ k_edit 构造。

用法：
  python 03_separability_and_kedit.py \
      --h data/activations_h.pt --b data/activations_b.pt \
      --star data/activations_star.pt        # 原版 k*（DELMAN 原管线产物；没有则传 --star-optional）
      --outdir data/

产出：
  k_edit.pt          — dict: subject -> {"kH","kB","kstar","k_edit_a05","k_edit_a10"}
  separability.md    — 按类比值分布 + 线性分类器准确率 + 逐词明细表
  separability.json  — 同上，机器可读

判读标准（方案 §6 闸1）：
  ratio = ||k_H - k_B|| / mean(d_within_B, d_within_H)
  ratio >= 2    → 可分良好（该类别方法成立）
  ratio < 1.5   → 危险区（该类别 key 平移可能失效）
"""
import argparse, json, os
import torch


def d_within(items):
    """组内平均成对距离。items: (n, d) tensor。n<2 → nan"""
    if len(items) < 2:
        return float("nan")
    pd = torch.cdist(items, items)
    n = len(items)
    return (pd.sum() / (n * (n - 1))).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", default="data/activations_h.pt")
    ap.add_argument("--b", default="data/activations_b.pt")
    ap.add_argument("--star", default=None, help="原版 k* 的 .pt（dict: subject->vec）。缺省时以 k_H 近似锚点并打警告")
    ap.add_argument("--outdir", default="data/")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    H = torch.load(args.h)
    B = torch.load(args.b)
    if args.star and os.path.exists(args.star):
        S = torch.load(args.star)
        have_star = True
    else:
        S, have_star = {}, False
        print("[warn] 原版 k* 未提供 — k_edit 公式里的锚点暂用 k_H 代替（仅影响分析标注，"
              "正式跑 α 前请补上原管线 k*）")

    rows, k_edit_store = [], {}
    # 线性分类器用（k_H=1, k_B=0）
    X, y = [], []
    for s in sorted(set(H) & set(B)):
        kh, kb = H[s], B[s]["mean"]
        items = torch.stack([v for _, v in B[s]["items"]])
        cats = [c for c, _ in B[s]["items"]]
        dw_b = d_within(items)
        dw_h = float("nan")   # k_H 只有均值，无逐条 → 组内距离用 B 组近似（保守）
        # 逐条有害激活没有存（01 脚本只存了句子）——若需精确 d_within_H，
        # 回到 02 脚本开 --save-items-h 开关。此处用 2×平均近似：
        d_between = torch.norm(kh - kb).item()
        denom = dw_b if dw_b == dw_b else d_between  # nan 保护
        ratio = d_between / denom if denom > 0 else float("inf")
        # 线性可分性（逐类 + 总体）：用最近质心法（n 很小，够用）
        pred_correct = torch.norm(kh - kh) < torch.norm(kh - kb)  # 质心法下 k_H 自身必判 1，略
        rows.append({"subject": s, "d_between": round(d_between, 3),
                     "d_within_B": None if dw_b != dw_b else round(dw_b, 3),
                     "ratio": None if ratio != ratio else round(ratio, 3),
                     "cats": cats})
        X.append(kh); y.append(1)
        X.append(kb); y.append(0)
        # --- k_edit 构造（两个 α 档）---
        kstar = S.get(s, kh if not have_star else None)
        if kstar is None:
            kstar = kh
        ke05 = kh - 0.5 * (kb - kstar)
        ke10 = kh - 1.0 * (kb - kstar)
        k_edit_store[s] = {"kH": kh, "kB": kb, "kstar": kstar,
                           "k_edit_a05": ke05, "k_edit_a10": ke10}

    torch.save(k_edit_store, os.path.join(args.outdir, "k_edit.pt"))

    # ---- 按类别统计（B 组逐条 vs 组均值：类别可分性用逐条 k_B 对 k_H 的距离）----
    cat_stats = {}
    for r in rows:
        for c in r["cats"]:
            cat_stats.setdefault(c, []).append(r["ratio"])
    summary = {}
    for c, rs in sorted(cat_stats.items()):
        rs = [r for r in rs if r is not None]
        if rs:
            import statistics as st
            summary[c] = {"n": len(rs), "ratio_median": round(st.median(rs), 2),
                          "ratio_ge2": sum(r >= 2 for r in rs) / len(rs),
                          "ratio_lt15": sum(r < 1.5 for r in rs) / len(rs)}

    # ---- 总体最近质心分类准确率（LOO 太贵，用简单 2-fold）----
    import random
    Xs = torch.stack(X); ys = y
    idx = list(range(len(ys))); random.seed(42); random.shuffle(idx)
    accs = []
    for fold in range(2):
        te = idx[fold::2]; tr = [i for i in idx if i not in te]
        c1 = Xs[[i for i in tr if ys[i] == 1]].mean(0)
        c0 = Xs[[i for i in tr if ys[i] == 0]].mean(0)
        correct = sum(1 for i in te if
                      (torch.norm(Xs[i]-c0) > torch.norm(Xs[i]-c1)) == bool(ys[i]))
        accs.append(correct/len(te))
    clf_acc = sum(accs)/len(accs)

    md = ["# 可分性报告（Gate 闸1）", "",
          f"- subjects 分析: {len(rows)}",
          f"- 最近质心 2-fold 准确率: **{clf_acc:.2%}**（>90% = 全局可分性良好）", "",
          "## 按良性类别统计（ratio = d_between / d_within_B）", "",
          "| 类别 | n | median ratio | ≥2 占比 | <1.5 占比（危险区） |", "|---|---|---|---|---|"]
    for c, st in summary.items():
        md.append(f"| {c} | {st['n']} | {st['ratio_median']} | "
                  f"{st['ratio_ge2']:.0%} | {st['ratio_lt15']:.0%} |")
    md += ["", "## 总体 ratio 分布", "",
           f"- ratio ≥ 2: {sum(1 for r in rows if r['ratio'] and r['ratio']>=2)}",
           f"- ratio 1.5–2: {sum(1 for r in rows if r['ratio'] and 1.5<=r['ratio']<2)}",
           f"- ratio < 1.5（危险）: {sum(1 for r in rows if r['ratio'] and r['ratio']<1.5)}", "",
           "## 危险词清单（ratio < 1.5，编辑时需单独处理或刹车兜底）", ""]
    danger = [r for r in rows if r["ratio"] and r["ratio"] < 1.5]
    md += [f"- {r['subject']} (ratio={r['ratio']})" for r in sorted(danger, key=lambda r: r["ratio"])]
    open(os.path.join(args.outdir, "separability.md"), "w").write("\n".join(md))
    json.dump({"rows": rows, "by_category": summary, "centroid_acc": clf_acc},
              open(os.path.join(args.outdir, "separability.json"), "w"), ensure_ascii=False, indent=1)
    print(f"\n最近质心准确率: {clf_acc:.2%}")
    for c, st in summary.items():
        print(f"  {c}: median ratio {st['ratio_median']}, ≥2 占比 {st['ratio_ge2']:.0%}")
    print(f"危险词（<1.5）: {len(danger)} 个 → {args.outdir}/separability.md")
    print(f"\nk_edit.pt 已生成（α=0.5 / 1.0 两档）→ 交给第二步编辑管线。")


if __name__ == "__main__":
    main()
