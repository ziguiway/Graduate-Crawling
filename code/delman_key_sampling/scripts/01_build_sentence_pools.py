#!/usr/bin/env python3
"""
Step 1a: 按 subject 聚合 200 条编辑用例 → 生成三类句子清单。

产出（都在 data/ 下）：
  subjects.json          — 177 个唯一 subject 的元数据（类型标注、所属 case_id 列表）
  harmful_contexts.jsonl — 线1：每 subject ≤4 条有害语境句（原 query + 同义改写）
  benign_contexts.jsonl  — 线2：每 subject ≤6 条良性语境句（四类标签 B1-B4）
  benign_calibration.jsonl — 线4：KL 宽校准集（= benign_contexts 平铺，原管线格式）

用法：
  python 01_build_sentence_pools.py --cases data/edit_cases.json --outdir data/ \
      --rewrite-model Qwen/Qwen2.5-72B-Instruct --benign-model Qwen/Qwen2.5-72B-Instruct

说明：
  - 改写和良性句生成都走本地 vLLM/HF 接口（服务器上有 72B）；若想先小规模跑，
    用 --dry-run 只用原 query 不调生成模型（改写与良性句置空，流程仍可走通）。
  - 良性句生成 prompt 明确要求"讨论该话题但绝不提供有害细节"，仍需人工抽查（方案 §6 闸2）。
"""
import argparse, json, os, re, collections

BENIGN_TEMPLATES = {
    # 四类配额：B1 新闻×2, B2 比喻×1, B3 教育/科学×2, B4 安全研究×1
    "B1_news": (
        "Write ONE short English sentence (15-25 words) discussing the topic \"{subject}\" "
        "in a news-reporting or historical-explanation context. The sentence must be strictly "
        "informational and harmless — it must NOT ask for, provide, or imply any harmful "
        "instructions. Output ONLY the sentence, no quotes, no explanation."
    ),
    "B2_metaphor": (
        "Write ONE short English sentence (12-20 words) where the phrase \"{subject}\" is used "
        "metaphorically, colloquially, or in a benign everyday context (e.g. describing a sports "
        "victory, a successful event, a joke). The sentence must be strictly harmless. "
        "Output ONLY the sentence."
    ),
    "B3_education": (
        "Write ONE short English question (15-25 words) asking for factual, educational "
        "information about \"{subject}\" (definition, history, mechanism, chemistry background, "
        "etc.), as a curious student would. The question must be strictly informational and "
        "harmless — it must NOT request operational instructions. Output ONLY the question."
    ),
    "B4_safety_research": (
        "Write ONE short English question (15-25 words) about how security researchers, doctors, "
        "or regulators detect, prevent, or defend against \"{subject}\". The question must be "
        "strictly defensive and harmless. Output ONLY the question."
    ),
}

QUOTA = [("B1_news", 2), ("B2_metaphor", 1), ("B3_education", 2), ("B4_safety_research", 1)]

# 动词短语 / 整句 subject：比喻类通常造不出自然句子 → 降级（方案 §3.2 坑1）
WORDISH_RE = re.compile(r"^[a-zA-Z][a-zA-Z .\-']*$")


def subject_type(subject: str) -> str:
    """粗分三类：noun(名词/专有名词短语)、verb(动词或动词短语)、sentence(整句)。"""
    s = subject.strip().lower()
    if len(s.split()) >= 5:
        return "sentence"
    verb_heads = {"steal", "forge", "hijacks", "hack", "hacking", "coerce", "insult",
                  "bully", "harasses", "bullies", "manipulate", "tamper", "fool",
                  "fooling", "corrupts", "destroy", "scraping", "downplays",
                  "threatening", "stealing", "using drugs", "gain unauthorized access",
                  "illegally disrupt", "unlawfully tap", "without official permission",
                  "secretly biased toward", "deserved to die", "against a specific utility company"}
    if s in verb_heads:
        return "verb"
    if s.startswith(("being ", "kill ", "hiring ", "holding ", "buy ", "sell ", "selling ",
                     "eating ", "eat ", "feed ", "huff ", "commit ", "smuggle ", "forging ",
                     "killing ", "mixing ", "inciting ", "using ", "snorting ", "removal",
                     "annexation")):
        return "verb"
    return "noun"


def load_and_group(cases_path):
    cases = json.load(open(cases_path))
    by_subject = collections.OrderedDict()
    for c in cases:
        r = c["requested_rewrite"]
        s = r["subject"]
        e = by_subject.setdefault(s, {"subject": s, "case_ids": [], "prompts": []})
        e["case_ids"].append(c["case_id"])
        e["prompts"].append(r["prompt_adv"].format(s))
    for e in by_subject.values():
        e["stype"] = subject_type(e["subject"])
    return list(by_subject.values())


def try_llm(model_name, prompts, max_new_tokens=80):
    """本地生成接口：优先 vLLM 的 OpenAI 兼容 server（LARK/服务器默认 8000 端口），
    失败则退回 transformers pipeline。返回与 prompts 等长的文本列表。"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
        out = []
        for p in prompts:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": p}],
                temperature=0.9, max_tokens=max_new_tokens,
            )
            out.append(resp.choices[0].message.content.strip().strip('"').strip())
        return out
    except Exception as ex:
        print(f"[warn] OpenAI-compatible server failed ({ex}); falling back to HF pipeline")
        from transformers import pipeline
        pipe = pipeline("text-generation", model=model_name, device_map="auto",
                        torch_dtype="auto", trust_remote_code=True)
        outs = pipe(prompts, max_new_tokens=max_new_tokens, do_sample=True,
                    temperature=0.9, batch_size=16)
        res = []
        for o in outs:
            t = o[0]["generated_text"]
            if isinstance(t, list):  # chat 格式
                t = t[-1]["content"]
            res.append(t.strip().strip('"').strip())
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/edit_cases.json")
    ap.add_argument("--outdir", default="data/")
    ap.add_argument("--rewrite-model", default=None, help="同义改写用模型；不设则 --dry-run")
    ap.add_argument("--benign-model", default=None, help="良性句生成用模型；不设则 --dry-run")
    ap.add_argument("--dry-run", action="store_true", help="不调生成模型：有害句只用原 query，良性句置空")
    ap.add_argument("--limit-subjects", type=int, default=0, help="调试：只处理前 N 个 subject")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    groups = load_and_group(args.cases)
    if args.limit_subjects:
        groups = groups[: args.limit_subjects]
    print(f"subjects: {len(groups)} "
          f"({collections.Counter(g['stype'] for g in groups)})")

    # ---------- 线1：有害语境句 ----------
    # 来源 A：原 query（天然多样，复用 subject 天然 ≥2 条）
    # 来源 B：同义改写 ×2（需要生成模型）
    harmful_out = open(os.path.join(args.outdir, "harmful_contexts.jsonl"), "w", encoding="utf-8")
    rewrite_jobs = []  # (subject_idx, template-filled rewrite prompts)
    for gi, g in enumerate(groups):
        n_orig = len(g["prompts"])
        n_need = max(0, 4 - n_orig)          # 还差的条数
        n_need = min(n_need, 2)              # 最多补 2 条改写
        s = g["subject"]
        rewrite_jobs.append([
            f"Rewrite the following request into a DIFFERENT sentence structure but with the "
            f"IDENTICAL harmful intent (paraphrase, do not answer it):\n\n{p.format(s) if '{}' in p else p}\n\n"
            f"Output ONLY the rewritten request."
            for p in g["prompts"][:1] for _ in range(n_need)
        ] or [[] for _ in range(0)])
    flat_rw = [p for jobs in rewrite_jobs for p in jobs]
    rw_results = [[] for _ in groups]
    if not args.dry_run and args.rewrite_model and flat_rw:
        outs = try_llm(args.rewrite_model, flat_rw)
        it = iter(outs)
        for gi, jobs in enumerate(rewrite_jobs):
            rw_results[gi] = [next(it) for _ in jobs]

    for gi, g in enumerate(groups):
        for p in g["prompts"]:
            harmful_out.write(json.dumps({
                "subject": g["subject"], "stype": g["stype"],
                "source": "original_query", "case_id": g["case_ids"][0],
                "text": p}, ensure_ascii=False) + "\n")
        for p in rw_results[gi]:
            if p:
                harmful_out.write(json.dumps({
                    "subject": g["subject"], "stype": g["stype"],
                    "source": "paraphrase", "case_id": g["case_ids"][0],
                    "text": p}, ensure_ascii=False) + "\n")
    harmful_out.close()

    # ---------- 线2：良性语境句 ----------
    benign_jobs, benign_meta = [], []
    for gi, g in enumerate(groups):
        stype = g["stype"]
        quota = QUOTA if stype == "noun" else [q for q in QUOTA if q[0] != "B2_metaphor"]
        for cat, n in quota:
            for k in range(n):
                benign_jobs.append(BENIGN_TEMPLATES[cat].format(subject=g["subject"]))
                benign_meta.append((gi, cat, n, k))
    benign_out = open(os.path.join(args.outdir, "benign_contexts.jsonl"), "w", encoding="utf-8")
    if not args.dry_run and args.benign_model and benign_jobs:
        outs = try_llm(args.benign_model, benign_jobs, max_new_tokens=60)
    else:
        outs = [""] * len(benign_jobs)
    seen = collections.defaultdict(set)
    for (gi, cat, n, k), text in zip(benign_meta, outs):
        if not text:
            continue
        s = groups[gi]["subject"]
        norm = re.sub(r"\W+", "", text.lower())
        if norm in seen[s]:                       # 同 subject 去重
            continue
        seen[s].add(norm)
        benign_out.write(json.dumps({
            "subject": s, "stype": groups[gi]["stype"], "category": cat,
            "text": text}, ensure_ascii=False) + "\n")
    benign_out.close()

    # ---------- 线4：KL 宽校准集（= 良性句平铺） ----------
    with open(os.path.join(args.outdir, "benign_contexts.jsonl"), encoding="utf-8") as f, \
         open(os.path.join(args.outdir, "benign_calibration.jsonl"), "w", encoding="utf-8") as w:
        for line in f:
            d = json.loads(line)
            w.write(json.dumps({"text": d["text"], "subject": d["subject"],
                                "category": d["category"]}, ensure_ascii=False) + "\n")

    # ---------- subjects 元数据 ----------
    with open(os.path.join(args.outdir, "subjects.json"), "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=1)

    print("done. outputs in", args.outdir)
    print("  NOTE: 人工抽查 10% 良性句（方案 §6 闸2）后再进激活提取。")


if __name__ == "__main__":
    main()
