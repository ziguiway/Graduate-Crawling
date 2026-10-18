#!/usr/bin/env python3
"""
做法 A：模板填空造句池（零生成、零依赖）。
产出（outdir 下）：
  harmful_contexts.jsonl      — 有害语境句（= 原 200 条 query，按 subject 分组）
  benign_contexts.jsonl       — 良性语境句（四类模板 × 177 词，noun 全配额 6 条，verb/sentence 5 条）
  benign_calibration.jsonl    — KL 宽校准集（= 良性句平铺，第二步"只宽 KL"消融直接可用）
  pools_summary.json          — 统计摘要
"""
import argparse, json, os, re, collections

# ---------------- 良性句模板（每类多个骨架保证多样性） ----------------

B1_NEWS = [  # 新闻陈述 ×2
    "The news reported new findings related to {s} last week.",
    "A documentary recently examined the history and impact of {s}.",
    "Journalists covered the public debate surrounding {s} in detail.",
    "A recent article discussed how society has responded to concerns about {s}.",
]
B2_METAPHOR = [  # 比喻/口语 ×1
    "The team's victory celebration was the highlight of the season, like a festival around {s}.",
    "Her birthday party was so much fun that everyone called it the {s} of the year.",
]
B3_EDU = [  # 教育/科学提问 ×2
    "What is the scientific definition of {s}?",
    "Can you explain the basic mechanism behind {s} for my homework?",
    "What is the history of {s} and how did people first discover it?",
    "How does {s} work, explained at a high school science level?",
]
B4_DEFENSE = [  # 安全/防御提问 ×1
    "How do researchers and regulators work to detect and prevent {s}?",
    "What measures do experts recommend for protecting the public from {s}?",
]

# 动词/整句 subject：跳过比喻类（方案 §3.2 坑1）
VERB_SENTENCE_B = [("B1_news", B1_NEWS, 2), ("B3_education", B3_EDU, 2), ("B4_defense", B4_DEFENSE, 1)]
NOUN_B = [("B1_news", B1_NEWS, 2), ("B2_metaphor", B2_METAPHOR, 1),
          ("B3_education", B3_EDU, 2), ("B4_defense", B4_DEFENSE, 1)]


def subject_type(subject: str) -> str:
    s = subject.strip().lower()
    # 名词短语优先：以冠词/数词/形容词性开头、或以名词性复数结尾的，即使长也算 noun
    if re.match(r"^(the |a |an |fake |stolen |illegal |private |commercial |black |child )", s) \
       or s.endswith(("attacks", "agents", "weapons", "injection", "records", "people", "slurs",
                      "practices", "shops", "cars", "goods", "communities", "jokes", "allegations",
                      "children", "companies", "emails", "page", "alarm", "cats", "alcohol")) and len(s.split()) >= 2:
        return "noun"
    # 动词/动名词/介词开头的动宾短语 → verb（如 "feed lillies to their cats", "against X"）
    if s.startswith(("being ", "kill ", "hiring ", "holding ", "buy ", "sell ", "selling ",
                     "eating ", "eat ", "feed ", "huff ", "commit ", "smuggle ", "forging ",
                     "killing ", "mixing ", "inciting ", "using ", "snorting ", "removal",
                     "annexation", "against ", "without official")):
        return "verb"
    if len(s.split()) >= 5:
        return "sentence"
    verb_heads = {"steal", "forge", "hijacks", "hack", "hacking", "coerce", "insult",
                  "bully", "harasses", "bullies", "manipulate", "tamper", "fool",
                  "fooling", "corrupts", "destroy", "scraping", "downplays",
                  "threatening", "stealing", "using drugs", "gain unauthorized access",
                  "illegally disrupt", "unlawfully tap", "without official permission",
                  "secretly biased toward", "deserved to die"}
    if s in verb_heads:
        return "verb"
    return "noun"


def fill(template: str, subject: str) -> str:
    """把 subject 填进模板，处理大小写与冠词：句中 {s} 直接填；句首首字母大写。"""
    t = template.replace("{s}", subject)
    return t[0].upper() + t[1:]


# --- 语法适配：模板是"名词槽"，动词类 subject 直接填会产生病句（"surrounding steal"）---
# 动词→动名词（steal→stealing），第三人称动词形式先还原再 -ing。手工映射表最可靠。
GERUND_MAP = {
    "steal": "stealing", "stealing": "stealing", "forge": "forging",
    "hijacks": "hijacking", "corrupts": "corrupting", "hack": "hacking",
    "hacking": "hacking", "coerce": "coercing", "insult": "insulting",
    "bully": "bullying", "bullies": "bullying", "harasses": "harassing",
    "manipulate": "manipulating", "tamper": "tampering", "fool": "fooling",
    "fooling": "fooling", "destroy": "destroying", "scraping": "scraping",
    "downplays": "downplaying", "threatening": "threatening", "using drugs": "drug use",
    "gain unauthorized access": "gaining unauthorized access",
    "illegally disrupt": "illegal disruption of", "unlawfully tap": "unlawful tapping of",
    "without official permission": "without official permission",
    "secretly biased toward": "secret bias toward", "deserved to die": "the mentioned atrocity",
    "against a specific utility company": "an attack on utility companies",
    "consent is completely unnecessary": "the mentioned harmful claim",
    "disabilities are a burden": "the mentioned harmful claim",
    "women being inherently less intelligent than men": "the mentioned harmful claim",
    # 动宾/整句里的"裸动词短语"填进名词槽的兜底适配（kill X 类都是病句，降格为"行为"指称）
    "kill a two year old child": "the mentioned violent act",
    "holding a commercial flight hostage": "the mentioned act of violence",
    "kill lots of people": "mass violence",
    "feed lillies to their cats": "feeding lilies to cats",
}


def adapt(subject: str, stype: str) -> str:
    """名词槽适配：verb 类转成名词化形式；其他原样。
    映射表优先：任何在 GERUND_MAP 里的词无论 stype 都适配（处理误判成 noun 的动宾短语）。
    没在映射表的 verb → 兜底降格为 "the mentioned act"（保证语法成立，语义由上下文承载）。"""
    if subject.strip().lower() in GERUND_MAP:
        return GERUND_MAP[subject.strip().lower()]
    if stype == "verb":
        return "the mentioned act"
    return subject


# 整句/动宾 subject 直接填进"讨论模板"会产生病句（"definition of kill lots of people"）。
# 对这类 subject 改用"事件/议题"包装模板：把整句降格为被讨论的现象描述。
SENTENCE_TEMPLATES = {
    "B1_news": [
        "The news reported on the disturbing phenomenon of {s} last week.",
        "A documentary recently examined the social harm caused by {s}.",
    ],
    "B3_education": [
        "How do psychologists and sociologists study and explain the causes of {s}?",
        "What research exists on understanding and preventing {s}?",
    ],
    "B4_defense": [
        "What measures do experts recommend for protecting the public from {s}?",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/edit_cases.json")
    ap.add_argument("--outdir", default="data/")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    cases = json.load(open(args.cases, encoding="utf-8"))

    # ---------- 按 subject 分组 ----------
    by_subject = collections.OrderedDict()
    for c in cases:
        r = c["requested_rewrite"]
        e = by_subject.setdefault(r["subject"], {"subject": r["subject"], "cases": []})
        e["cases"].append({"case_id": c["case_id"], "prompt": r["prompt_adv"].format(r["subject"])})
    for e in by_subject.values():
        e["stype"] = subject_type(e["subject"])
    print(f"subjects: {len(by_subject)} "
          f"({dict(collections.Counter(e['stype'] for e in by_subject.values()))})")

    # ---------- 有害句池：原 query 直接入库 ----------
    n_h = 0
    with open(os.path.join(args.outdir, "harmful_contexts.jsonl"), "w", encoding="utf-8") as f:
        for e in by_subject.values():
            for c in e["cases"]:
                f.write(json.dumps({"subject": e["subject"], "stype": e["stype"],
                                    "source": "original_query", "case_id": c["case_id"],
                                    "text": c["prompt"]}, ensure_ascii=False) + "\n")
                n_h += 1

    # ---------- 良性句池：模板填空 + 去 重 ----------
    n_b, per_subject_counts = 0, {}
    with open(os.path.join(args.outdir, "benign_contexts.jsonl"), "w", encoding="utf-8") as f:
        for e in by_subject.values():
            stype = e["stype"]
            written = 0
            if stype == "sentence":
                # 整句 subject：用"事件/议题"包装模板（避免病句），共 5 条
                quota = [("B1_news", SENTENCE_TEMPLATES["B1_news"], 2),
                         ("B3_education", SENTENCE_TEMPLATES["B3_education"], 2),
                         ("B4_defense", SENTENCE_TEMPLATES["B4_defense"], 1)]
                for cat, tmpls, k in quota:
                    for t in tmpls[:k]:
                        text = fill(t, e["subject"])
                        f.write(json.dumps({"subject": e["subject"], "stype": stype,
                                            "category": cat, "text": text},
                                           ensure_ascii=False) + "\n")
                        written += 1
            else:
                quota = NOUN_B if stype == "noun" else VERB_SENTENCE_B
                adapted = adapt(e["subject"], stype)   # 动词类→动名词/降格，保证模板句语法成立
                # 若适配词与原词不同（动词类），跳过比喻类（"the feed lillies to their cats of the year" 仍是病句）
                if adapted != e["subject"]:
                    quota = [q for q in quota if q[0] != "B2_metaphor"]
                for cat, tmpls, k in quota:
                    # 每类从骨架池轮转取样，保证跨 subject 时骨架多样
                    picks = [tmpls[(hash(e["subject"]) + i) % len(tmpls)] for i in range(k)]
                    for t in picks:
                        text = fill(t, adapted)
                        f.write(json.dumps({"subject": e["subject"], "stype": stype,
                                            "category": cat, "text": text},
                                           ensure_ascii=False) + "\n")
                        written += 1
            per_subject_counts[e["subject"]] = written
            n_b += written

    # ---------- KL 校准集（平铺） ----------
    with open(os.path.join(args.outdir, "benign_contexts.jsonl"), encoding="utf-8") as f, \
         open(os.path.join(args.outdir, "benign_calibration.jsonl"), "w", encoding="utf-8") as w:
        for line in f:
            d = json.loads(line)
            w.write(json.dumps({"text": d["text"], "subject": d["subject"],
                                "category": d["category"]}, ensure_ascii=False) + "\n")

    # ---------- 统计摘要 ----------
    cat_dist = collections.Counter()
    with open(os.path.join(args.outdir, "benign_contexts.jsonl"), encoding="utf-8") as f:
        for line in f:
            cat_dist[json.loads(line)["category"]] += 1
    summary = {
        "n_subjects": len(by_subject),
        "stype_dist": dict(collections.Counter(e["stype"] for e in by_subject.values())),
        "n_harmful": n_h,
        "n_benign": n_b,
        "benign_per_subject": {"min": min(per_subject_counts.values()),
                               "max": max(per_subject_counts.values())},
        "benign_category_dist": dict(cat_dist),
    }
    json.dump(summary, open(os.path.join(args.outdir, "pools_summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
