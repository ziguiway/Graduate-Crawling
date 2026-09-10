"""Validate the processed LLM-MFEFND auxiliary-data contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "text",
    "image_path",
    "label",
    "llm_background",
    "llm_comment_join",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-csv", type=Path, required=True)
    parser.add_argument("--finefake-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    df = pd.read_csv(args.aux_csv)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(f"missing_columns={missing}")

    image_paths = df["image_path"].astype(str).map(lambda path: args.finefake_root / path)
    labels = pd.to_numeric(df["label"], errors="coerce")
    report = {
        "csv": str(args.aux_csv),
        "rows": len(df),
        "expected_rows": args.expected_rows,
        "row_count_ok": args.expected_rows is None or len(df) == args.expected_rows,
        "image_exists": int(image_paths.map(Path.exists).sum()),
        "image_missing": int((~image_paths.map(Path.exists)).sum()),
        "duplicate_text": int(df["text"].astype(str).duplicated().sum()),
        "invalid_labels": int(labels.isna().sum() + (~labels.dropna().isin([0, 1])).sum()),
        "label_counts": {str(int(label)): int(count) for label, count in labels.dropna().value_counts().sort_index().items()},
        "empty_background": int(df["llm_background"].fillna("").astype(str).str.strip().eq("").sum()),
        "empty_comments": int(df["llm_comment_join"].fillna("").astype(str).str.strip().eq("").sum()),
    }
    report["valid"] = bool(
        report["row_count_ok"]
        and report["image_missing"] == 0
        and report["invalid_labels"] == 0
        and report["empty_background"] == 0
        and report["empty_comments"] == 0
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
