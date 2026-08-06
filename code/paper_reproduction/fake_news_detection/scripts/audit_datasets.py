from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

import pandas as pd


def load_pickle_df(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pkl.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(obj)!r}")
    return obj


def summarize_df(name: str, df: pd.DataFrame) -> None:
    print(f"\n## {name}")
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    for col in ["label", "category", "topic", "platform", "fine-grained label"]:
        if col in df.columns:
            print(f"\n{col} counts:")
            print(df[col].value_counts(dropna=False).to_string())


def check_finefake_images(df: pd.DataFrame, dataset_root: Path) -> None:
    if "image_path" not in df.columns:
        return
    paths = df["image_path"].dropna().astype(str)
    hits = 0
    misses = []
    for rel_path in paths:
        candidates = [
            dataset_root / rel_path,
            dataset_root / rel_path.lstrip("./"),
            dataset_root / "Image" / rel_path,
            dataset_root / "Image" / Path(rel_path).name,
        ]
        if any(c.exists() for c in candidates):
            hits += 1
        else:
            misses.append(rel_path)
    print(f"\nimage path check: {hits}/{len(paths)} paths found")
    if misses:
        print("missing sample paths:")
        for item in misses[:10]:
            print(f"  - {item}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="datasets",
    )
    args = parser.parse_args()

    root = Path(args.root)
    weibo_root = root / "WeiBo21"
    finefake_root = root / "FineFake" / "extracted"

    for name in ["train.pkl", "val.pkl", "test.pkl"]:
        df = load_pickle_df(weibo_root / name)
        summarize_df(f"WeiBo21/{name}", df)

    finefake_df = load_pickle_df(finefake_root / "FineFake.pkl")
    summarize_df("FineFake/FineFake.pkl", finefake_df)
    check_finefake_images(finefake_df, finefake_root)


if __name__ == "__main__":
    main()
