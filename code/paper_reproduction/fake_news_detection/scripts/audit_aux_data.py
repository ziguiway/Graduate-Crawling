from __future__ import annotations

import pickle as pkl
from pathlib import Path

import pandas as pd


def load_pickle_df(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pkl.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(obj)!r}")
    return obj


def main() -> None:
    data_root = Path("datasets")
    aux_root = Path("LLM-MFEFND/data")

    finefake = load_pickle_df(data_root / "FineFake" / "extracted" / "FineFake.pkl")
    finefake_texts = set(finefake["text"].astype(str))
    finefake_aux = pd.read_csv(aux_root / "GPT-DS-GLM-Weibo21-FineFake.csv")
    finefake_matches = finefake_aux["text"].astype(str).isin(finefake_texts).sum()
    print("FineFake auxiliary data")
    print(f"rows: {len(finefake_aux)}")
    print(f"text matches: {finefake_matches}/{len(finefake_aux)}")

    weibo_parts = [
        load_pickle_df(data_root / "WeiBo21" / name)
        for name in ["train.pkl", "val.pkl", "test.pkl"]
    ]
    weibo = pd.concat(weibo_parts, ignore_index=True)
    weibo_texts = set(weibo["content"].astype(str))
    print("\nWeiBo21 auxiliary data")
    for name in [
        "GPT-DS-GLM-Weibo21.csv",
        "background_new_git.csv",
        "comments_new_git.csv",
        "content_new_git.csv",
        "background_new.csv",
        "comments_new.csv",
        "content_new.csv",
    ]:
        df = pd.read_csv(aux_root / name)
        matches = df["content"].astype(str).isin(weibo_texts).sum()
        print(f"{name}: rows={len(df)}, content matches={matches}/{len(df)}")


if __name__ == "__main__":
    main()
