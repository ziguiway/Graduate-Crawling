from __future__ import annotations

import argparse
import pickle as pkl
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_pickle_df(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pkl.load(f)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Expected DataFrame in {path}, got {type(obj)!r}")
    return obj


def load_weibo21(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    data_root = root / "WeiBo21"
    train_df = load_pickle_df(data_root / "train.pkl")
    val_df = load_pickle_df(data_root / "val.pkl")
    test_df = load_pickle_df(data_root / "test.pkl")
    train_df = train_df[train_df["category"] != "无法确定"].copy()
    val_df = val_df[val_df["category"] != "无法确定"].copy()
    test_df = test_df[test_df["category"] != "无法确定"].copy()
    return train_df, val_df, test_df, "content"


def load_finefake(root: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    df = load_pickle_df(root / "FineFake" / "extracted" / "FineFake.pkl")
    df = df[df["topic"] != "Uncategorized"].copy()
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return train_df, val_df, test_df, "text"


def print_metrics(name: str, y_true: pd.Series, y_pred: list[int]) -> None:
    print(f"\n## {name}")
    print(f"accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"f1:        {f1_score(y_true, y_pred, zero_division=0):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["weibo21", "finefake"], default="weibo21")
    parser.add_argument("--root", default="datasets")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    root = Path(args.root)
    if args.dataset == "weibo21":
        train_df, val_df, test_df, text_col = load_weibo21(root)
        analyzer = "char"
        ngram_range = (2, 4)
    else:
        train_df, val_df, test_df, text_col = load_finefake(root, args.seed)
        analyzer = "word"
        ngram_range = (1, 2)

    print(f"dataset: {args.dataset}")
    print(f"train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=analyzer,
                    ngram_range=ngram_range,
                    max_features=100000,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    model.fit(train_df[text_col].astype(str), train_df["label"].astype(int))
    val_pred = model.predict(val_df[text_col].astype(str))
    test_pred = model.predict(test_df[text_col].astype(str))
    print_metrics("validation", val_df["label"].astype(int), val_pred)
    print_metrics("test", test_df["label"].astype(int), test_pred)


if __name__ == "__main__":
    main()
