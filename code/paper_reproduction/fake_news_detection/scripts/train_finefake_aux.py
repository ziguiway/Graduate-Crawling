from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, random_split


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

from LLM_MFEFND import MultiDomainFENDModel
from utils.multimodal_dataloader import FineFakeAuxMultimodalDataset


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            labels.extend(batch["label"].cpu().tolist())
            scores.extend(output["classify_pred"].cpu().tolist())
    predictions = [int(score >= 0.5) for score in scores]
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "auc": auc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LLM-MFEFND on the available FineFake auxiliary subset")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=2, help="0 means all batches")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FineFakeAuxMultimodalDataset(
        aux_csv=ROOT / "LLM-MFEFND/data/GPT-DS-GLM-Weibo21-FineFake.csv",
        finefake_root=ROOT / "datasets/FineFake/extracted",
        tokenizer_name="bert-base-uncased",
        max_len=170,
    )
    val_size = max(1, round(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultiDomainFENDModel(
        emb_dim=768,
        mlp_dims=[384],
        domain_num=6,
        dropout=0.2,
        dataset="en",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
    loss_fn = torch.nn.BCELoss()
    global_step = 0

    print(f"dataset_size={len(dataset)} train={train_size} val={val_size} device={device}")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            loss = loss_fn(output["classify_pred"], batch["label"].float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                break
        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch + 1} steps={global_step} "
            f"loss={sum(losses) / len(losses):.6f} "
            f"val_accuracy={metrics['accuracy']:.4f} "
            f"val_f1_macro={metrics['f1_macro']:.4f} val_auc={metrics['auc']:.4f}"
        )
        if args.max_steps and global_step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
