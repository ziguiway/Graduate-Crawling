from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

import torch

from utils.multimodal_dataloader import build_finefake_aux_dataloader
from LLM_MFEFND import MultiDomainFENDModel


def main() -> None:
    loader = build_finefake_aux_dataloader(batch_size=2, num_workers=0)
    batch = next(iter(loader))
    model = MultiDomainFENDModel(
        emb_dim=768,
        mlp_dims=[384],
        domain_num=6,
        dropout=0.2,
        dataset="en",
    ).eval()

    batch = {key: value for key, value in batch.items()}
    with torch.no_grad():
        output = model(**batch)

    print("forward_ok")
    print({key: tuple(value.shape) for key, value in batch.items()})
    print({key: tuple(value.shape) if hasattr(value, "shape") else type(value) for key, value in output.items()})


if __name__ == "__main__":
    main()
