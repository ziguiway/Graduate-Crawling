from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

import torch

from LLM_MFEFND import MultiDomainFENDModel
from utils.multimodal_dataloader import build_finefake_aux_dataloader


def main() -> None:
    torch.manual_seed(2026)
    loader = build_finefake_aux_dataloader(batch_size=2, num_workers=0, shuffle=True)
    batch = next(iter(loader))
    model = MultiDomainFENDModel(
        emb_dim=768,
        mlp_dims=[384],
        domain_num=6,
        dropout=0.2,
        dataset="en",
    ).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    before = next(parameter.detach().clone() for name, parameter in model.named_parameters() if name.startswith("hpt."))
    output = model(**batch)
    loss = torch.nn.functional.binary_cross_entropy(
        output["classify_pred"], batch["label"].float()
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = next(parameter.detach() for name, parameter in model.named_parameters() if name.startswith("hpt."))

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for name, parameter in model.named_parameters() if name.startswith("hpt."))
    assert not torch.equal(before, after)
    print("train_step_ok")
    print(f"loss={loss.item():.6f}")
    print("hpt_gradient_ok")
    print("optimizer_update_ok")


if __name__ == "__main__":
    main()
