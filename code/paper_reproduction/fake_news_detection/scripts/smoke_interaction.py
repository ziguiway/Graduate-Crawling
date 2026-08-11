from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

from pivot import BidirectionalCrossAttention


def main() -> None:
    torch.manual_seed(2026)
    module = BidirectionalCrossAttention(hidden_size=32, num_heads=4)
    news = torch.randn(3, 7, 32)
    auxiliary = torch.randn(3, 5, 32)
    news_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]] * 3)
    auxiliary_mask = torch.tensor([[1, 1, 1, 0, 0]] * 3)

    interaction, weight, details = module(
        news,
        auxiliary,
        news_mask=news_mask,
        auxiliary_mask=auxiliary_mask,
    )
    assert interaction.shape == (3, 32)
    assert weight.shape == (3, 1)
    assert torch.all((weight >= 0) & (weight <= 1))
    assert details["forward_feature"].shape == (3, 32)
    assert details["reverse_feature"].shape == (3, 32)

    interaction.square().mean().backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    print("interaction_forward_ok")
    print(f"interaction_shape={tuple(interaction.shape)}")
    print(f"weight_range=({weight.min().item():.4f}, {weight.max().item():.4f})")
    print("interaction_backward_ok")


if __name__ == "__main__":
    main()
