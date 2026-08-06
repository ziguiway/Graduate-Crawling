from __future__ import annotations

import torch
import torch.nn as nn


class LiteMAE(nn.Module):
    """A lightweight stand-in for the paper's MAE encoder.

    It keeps the `forward_ying` interface so we can exercise the rest of the
    pipeline before wiring in the real pretrained MAE checkpoint.
    """

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

    def load_state_dict(self, state_dict, strict=True):  # noqa: D401
        return nn.modules.module._IncompatibleKeys([], list(state_dict.keys()))

    def forward_ying(self, x: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, patch_tokens], dim=1)
        return self.norm(tokens)


def mae_vit_base_patch16(*args, **kwargs):
    return LiteMAE()
