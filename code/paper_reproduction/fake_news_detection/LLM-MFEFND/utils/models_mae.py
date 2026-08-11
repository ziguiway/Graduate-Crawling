from __future__ import annotations

import os
from pathlib import Path

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
    use_real_backbones = os.environ.get("LLM_MFEFND_REAL_BACKBONES", "0") == "1"
    if use_real_backbones:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise RuntimeError("Real MAE path requires timm; install project dependencies first") from exc

        model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        checkpoint_path = Path(
            os.environ.get(
                "LLM_MFEFND_MAE_CHECKPOINT",
                "pretrained/mae_pretrain_vit_base.pth",
            )
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Real MAE checkpoint not found: {checkpoint_path}. "
                "Set LLM_MFEFND_MAE_CHECKPOINT or use the development fallback."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Unexpected MAE checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        return TimmMAE(model)
    return LiteMAE()


class TimmMAE(nn.Module):
    """MAE encoder wrapper exposing the official repo's ``forward_ying`` API."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward_ying(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)
