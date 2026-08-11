from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

from mlime import explain, masked_image_patches, masked_token_inputs


def main() -> None:
    token_ids = torch.arange(1, 9)

    def token_predictor(masks: torch.Tensor) -> torch.Tensor:
        masked = masked_token_inputs(token_ids, masks)
        return torch.sigmoid((masked.float() * torch.tensor([3., 2., 1., 0., 0., 0., 0., 0.])).sum(1) / 5 - 2)

    token_result = explain(token_predictor, feature_count=8, num_samples=128)
    assert token_result.weights[0] > token_result.weights[7]

    image = torch.ones(3, 28, 28)

    def image_predictor(masks: torch.Tensor) -> torch.Tensor:
        masked = masked_image_patches(image, masks, grid_size=(2, 2))
        return masked[:, :, :14, :14].mean(dim=(1, 2, 3))

    image_result = explain(image_predictor, feature_count=4, num_samples=128)
    assert image_result.weights[0] > image_result.weights[3]
    print("mlime_ok", {"token_top": token_result.top_features(3), "image_top": image_result.top_features(2), "r2": round(token_result.r2, 4)})


if __name__ == "__main__":
    main()
