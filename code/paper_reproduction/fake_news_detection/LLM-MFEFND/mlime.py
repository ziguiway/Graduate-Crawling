"""Model-agnostic multimodal LIME utilities.

This implements the paper's post-hoc idea: perturb one modality, keep the
other modalities fixed, query the complete predictor, and fit a local ridge
surrogate.  The predictor is intentionally injected so the explainer can be
used with either the equation-aligned or official HPT implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class MLIMEResult:
    """Local explanation for one binary prediction."""

    prediction: float
    weights: torch.Tensor
    intercept: float
    r2: float

    def top_features(self, k: int = 10) -> list[tuple[int, float]]:
        k = min(max(k, 0), self.weights.numel())
        indices = torch.argsort(self.weights.abs(), descending=True)[:k]
        return [(int(i), float(self.weights[i])) for i in indices]


def _cosine_kernel(masks: torch.Tensor, kernel_width: float) -> torch.Tensor:
    reference = torch.ones((1, masks.size(1)), dtype=masks.dtype, device=masks.device)
    cosine = torch.nn.functional.cosine_similarity(masks, reference, dim=1)
    return torch.exp(-((1.0 - cosine) ** 2) / (kernel_width**2))


def explain(
    predictor: Callable[[torch.Tensor], torch.Tensor],
    feature_count: int,
    num_samples: int = 256,
    kernel_width: float = 0.25,
    ridge_alpha: float = 1.0,
    seed: int = 2026,
) -> MLIMEResult:
    """Fit a local weighted ridge model around the all-on instance.

    ``predictor`` receives a ``[num_samples, feature_count]`` binary mask and
    must return one probability per row. The caller maps masks to model inputs,
    so the same function supports text words, image patches, or auxiliary
    feature groups.
    """
    if feature_count < 1 or num_samples < 2:
        raise ValueError("feature_count must be >= 1 and num_samples must be >= 2")
    if kernel_width <= 0 or ridge_alpha < 0:
        raise ValueError("kernel_width must be positive and ridge_alpha non-negative")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    masks = torch.randint(0, 2, (num_samples - 1, feature_count), generator=generator).float()
    masks = torch.cat([torch.ones((1, feature_count)), masks], dim=0)
    predictions = predictor(masks).reshape(-1).float().cpu()
    if predictions.numel() != num_samples:
        raise ValueError("predictor must return one prediction per mask")

    weights = _cosine_kernel(masks, kernel_width).cpu()
    design = torch.cat([torch.ones((num_samples, 1)), masks], dim=1)
    sqrt_weights = weights.sqrt().unsqueeze(1)
    regularizer = torch.eye(feature_count + 1)
    regularizer[0, 0] = 0.0
    lhs = design.T @ (sqrt_weights**2 * design) + ridge_alpha * regularizer
    rhs = design.T @ (sqrt_weights[:, 0] ** 2 * predictions)
    coefficients = torch.linalg.solve(lhs, rhs)
    fitted = design @ coefficients
    centered = predictions - (weights @ predictions) / weights.sum().clamp_min(1e-8)
    residual = ((weights * (predictions - fitted)) ** 2).sum()
    total = (weights * centered**2).sum().clamp_min(1e-8)
    r2 = float(1.0 - residual / total)
    return MLIMEResult(
        prediction=float(predictions[0]),
        weights=coefficients[1:],
        intercept=float(coefficients[0]),
        r2=r2,
    )


def masked_token_inputs(
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Expand one tokenized example into masked copies for MLIME."""
    if input_ids.ndim != 1 or masks.ndim != 2 or masks.size(1) != input_ids.numel():
        raise ValueError("input_ids must be [tokens] and masks must be [samples, tokens]")
    return torch.where(masks.bool(), input_ids.unsqueeze(0), torch.tensor(pad_token_id, device=input_ids.device))


def masked_image_patches(
    image: torch.Tensor,
    masks: torch.Tensor,
    grid_size: tuple[int, int] = (14, 14),
) -> torch.Tensor:
    """Expand one image into copies with rectangular patches zeroed out."""
    if image.ndim != 3 or masks.ndim != 2:
        raise ValueError("image must be [channels, height, width] and masks [samples, patches]")
    rows, cols = grid_size
    if masks.size(1) != rows * cols:
        raise ValueError("mask count must equal grid_size product")
    _, height, width = image.shape
    output = image.unsqueeze(0).repeat(masks.size(0), 1, 1, 1).clone()
    for patch, active in enumerate(masks.T):
        row, col = divmod(patch, cols)
        y0, y1 = row * height // rows, (row + 1) * height // rows
        x0, x1 = col * width // cols, (col + 1) * width // cols
        output[~active.bool(), :, y0:y1, x0:x1] = 0
    return output
