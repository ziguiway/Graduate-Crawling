from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

from pivot import OfficialHierarchicalProgressiveTransformer


def main() -> None:
    torch.manual_seed(2026)
    model = OfficialHierarchicalProgressiveTransformer(
        feature_dim=32,
        num_tokens=5,
        star_tokens=4,
        num_heads=4,
        num_rounds=4,
        transformer_slots=18,
        dropout=0.0,
    )
    features = [torch.randn(2, 32) for _ in range(5)]
    output, trace = model(*features, return_trace=True)

    assert output.shape == (2, 32)
    assert trace["order"] == ["comments", "background", "aligned", "image", "text"]
    assert [stage["slot"] for stage in trace["rounds"][0]] == [0, 2, 3, 4, 5]
    assert [stage["slot"] for stage in trace["rounds"][-1]] == [9, 11, 12, 13, 14]

    output.square().mean().backward()
    used_slots = {stage["slot"] for round_trace in trace["rounds"] for stage in round_trace}
    for slot, layer in enumerate(model.transformers):
        for parameter in layer.parameters():
            if slot in used_slots:
                assert parameter.grad is not None
            else:
                assert parameter.grad is None
    print("official_hpt_forward_ok")
    print(f"output_shape={tuple(output.shape)}")
    print(f"fusion_order={'>'.join(trace['order'])}")
    print(f"used_transformer_slots={sorted(used_slots)}")
    print("official_hpt_backward_ok")


if __name__ == "__main__":
    main()
