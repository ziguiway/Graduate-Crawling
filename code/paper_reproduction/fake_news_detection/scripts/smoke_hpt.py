from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

from pivot import HierarchicalProgressiveTransformer


def main() -> None:
    torch.manual_seed(2026)
    model = HierarchicalProgressiveTransformer(
        feature_dim=32,
        num_tokens=5,
        num_heads=4,
        num_rounds=4,
        dropout=0.0,
    )
    features = [torch.randn(3, 32) for _ in range(5)]
    output, trace = model(*features, return_trace=True)

    assert output.shape == (3, 32)
    assert trace["order"] == ["text", "image", "aligned", "background", "comments"]
    assert len(trace["rounds"]) == 4
    assert all(len(round_trace) == 5 for round_trace in trace["rounds"])
    assert all(shape == (3, 5, 32) for shape in trace["shared_shapes"])

    loss = output.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    print("hpt_forward_ok")
    print(f"output_shape={tuple(output.shape)}")
    print(f"fusion_order={'>'.join(trace['order'])}")
    print(f"rounds={len(trace['rounds'])}; shared_shape={trace['shared_shapes'][-1]}")
    print("backward_ok")


if __name__ == "__main__":
    main()
