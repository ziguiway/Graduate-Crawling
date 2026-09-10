from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

from pivot import ConcatenationFusion, HierarchicalProgressiveTransformer, OfficialHierarchicalProgressiveTransformer


def main() -> None:
    torch.manual_seed(2026)
    inputs = [torch.randn(2, 768) for _ in range(5)]
    equation_hpt = HierarchicalProgressiveTransformer(dropout=0.0).eval()
    official_hpt = OfficialHierarchicalProgressiveTransformer(dropout=0.0).eval()
    concat = ConcatenationFusion(dropout=0.0).eval()
    names = ("text", "image", "aligned", "background", "comments")
    kwargs = dict(zip(names, inputs))
    equation_output, equation_trace = equation_hpt(**kwargs, return_trace=True)
    official_output, official_trace = official_hpt(**kwargs, return_trace=True)
    concat_output = concat(**kwargs)
    assert equation_output.shape == official_output.shape == concat_output.shape == (2, 768)
    assert len(equation_trace["rounds"]) == len(official_trace["rounds"]) == 4
    assert equation_trace["order"] == official_trace["order"]
    print("hpt_compare_ok")
    print({"equation_parameters": sum(p.numel() for p in equation_hpt.parameters()), "official_parameters": sum(p.numel() for p in official_hpt.parameters()), "concat_parameters": sum(p.numel() for p in concat.parameters()), "equation_official_mean_abs_gap": float((equation_output - official_output).abs().mean().detach())})
    print({"equation_order": equation_trace["order"], "official_order": official_trace["order"]})


if __name__ == "__main__":
    main()
