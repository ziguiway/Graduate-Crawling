from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "LLM-MFEFND"))

    from utils.multimodal_dataloader import build_finefake_aux_dataloader

    loader = build_finefake_aux_dataloader(batch_size=2)
    batch = next(iter(loader))
    print("batch keys:")
    for key, value in batch.items():
        print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
