"""Run MLIME through the complete reconstructed LLM-MFEFND predictor.

The default run is a data-flow check with a randomly initialized classifier.
Use --checkpoint for explanations from a trained model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LLM-MFEFND"))

from LLM_MFEFND import MultiDomainFENDModel
from mlime import explain
from utils.multimodal_dataloader import FineFakeAuxMultimodalDataset


MODALITIES = (
    ("content", "content_masks"),
    ("image_features", None),
    ("background", "background_masks"),
    ("comment", "comment_masks"),
    ("clip_content_features", None),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FineFakeAuxMultimodalDataset(
        aux_csv=ROOT / "LLM-MFEFND/data/GPT-DS-GLM-Weibo21-FineFake.csv",
        finefake_root=ROOT / "datasets/FineFake/extracted",
        tokenizer_name="bert-base-uncased",
        max_len=170,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    batch = {key: value.to(device) for key, value in batch.items()}
    model = MultiDomainFENDModel(768, [384], 6, 0.2, "en").to(device)
    loaded_checkpoint = args.checkpoint is not None
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
    model.eval()

    def predictor(masks: torch.Tensor) -> torch.Tensor:
        masks = masks.to(device)
        outputs = []
        with torch.no_grad():
            for start in range(0, masks.size(0), 4):
                chunk = masks[start : start + 4]
                # All five modalities are controlled by one mask row.
                candidate = {
                    key: value[:1].repeat((chunk.size(0),) + (1,) * (value.ndim - 1))
                    for key, value in batch.items()
                }
                for index, (input_key, mask_key) in enumerate(MODALITIES):
                    present = chunk[:, index].bool()
                    if input_key == "image_features":
                        candidate[input_key] *= present.view(-1, 1, 1, 1, 1)
                    else:
                        if mask_key is not None:
                            absent = ~present
                            candidate[input_key][absent] = 0
                            candidate[mask_key][absent] = 0
                            # Keep one valid position so attention never receives
                            # an all-padding sequence after a modality is removed.
                            candidate[input_key][absent, 0] = batch[input_key][0, 0]
                            candidate[mask_key][absent, 0] = 1
                        else:
                            candidate[input_key] *= present.view(-1, *([1] * (candidate[input_key].ndim - 1)))
                outputs.append(model(**candidate)["classify_pred"].cpu())
        return torch.cat(outputs)

    result = explain(predictor, feature_count=len(MODALITIES), num_samples=args.num_samples, seed=args.seed)
    print(
        "mlime_model_ok",
        {
            "checkpoint_loaded": loaded_checkpoint,
            "prediction": round(result.prediction, 6),
            "r2": round(result.r2, 6),
            "top_modalities": [
                (MODALITIES[index][0], round(weight, 6))
                for index, weight in result.top_features(len(MODALITIES))
            ],
        },
    )


if __name__ == "__main__":
    main()
