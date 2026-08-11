from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from LLM_MFEFND import Trainer
from utils.multimodal_dataloader import FineFakeAuxMultimodalDataset


class Run:
    """Small public-data training scaffold replacing the missing grid_search.py."""

    def __init__(self, config: dict):
        self.config = config
        self.project_root = Path(__file__).resolve().parents[1]

    def _resolve_path(self, value, default: Path) -> Path:
        if value is None:
            return default
        path = Path(value)
        if path.is_absolute():
            return path
        # Support both `python LLM-MFEFND/main.py` from the parent directory
        # and the upstream README's `cd LLM-MFEFND && python main.py`.
        candidates = (Path.cwd() / path, self.project_root / path, self.project_root.parent / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return Path.cwd() / path

    def _get_loaders(self):
        if self.config["dataset"] != "en":
            raise RuntimeError(
                "The public repository does not include the complete WeiBo21 "
                "multimodal images and LLM auxiliary data. Use --dataset en "
                "with the available FineFake auxiliary subset, or obtain the "
                "full data from the authors."
            )

        use_real_backbones = os.environ.get("LLM_MFEFND_REAL_BACKBONES", "0") == "1"
        dataset = FineFakeAuxMultimodalDataset(
            aux_csv=self._resolve_path(
                self.config.get("aux_csv"),
                self.project_root / "LLM-MFEFND/data/GPT-DS-GLM-Weibo21-FineFake.csv",
            ),
            finefake_root=self._resolve_path(
                self.config.get("finefake_root"),
                self.project_root / "datasets/FineFake/extracted",
            ),
            tokenizer_name=self.config.get("tokenizer_name", "bert-base-uncased"),
            max_len=self.config["max_len"],
            use_cn_clip=use_real_backbones,
        )
        val_size = max(1, round(len(dataset) * self.config.get("val_ratio", 0.2)))
        test_size = max(1, round(len(dataset) * self.config.get("test_ratio", 0.2)))
        train_size = len(dataset) - val_size - test_size
        train_set, val_set, test_set = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.config["seed"]),
        )
        loader_args = {
            "batch_size": self.config["batchsize"],
            "num_workers": self.config["num_workers"],
            "pin_memory": torch.cuda.is_available(),
        }
        return (
            DataLoader(train_set, shuffle=True, **loader_args),
            DataLoader(val_set, shuffle=False, **loader_args),
            DataLoader(test_set, shuffle=False, **loader_args),
        )

    def main(self):
        train_loader, val_loader, test_loader = self._get_loaders()
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("llm-mfefnd")
        trainer = Trainer(
            emb_dim=self.config["emb_dim"],
            mlp_dims=self.config["model"]["mlp"]["dims"],
            use_cuda=self.config["use_cuda"],
            lr=self.config["lr"],
            dropout=self.config["model"]["mlp"]["dropout"],
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            category_dict=self.config["category_dict"],
            weight_decay=self.config["weight_decay"],
            save_param_dir=self.config["save_param_dir"],
            dataset=self.config["dataset"],
            hpt_variant=self.config.get("hpt_variant", "official"),
            use_interactions=self.config.get("use_interactions", True),
            early_stop=self.config["early_stop"],
            epoches=self.config["epoch"],
        )
        return trainer.train(logger=logger)
