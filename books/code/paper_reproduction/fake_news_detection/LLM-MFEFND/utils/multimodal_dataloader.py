from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer

try:
    from cn_clip.clip import image_transform as cn_clip_image_transform
    from cn_clip.clip import tokenize as cn_clip_tokenize
except Exception:  # pragma: no cover - optional real-backbone dependency
    cn_clip_image_transform = None
    cn_clip_tokenize = None


def _read_aux_finefake(aux_csv: Path, finefake_root: Path) -> pd.DataFrame:
    df = pd.read_csv(aux_csv)
    required = {
        "text",
        "image_path",
        "label",
        "llm_background",
        "llm_comment_join",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {aux_csv}: {sorted(missing)}")

    df = df.copy()
    df["image_abs_path"] = df["image_path"].astype(str).map(lambda path: finefake_root / path)
    df = df[df["image_abs_path"].map(lambda path: path.exists())].reset_index(drop=True)
    return df


class FineFakeAuxMultimodalDataset(Dataset):
    """FineFake auxiliary subset formatted for LLM-MFEFND.

    This uses the 200-row public auxiliary CSV released with LLM-MFEFND. The
    `clip_content_features` tensor is a placeholder token sequence produced by
    the same BERT tokenizer; the full paper path still needs the CN-CLIP text
    tokenizer.
    """

    def __init__(
        self,
        aux_csv: Path,
        finefake_root: Path,
        tokenizer_name: str = "bert-base-uncased",
        max_len: int = 170,
        clip_max_len: int = 77,
        use_cn_clip: bool = False,
    ):
        self.data = _read_aux_finefake(aux_csv, finefake_root)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len
        self.use_cn_clip = use_cn_clip and cn_clip_tokenize is not None
        self.clip_max_len = 52 if self.use_cn_clip else clip_max_len
        if self.use_cn_clip:
            self.image_transform = cn_clip_image_transform(224)
        else:
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.data)

    def _tokenize(self, text: str, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def _tokenize_clip(self, text: str) -> torch.Tensor:
        if self.use_cn_clip:
            return cn_clip_tokenize(text, context_length=self.clip_max_len).squeeze(0)
        return self._tokenize(text, self.clip_max_len)[0]

    def _load_image(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.image_transform(image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.data.iloc[index]
        content, content_mask = self._tokenize(str(row["text"]), self.max_len)
        background, background_mask = self._tokenize(str(row["llm_background"]), self.max_len)
        comment, comment_mask = self._tokenize(str(row["llm_comment_join"]), self.max_len)
        clip_content = self._tokenize_clip(str(row["text"]))
        image = self._load_image(Path(row["image_abs_path"]))
        return {
            "content": content,
            "content_masks": content_mask,
            "background": background,
            "background_masks": background_mask,
            "comment": comment,
            "comment_masks": comment_mask,
            "image_features": image.unsqueeze(0),
            "clip_content_features": clip_content,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def build_finefake_aux_dataloader(
    aux_csv: str | Path = "LLM-MFEFND/data/GPT-DS-GLM-Weibo21-FineFake.csv",
    finefake_root: str | Path = "datasets/FineFake/extracted",
    tokenizer_name: str = "bert-base-uncased",
    batch_size: int = 4,
    max_len: int = 170,
    use_cn_clip: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = FineFakeAuxMultimodalDataset(
        aux_csv=Path(aux_csv),
        finefake_root=Path(finefake_root),
        tokenizer_name=tokenizer_name,
        max_len=max_len,
        use_cn_clip=use_cn_clip,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
