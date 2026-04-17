"""
dataset.py
==========
PyTorch Dataset for the 102-Flower Text-to-Image task.

Each sample exposes
-------------------
  image        : FloatTensor [3, H, W]  (normalised to [-1, 1])
  caption      : str                    (one of N captions for this image)
  wrong_caption: str                    (caption from a DIFFERENT class)
  class_id     : int                    (0-indexed flower class)
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ── Image transforms ───────────────────────────────────────────────────────
def get_transform(image_size: int, split: str = "train") -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])


class FlowerCaptionDataset(Dataset):
    """
    Parameters
    ----------
    data_root    : str  Path to flowers/{train|valid|test} directory.
    captions_json: str  Path to generated captions JSON.
    split        : str  'train', 'valid', or 'test'.
    image_size   : int  Square output size.
    captions_per_image : int  Number of captions per image.
    seed         : int  RNG seed.
    """

    def __init__(
        self,
        data_root: str,
        captions_json: str,
        split: str = "train",
        image_size: int = 64,
        captions_per_image: int = 5,
        seed: int = 42,
    ):
        self.data_root = Path(data_root) / split
        self.split     = split
        self.image_size = image_size
        self.transform  = get_transform(image_size, split)
        self.captions_per_image = captions_per_image
        random.seed(seed)

        # ── Load captions ──────────────────────────────────────────────────
        if not os.path.exists(captions_json):
            raise FileNotFoundError(
                f"Captions file not found: {captions_json}\n"
                f"Run: python data/generate_captions.py first."
            )
        with open(captions_json, "r") as f:
            all_captions: Dict = json.load(f)  # { class_id_str: [cap1, cap2, ...] }

        # ── Discover images ────────────────────────────────────────────────
        self.samples: List[Tuple[Path, int, List[str]]] = []
        # Maps class_folder_name → 0-indexed int
        class_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}
        self.idx_to_class = {i: d for d, i in self.class_to_idx.items()}

        for cls_dir in class_dirs:
            cls_id = self.class_to_idx[cls_dir.name]
            caps   = all_captions.get(cls_dir.name, [f"a flower of type {cls_dir.name}"])
            imgs   = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png"))
            for img_path in imgs:
                self.samples.append((img_path, cls_id, caps))

        # Pre-compute per-class caption pools for wrong-caption sampling
        self._class_caps: Dict[int, List[str]] = {}
        for _, cls_id, caps in self.samples:
            self._class_caps.setdefault(cls_id, []).extend(caps)
        # Deduplicate
        self._class_caps = {k: list(set(v)) for k, v in self._class_caps.items()}
        self._all_class_ids = list(self._class_caps.keys())

        print(f"[Dataset] {split}: {len(self.samples)} images, "
              f"{len(self.class_to_idx)} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, cls_id, caps = self.samples[idx]

        # ── Image ─────────────────────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            image = torch.zeros(3, self.image_size, self.image_size)

        # ── Caption (pick one at random) ───────────────────────────────────
        caption = random.choice(caps)

        # ── Wrong caption (from a different class) ─────────────────────────
        wrong_cls = cls_id
        while wrong_cls == cls_id:
            wrong_cls = random.choice(self._all_class_ids)
        wrong_caption = random.choice(self._class_caps[wrong_cls])

        return {
            "image":         image,
            "caption":       caption,
            "wrong_caption": wrong_caption,
            "class_id":      torch.tensor(cls_id, dtype=torch.long),
        }

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


def build_dataloader(
    data_root: str,
    captions_json: str,
    split: str,
    image_size: int,
    batch_size: int,
    num_workers: int = 4,
    captions_per_image: int = 5,
    seed: int = 42,
) -> DataLoader:
    dataset = FlowerCaptionDataset(
        data_root=data_root,
        captions_json=captions_json,
        split=split,
        image_size=image_size,
        captions_per_image=captions_per_image,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.config import CFG

    dl = build_dataloader(
        data_root=CFG.data.data_root,
        captions_json=CFG.data.captions_json,
        split="train",
        image_size=CFG.data.image_size,
        batch_size=4,
        num_workers=0,
    )
    batch = next(iter(dl))
    print("image shape :", batch["image"].shape)
    print("caption     :", batch["caption"][0])
    print("wrong cap   :", batch["wrong_caption"][0])
    print("class_id    :", batch["class_id"])
