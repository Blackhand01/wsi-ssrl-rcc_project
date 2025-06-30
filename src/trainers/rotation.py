from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from PIL import Image

from utils.training_utils import (
    BaseTrainer,
    choose_device,
    count_samples,
    save_checkpoint,
    register_trainer,
    create_backbone,
)

#######################################################################
# Data‑loading utilities
#######################################################################

class RotationPreprocSample:
    """Generate **one** randomly rotated view and its label.

    The image is rotated by one of {0°, 90°, 180°, 270°}.  The rotation index
    (0‑3) is returned as classification label.
    """

    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size
        self.base_transform = T.Compose([
            T.RandomResizedCrop(patch_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        self.to_tensor = T.ToTensor()

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, int]:
        img = sample.get("jpg") or next((v for v in sample.values() if isinstance(v, Image.Image)), None)
        if not isinstance(img, Image.Image):
            raise RuntimeError("No image found in Rotation sample")

        img = img.convert("RGB")
        img = self.base_transform(img)

        rot_idx = random.randint(0, 3)
        img = img.rotate(90 * rot_idx, expand=False)

        return self.to_tensor(img), rot_idx


def build_rotation_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    """Loader returning **(tensor, label)** tuples for training."""

    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=1000,
            empty_check=False,
        )
        .decode("pil")
        .map(RotationPreprocSample(patch_size))
    )
    use_cuda = device.type == "cuda"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # shardshuffle already applied
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=True,
    )

#######################################################################
# Trainer
#######################################################################

@register_trainer("rotation")
class RotationTrainer(BaseTrainer):
    """Self‑supervised **RotNet** trainer (predict image rotation)."""

    N_CLASSES = 4  # 0°, 90°, 180°, 270°

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)

        # --- Hyper‑parameters --------------------------------------------------
        t = model_cfg["training"]
        self.epochs        = int(t["epochs"])
        self.batch_size    = int(t["batch_size"])
        self.lr            = float(t["learning_rate"])
        self.weight_decay  = float(t["weight_decay"])
        self.optimizer_name= t.get("optimizer", "adam").lower()
        self.patch_size    = int(model_cfg.get("patch_size", 224))

        # --- Device & dataset --------------------------------------------------
        self.device        = choose_device()
        self.train_pattern = str(Path(data_cfg["train"]))
        self.num_train     = count_samples(Path(self.train_pattern))
        self.batches_train = math.ceil(self.num_train / self.batch_size)

        self.train_loader  = build_rotation_loader(
            shards_pattern=self.train_pattern,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
        )

        # --- Model -------------------------------------------------------------
        backbone_name = model_cfg.get("backbone", "resnet18").lower()
        self.model = create_backbone(backbone_name, num_classes=self.N_CLASSES, pretrained=False).to(self.device)

        # --- Optimizer & loss --------------------------------------------------
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        self.criterion = nn.CrossEntropyLoss()

        # --- Tracking ----------------------------------------------------------
        self.best_epoch = 0
        self.best_loss  = float("inf")

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        imgs, labels = batch
        imgs, labels = imgs.to(self.device), labels.to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(imgs)
        loss   = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()

        return float(loss.item()), imgs.size(0)

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        if epoch_loss < self.best_loss:
            self.best_loss  = epoch_loss
            self.best_epoch = epoch
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=self.model,
                optimizer=self.optimizer,
                class_to_idx={},
                model_cfg=self.model_cfg,
                data_cfg=self.data_cfg,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> Tuple[int, float]:
        """Return `(best_epoch, best_loss)`."""
        return self.best_epoch, self.best_loss

    # ------------------------------------------------------------------
    # Feature extraction utility
    # ------------------------------------------------------------------

    def extract_features_to(self, output_path: str) -> None:
        """Extract backbone features for *all* training patches and save to *output_path*."""
        from .extract_features import extract_features  # lazy import to avoid circular deps

        def _make_inference_loader() -> DataLoader:
            ds = (
                wds.WebDataset(self.train_pattern)
                .decode("pil")
                .map(lambda sample: {
                    "img": T.ToTensor()(next((v for k, v in sample.items() if isinstance(v, Image.Image)), None).convert("RGB")),
                    "key": sample["__key__"] + "." + next((k for k in sample.keys() if k.endswith(".jpg")), "")
                })
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        dataloader = _make_inference_loader()
        feats = extract_features(self.model, dataloader, self.device)
        torch.save(feats, output_path)
        print(f"✅ Rotation features saved to {output_path}")
