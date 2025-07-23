from __future__ import annotations
import io
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
from .extract_features import extract_features


#######################################################################
# Dataset & DataLoader for Rotation pretext
#######################################################################

class RotationPreprocessor:
    """
    Turn a WebDataset sample into (rotated_tensor, rotation_label).
    Rotation label ∈ {0,1,2,3} → {0°, 90°,180°,270°}.
    """
    def __init__(self, patch_size: int, seed: int | None = None) -> None:
        self.patch_size = patch_size
        self.base = T.Compose([
            T.RandomResizedCrop(patch_size),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        ])
        self.to_tensor = T.ToTensor()
        if seed is not None:
            random.seed(seed)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, int]:
        # WebDataset.decode("pil") gives sample["jpg"] = PIL.Image
        img = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)), None
        )
        if img is None:
            raise RuntimeError("No image found in WebDataset sample")
        img = img.convert("RGB")
        img = self.base(img)
        rot = random.randint(0, 3)
        img = img.rotate(90 * rot, expand=False)
        return self.to_tensor(img), rot


def build_rotation_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    seed: int | None = None,
) -> DataLoader:
    """
    DataLoader yielding (img_tensor, rot_label) batches for training.
    Automatically handles both single-*.tar or flat directory of .tar shards.
    """
    # if they passed a folder, look for all .tar inside
    p = Path(shards_pattern)
    pattern = str(p / "*.tar") if p.is_dir() else shards_pattern

    ds = (
        wds.WebDataset(
            pattern,
            handler=wds.warn_and_continue,
            shardshuffle=False,   # inferenza → niente shuffle
            empty_check=False     # evita ValueError se qualche worker resta senza shard
        )
        .decode("pil")
        .map(RotationPreprocessor(patch_size, seed))
    )
    use_cuda = (device.type == "cuda")
    n_shards = 1 if Path(pattern).is_file() else len(list(Path().glob(pattern)))
    num_workers = min(4 if use_cuda else 0, n_shards)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )


#######################################################################
# Rotation Trainer
#######################################################################

@register_trainer("rotation")
class RotationTrainer(BaseTrainer):
    """Self-supervised rotation prediction (no downstream classifier here)."""

    N_CLASSES = 4

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)
        self.device = choose_device()
        self._init_params(model_cfg)
        self._init_model_and_optimizer(model_cfg)
        self._init_tracking()

    def _init_params(self, cfg: Dict[str, Any]) -> None:
        t = cfg["training"]
        self.epochs       = int(t["epochs"])
        self.batch_size   = int(t["batch_size"])
        self.lr           = float(t["learning_rate"])
        self.weight_decay = float(t["weight_decay"])
        self.optimizer_name = t.get("optimizer", "adam").lower()
        self.patch_size   = int(cfg.get("patch_size", 224))
        self.seed         = cfg.get("seed", None)

    def build_loader(self, split: str) -> DataLoader:
        shards = self.data_cfg[split]
        loader = build_rotation_loader(
            shards_pattern=shards,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            seed=self.seed,
        )
        try:
            self.batches_train = len(loader)
        except Exception:
            n = count_samples(Path(shards))
            self.batches_train = math.ceil(n / self.batch_size)
        return loader

    def _init_model_and_optimizer(self, cfg: Dict[str, Any]) -> None:
        backbone_name = cfg.get("backbone", "resnet18").lower()
        self.model = create_backbone(
            backbone_name, num_classes=self.N_CLASSES, pretrained=False
        ).to(self.device)
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        self.criterion = nn.CrossEntropyLoss()

    def _init_tracking(self) -> None:
        self.best_loss = float("inf")
        self.best_epoch = 0

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        imgs, labels = batch
        imgs = imgs.to(self.device)
        labels = labels.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(imgs)
        loss = self.criterion(logits, labels)
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), imgs.size(0)

    def validate_epoch(self) -> Tuple[float, float]:
        raise NotImplementedError("RotationTrainer does not support validation")

    def post_epoch(self, epoch: int, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss, self.best_epoch = loss, epoch
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=self.model,
                optimizer=self.optimizer,
                metadata={
                    "model_cfg": self.model_cfg,
                    "data_cfg": self.data_cfg,
                },
            )

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    def get_resume_model_and_optimizer(self) -> Tuple[Any, Any]:
        return self.model, self.optimizer

    def extract_features_to(
        self,
        output_path: Path | str,
        split: str = "train",
    ) -> None:
        """
        Extract patch embeddings and keys, saving {"features":Tensor[N,D], "keys":List[str]}.
        Emits (img_tensor, key) via manual decode to avoid mis-decoding.
        """
        if split not in self.data_cfg:
            raise ValueError(f"Unknown split '{split}' – available: {list(self.data_cfg)}")
        shards = self.data_cfg[split]
        p = Path(shards)
        pattern = str(p / "*.tar") if p.is_dir() else shards

        # decode bytes → PIL → Tensor, keep key
        ds = (
            wds.WebDataset(
                pattern,
                handler=wds.warn_and_continue,
                shardshuffle=False,   # inferenza → niente shuffle
                empty_check=False     # evita ValueError se qualche worker resta senza shard
            )
            .to_tuple("jpg", "__key__")
            .map_tuple(
                lambda b: T.ToTensor()(Image.open(io.BytesIO(b)).convert("RGB")),
                lambda k: k,
            )
        )

        use_cuda = (self.device.type == "cuda")
        # worker ≤ shard → niente processi sprecati
        n_shards = 1 if Path(pattern).is_file() else len(list(Path().glob(pattern)))
        num_workers = min(4 if use_cuda else 0, n_shards)

        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        self.model.eval()
        with torch.no_grad():
            feats_dict = extract_features(self.model, loader, self.device)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats_dict, output_path)
        print(f"✅ Rotation features ({split}) saved → {output_path}")
