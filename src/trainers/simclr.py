'''  
src/trainers/simclr.py
------------------------
SimCLR trainer for self-supervised contrastive learning on RCC patches.

Implements the SimCLR training loop with NT-Xent loss, checkpointing, and logging,
using shared utilities from training_utils.

Author: Stefano Roy Bisignano – 2025-05
'''

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.training_utils import (
    BaseTrainer,
    choose_device,
    count_samples,
    save_checkpoint,
    register_trainer,
    create_backbone,
)

LOGGER = logging.getLogger(__name__)


class SimCLRPreprocSample:
    """
    Callable to generate two augmented views for SimCLR from a WebDataset sample.
    """
    def __init__(self,
                 patch_size: int,
                 aug_cfg: Dict[str, Any]):
        self.patch_size = patch_size
        self.aug_cfg = aug_cfg
        self.transform1 = self._build_transform()
        self.transform2 = self._build_transform()

    def _build_transform(self) -> T.Compose:
        transforms: List[Any] = [
            T.RandomResizedCrop(self.patch_size),
            T.RandomHorizontalFlip(),
        ]
        cj = self.aug_cfg.get("color_jitter", {})
        if cj:
            transforms.append(T.ColorJitter(**cj))
        if self.aug_cfg.get("gaussian_blur", False):
            transforms.append(T.GaussianBlur(kernel_size=3))
        if self.aug_cfg.get("grayscale", False):
            transforms.append(T.RandomGrayscale(p=0.2))
        transforms.append(T.ToTensor())
        return T.Compose(transforms)

    def __call__(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next(
            (v for v in sample.values() if hasattr(v, 'convert')), None)
        if not hasattr(img, 'convert'):
            raise RuntimeError("No image found in sample for SimCLR")
        return self.transform1(img), self.transform2(img)


def build_simclr_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=1000,
            empty_check=False,
        )
        .decode("pil")
        .map(SimCLRPreprocSample(patch_size, aug_cfg))
    )
    use_cuda = (device.type == "cuda")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=True,
    )


class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive self-supervised learning.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        mask = (~torch.eye(2 * batch_size, device=z.device).bool()).float()
        exp_sim = torch.exp(sim) * mask
        pos = torch.exp((z1 * z2).sum(dim=1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        loss = -torch.log(pos / exp_sim.sum(dim=1)).mean()
        return loss


@register_trainer("simclr")
class SimCLRTrainer(BaseTrainer):
    """
    Trainer for SimCLR self-supervised learning on RCC patch dataset.
    """
    def __init__(self,
                 model_cfg: Dict[str, Any],
                 data_cfg: Dict[str, Any]):
        super().__init__(model_cfg, data_cfg)
        tcfg = model_cfg["training"]
        self.epochs = int(tcfg["epochs"])
        self.batch_size = int(tcfg["batch_size"])
        self.lr = float(tcfg["learning_rate"])
        self.weight_decay = float(tcfg["weight_decay"])
        self.optimizer_name = tcfg.get("optimizer", "adam").lower()
        self.temperature = float(tcfg.get("temperature", 0.5))
        self.patch_size = int(model_cfg.get("patch_size", 224))
        self.aug_cfg = model_cfg.get("augmentation", {})

        # Device selection
        self.device = choose_device()
        LOGGER.info("Selected device: %s", self.device)

        # Dataset stats
        self.train_pattern = str(Path(data_cfg["train"]))
        self.num_train = count_samples(Path(self.train_pattern))
        self.batches_train = math.ceil(self.num_train / self.batch_size)
        LOGGER.info(
            "Dataset sizes → train: %d (%d batches)",
            self.num_train, self.batches_train,
        )

        # DataLoader
        self.train_loader = build_simclr_loader(
            shards_pattern=self.train_pattern,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )

        # Model: encoder + projection head
        backbone_name = model_cfg.get("backbone", "resnet18").lower()
        base = create_backbone(backbone_name, num_classes=0, pretrained=False)
        num_ftrs = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base.to(self.device)
        self.projection_head = nn.Sequential(
            nn.Linear(num_ftrs, model_cfg.get("proj_dim", 128)),
            nn.ReLU(),
            nn.Linear(model_cfg.get("proj_dim", 128), model_cfg.get("proj_dim", 128)),
        ).to(self.device)

        # Optimizer & Loss
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = NTXentLoss(self.temperature)
        self.best_epoch = 0

    def train(self) -> None:
        """
        Runs SimCLR training loop with NT-Xent loss and checkpointing.
        """
        LOGGER.info(
            "Starting training: epochs=%d | bs=%d | lr=%.2e | wd=%.2e | opt=%s",
            self.epochs, self.batch_size, self.lr,
            self.weight_decay, self.optimizer_name,
        )
        best_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            # Progress bar per epoch
            bar = tqdm(
                self.train_loader,
                desc=f"Train E{epoch}",
                total=self.batches_train,
                unit="batch",
                leave=True,
                dynamic_ncols=True,
            )
            running_loss = 0.0
            processed = 0
            for x1, x2 in bar:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                self.optimizer.zero_grad()
                h1, h2 = self.encoder(x1), self.encoder(x2)
                z1, z2 = self.projection_head(h1), self.projection_head(h2)
                loss = self.criterion(z1, z2)
                loss.backward()
                self.optimizer.step()

                batch_size = x1.size(0)
                running_loss += loss.item() * batch_size
                processed += batch_size
                bar.set_postfix(
                    loss=f"{running_loss/processed:.4f}"
                )

            epoch_loss = running_loss / (self.batches_train * self.batch_size)
            LOGGER.info("Epoch %d completed: loss=%.4f", epoch, epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.best_epoch = epoch
                save_checkpoint(
                    ckpt_dir=Path(self.train_pattern).parent / "checkpoints",
                    prefix=self.__class__.__name__,
                    epoch=epoch,
                    best=True,
                    model=torch.nn.Sequential(self.encoder, self.projection_head),
                    optimizer=self.optimizer,
                    class_to_idx={}, model_cfg=self.model_cfg, data_cfg=self.data_cfg,
                )

        LOGGER.info(
            "Training complete. Best loss=%.4f at epoch %d",
            best_loss, self.best_epoch,
        )
