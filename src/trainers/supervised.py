from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader

from utils.training_utils import (
    BaseTrainer,
    choose_device,
    count_samples,
    save_checkpoint,
    register_trainer,
    create_backbone,
)


def build_simclr_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    """
    Crea un DataLoader per SimCLR
    """
    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=1000,
            empty_check=False,
        )
        .decode("pil")
        .map(lambda sample: SimCLRPreprocSample(patch_size, aug_cfg)(sample))
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


class SimCLRPreprocSample:
    """
    Genera due viste augmentate per SimCLR
    """
    def __init__(self, patch_size: int, aug_cfg: Dict[str, Any]):
        self.patch_size = patch_size
        self.aug_cfg = aug_cfg
        self.transform1 = self._build_transform()
        self.transform2 = self._build_transform()

    def _build_transform(self) -> T.Compose:
        transforms: list[Any] = [
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

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next(
            (v for v in sample.values() if hasattr(v, 'convert')), None
        )
        if not hasattr(img, 'convert'):
            raise RuntimeError("No image found in sample for SimCLR")
        return self.transform1(img), self.transform2(img)


class NTXentLoss(nn.Module):
    """
    NT-Xent loss
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = nn.functional.normalize(z, dim=1)
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
    Trainer per SimCLR, senza stampe interne.
    """
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any]
    ):
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

        # Device
        self.device = choose_device()

        # Dataset
        self.train_pattern = str(Path(data_cfg["train"]))
        self.num_train = count_samples(Path(self.train_pattern))
        self.batches_train = math.ceil(self.num_train / self.batch_size)

        # DataLoader
        self.train_loader = build_simclr_loader(
            shards_pattern=self.train_pattern,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )

        # Model e proiezione
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

        # Ottimizzatore e loss
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = NTXentLoss(self.temperature)

        # Tracking
        self.best_epoch = 0

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        """
        Esegue un passo di training per un batch SimCLR.
        Returns: (batch_loss, batch_size)
        """
        x1, x2 = batch
        x1, x2 = x1.to(self.device), x2.to(self.device)
        self.encoder.train()
        self.projection_head.train()
        self.optimizer.zero_grad()
        h1, h2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projection_head(h1), self.projection_head(h2)
        loss = self.criterion(z1, z2)
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), x1.size(0)

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        """
        Salva checkpoint se epoch_loss migliora.
        """
        if epoch_loss < getattr(self, 'best_loss', float('inf')):
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            save_checkpoint(
                ckpt_dir=Path(self.train_pattern).parent / "checkpoints",
                prefix=self.__class__.__name__, epoch=epoch, best=True,
                model=torch.nn.Sequential(self.encoder, self.projection_head),
                optimizer=self.optimizer,
                class_to_idx={}, model_cfg=self.model_cfg, data_cfg=self.data_cfg,
            )

    def summary(self) -> Tuple[int, float]:
        """
        Restituisce (best_epoch, best_loss).
        """
        return getattr(self, 'best_epoch', 0), getattr(self, 'best_loss', 0.0)
