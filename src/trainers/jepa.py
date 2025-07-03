# trainers/jepa.py

from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class JEPADataPrep:
    """Preprocessing per JEPA: patch e contesto."""
    def _build_aug(self, size):
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=3),
            T.ToTensor(),
        ])

    def __init__(self, patch_size: int, context_size: int, aug_cfg: Dict[str, Any]) -> None:
        self.patch_size = patch_size
        self.context_size = context_size
        self.aug_cfg = aug_cfg
        self.local_transform = self._build_aug(self.patch_size)
        self.context_transform = self._build_aug(self.context_size)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next((v for v in sample.values() if hasattr(v, "convert")), None)
        if not hasattr(img, "convert"):
            raise RuntimeError("No image found in JEPA sample")
        img = img.convert("RGB")
        local = self.local_transform(img)
        context = self.context_transform(img)
        return local, context


def build_jepa_loader(
    shards_pattern: str,
    patch_size: int,
    context_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    """WebDataset loader per JEPA: patch locale + contesto."""
    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=1000,
            empty_check=False,
        )
        .decode("pil")
        .map(JEPADataPrep(patch_size, context_size, aug_cfg))
    )
    use_cuda = device.type == "cuda"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=True,
    )


class JEPAHead(nn.Module):
    """Head per predire embedding di contesto da embedding locale."""
    def __init__(self, feat_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor(z)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature

        pos = torch.cat([
            torch.diag(sim, N),
            torch.diag(sim, -N)
        ], dim=0)

        mask = ~torch.eye(2 * N, device=sim.device).bool()
        neg = sim.masked_select(mask).view(2 * N, 2 * N - 1)

        neg_logsumexp = torch.logsumexp(neg, dim=1)
        loss = -(pos - neg_logsumexp).mean()
        return loss


@register_trainer("jepa")
class JEPATrainer(BaseTrainer):
    """Trainer per JEPA."""
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)
        t = model_cfg["training"]
        self.epochs = int(t["epochs"])
        self.batch_size = int(t["batch_size"])
        self.lr = float(t["learning_rate"])
        self.weight_decay = float(t["weight_decay"])
        self.patch_size = int(model_cfg.get("patch_size", 224))
        self.context_size = int(model_cfg.get("context_size", 224))
        self.hidden_dim = int(model_cfg.get("hidden_dim", 256))
        self.aug_cfg = model_cfg.get("augmentation", {})
        self.temperature = float(t.get("temperature", 0.5))

        self.device = choose_device()
        self.train_pattern = str(Path(data_cfg["train"]))
        self.num_train = count_samples(Path(self.train_pattern))
        self.batches_train = math.ceil(self.num_train / self.batch_size)
        self.train_loader = build_jepa_loader(
            shards_pattern=self.train_pattern,
            patch_size=self.patch_size,
            context_size=self.context_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )

        backbone_name = model_cfg.get("backbone", "resnet18").lower()
        base = create_backbone(backbone_name, num_classes=0, pretrained=False)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base.to(self.device)
        self.head = JEPAHead(feat_dim, self.hidden_dim).to(self.device)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, feat_dim),
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.head.parameters()) +
            list(self.projector.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = NTXentLoss(self.temperature)

        self.best_epoch = 0
        self.best_loss = float("inf")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        local, context = batch
        local, context = local.to(self.device), context.to(self.device)
        self.encoder.train()
        self.projector.train()
        self.head.train()
        self.optimizer.zero_grad()

        h_local = self.encoder(local)
        h_context = self.encoder(context)
        z1 = self.head(h_local)
        z2 = self.projector(h_context)

        loss = self.criterion(z1, z2)
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), local.size(0)

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=torch.nn.Sequential(self.encoder, self.projector, self.head),
                optimizer=self.optimizer,
                class_to_idx={},
                model_cfg=self.model_cfg,
                data_cfg=self.data_cfg,
            )

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    def extract_features_to(self, output_path: str) -> None:
        """Estrai feature con l’encoder."""
        def _make_loader():
            ds = (
                wds.WebDataset(self.train_pattern)
                .decode("pil")
                .map(lambda s: {
                    "img": T.ToTensor()(
                        next((v for v in s.values() if hasattr(v, "convert")), None).convert("RGB")
                    ),
                    "key": s["__key__"] + "." + next((k for k in s.keys() if k.endswith(".jpg")), "")
                })
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        dataloader = _make_loader()
        feats = extract_features(self.encoder, dataloader, self.device)
        torch.save(feats, output_path)
        print(f"✅ Features saved to {output_path}")