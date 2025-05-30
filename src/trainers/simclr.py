import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from launch_training import BaseTrainer, register_trainer  # noqa: E402

LOGGER = logging.getLogger(__name__)

def _default_transforms(patch_size: int) -> T.Compose:
    """
    Build default image transformations for SimCLR: two views.
    """
    return T.Compose([
        T.RandomResizedCrop(patch_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.5, 0.5, 0.5, 0.2),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ])

class _PreprocSimCLR:
    """
    For SimCLR: apply two augmentations to each image.
    """
    def __init__(self, patch_size: int):
        self.tfms = _default_transforms(patch_size)

    def __call__(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Cerca la prima immagine PIL
        img = next((v for k, v in sample.items() if isinstance(v, Image.Image)), None)
        if img is None:
            raise RuntimeError(f"No image found in sample {sample.get('__key__', '')}")
        return self.tfms(img), self.tfms(img)

@register_trainer("simclr")
class SimCLRTrainer(BaseTrainer):
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        super().__init__(model_cfg, data_cfg)

        # Hyperparameters
        t = model_cfg["training"]
        self.epochs = int(t["epochs"])
        self.batch_size = int(t["batch_size"])
        self.lr = float(t["learning_rate"])
        self.weight_decay = float(t["weight_decay"])
        self.temperature = float(t["temperature"])
        self.patch_size = int(model_cfg.get("patch_size", 112))

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        LOGGER.info("Using device: %s", self.device)

        # Data patterns
        self.train_pattern = data_cfg["train"]
        train_dir = Path(self.train_pattern).parent
        self.dataset_root = train_dir.parent

        # DataLoader
        self.train_loader = self._make_loader(self.train_pattern)

        # Model
        backbone_name = model_cfg.get("backbone", "resnet18")
        self.backbone = getattr(models, backbone_name)(weights=None)
        dim_mlp = self.backbone.fc.in_features
        proj_dim = model_cfg.get("proj_dim", 128)
        self.backbone.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, proj_dim),
        )
        self.backbone = self.backbone.to(self.device)
        self.projector = self.projector.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) + list(self.projector.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )

    def _make_loader(self, shards_pattern: str) -> DataLoader:
        ds = (
            wds.WebDataset(
                shards_pattern,
                handler=wds.warn_and_continue,
                shardshuffle=1000,
                empty_check=False,
            )
            .decode("pil")
            .shuffle(1000)
        )

        preproc = _PreprocSimCLR(self.patch_size)
        ds = ds.map(preproc)

        is_cuda = (self.device.type == "cuda")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if is_cuda else 0,
            pin_memory=is_cuda,
            drop_last=True,
        )

    def _contrastive_loss(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)
        z = nn.functional.normalize(z, dim=1)

        sim = torch.matmul(z, z.T) / self.temperature
        mask = (~torch.eye(2 * batch_size, dtype=bool)).to(self.device)

        sim = sim.masked_select(mask).view(2 * batch_size, -1)
        positives = torch.cat([
            torch.diag(sim, batch_size),
            torch.diag(sim, -batch_size)
        ], dim=0)

        labels = torch.zeros(2 * batch_size, device=self.device, dtype=torch.long)
        loss = nn.functional.cross_entropy(sim, labels)
        return loss

    def train(self):
        self.backbone.train()
        self.projector.train()

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            num_batches = 0

            for batch_idx, (x1, x2) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
                x1, x2 = x1.to(self.device), x2.to(self.device)

                z1 = self.projector(self.backbone(x1))
                z2 = self.projector(self.backbone(x2))

                loss = self._contrastive_loss(z1, z2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * x1.size(0)
                num_batches += 1

            avg_loss = total_loss / (num_batches * self.batch_size)
            LOGGER.info(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")

            self._save_checkpoint(epoch)


    def _save_checkpoint(self, epoch):
        ckpt_dir = self.dataset_root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        torch.save({
            "epoch": epoch,
            "backbone_state": self.backbone.state_dict(),
            "projector_state": self.projector.state_dict(),
        }, ckpt_dir / f"simclr_epoch{epoch:03d}.pt")
        LOGGER.info(f"💾 Saved checkpoint at epoch {epoch}")