# trainers/simclr.py

from __future__ import annotations
"""SimCLR trainer completamente self-contained.
Includes:
  • resume compatibile (get_resume_model_and_optimizer)
  • estrazione feature → .pt su qualunque split
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds

from pathlib import Path
from typing import Any, Dict, Tuple, List
from torch.utils.data import DataLoader
from PIL import Image
import io

from utils.training_utils import (
    BaseTrainer,
    choose_device,
    save_checkpoint,
    register_trainer,
    create_backbone,
)
from .extract_features import extract_features


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class _SimCLRPreproc:
    """Generates two augmented views per image."""
    def __init__(self, patch_size: int, aug_cfg: Dict[str, Any]):
        self.patch_size = patch_size
        self.aug_cfg = aug_cfg
        self.transform = self._build_transform()

    def _build_transform(self) -> T.Compose:
        aug: List[nn.Module] = [T.RandomResizedCrop(self.patch_size)]
        if self.aug_cfg.get("horizontal_flip", False):
            aug.append(T.RandomHorizontalFlip())
        if rotation := self.aug_cfg.get("rotation"):
            aug.append(T.RandomChoice([T.RandomRotation((a, a)) for a in rotation]))
        if cj := self.aug_cfg.get("color_jitter", {}):
            aug.append(T.ColorJitter(**cj))
        if self.aug_cfg.get("grayscale", False):
            aug.append(T.RandomGrayscale(p=0.2))
        if self.aug_cfg.get("gaussian_blur", False):
            aug.append(T.GaussianBlur(kernel_size=3))
        aug.append(T.ToTensor())
        return T.Compose(aug)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img: Image.Image = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)),
            None
        )
        if img is None:
            raise RuntimeError("No image found in sample – invalid WebDataset shard?")
        return self.transform(img), self.transform(img)


def build_simclr_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    """Returns a DataLoader with two augmented views per sample."""
    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=1000,
            empty_check=False,
        )
        .decode("pil")
        .map(_SimCLRPreproc(patch_size, aug_cfg))
    )
    use_cuda = device.type == "cuda"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,      # WebDataset already shuffles
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        drop_last=True,     # even-sized batches required for contrastive loss
    )


# ---------------------------------------------------------------------------
# Contrastive loss
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss (SimCLR)."""
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        mask = ~torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        neg = sim.masked_select(mask).view(2 * N, 2 * N - 1)
        return -(pos - torch.logsumexp(neg, dim=1)).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@register_trainer("simclr")
class SimCLRTrainer(BaseTrainer):
    """
    Self-supervised SimCLR trainer:
      • train encoder+projector on 'train' only
      • extract_features_to(split) per ogni split
      • supports resume & checkpointing
    """

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)
        self.device = choose_device()
        self._init_params(model_cfg)
        self._init_model_and_optimizer(model_cfg)
        self._init_tracking()

    def _init_params(self, cfg: Dict[str, Any]) -> None:
        t = cfg["training"]
        self.epochs      = int(t["epochs"])
        self.batch_size  = int(t["batch_size"])
        self.lr          = float(t["learning_rate"])
        self.weight_decay= float(t["weight_decay"])
        self.temperature = float(t.get("temperature", 0.5))
        self.patch_size  = int(cfg.get("patch_size", 224))
        self.aug_cfg     = cfg.get("augmentation", {})
        self.proj_dim    = int(cfg.get("proj_dim", 128))

    def build_loader(self, split: str) -> DataLoader:
        pattern = self.data_cfg[split]
        return build_simclr_loader(
            shards_pattern=pattern,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )

    def _init_model_and_optimizer(self, cfg: Dict[str, Any]) -> None:
        backbone = cfg.get("backbone", "resnet18").lower()
        base = create_backbone(backbone, num_classes=0, pretrained=False)
        D = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder   = base.to(self.device)
        self.projector = nn.Sequential(
            nn.Linear(D, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = NTXentLoss(self.temperature)

    def _init_tracking(self) -> None:
        self.best_epoch = 0
        self.best_loss  = float("inf")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        x1, x2 = [b.to(self.device) for b in batch]
        self.encoder.train();   self.projector.train()
        self.optimizer.zero_grad()
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        loss = self.criterion(z1, z2)
        loss.backward(); self.optimizer.step()
        return float(loss.item()), x1.size(0)

    def validate_epoch(self) -> Tuple[float, float]:
        raise NotImplementedError("SimCLRTrainer does not support validation")

    def post_epoch(self, epoch: int, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss, self.best_epoch = loss, epoch
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=torch.nn.Sequential(self.encoder, self.projector),
                optimizer=self.optimizer,
                metadata={"model_cfg": self.model_cfg, "data_cfg": self.data_cfg},
            )

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    def get_resume_model_and_optimizer(self):
        if not hasattr(self, "encoder") or not hasattr(self, "optimizer"):
            raise RuntimeError("Encoder or optimizer not initialized")
        return torch.nn.Sequential(self.encoder, self.projector), self.optimizer

    def extract_features_to(
        self,
        output_path: Path | str,
        split: str = "train",
    ) -> None:
        """
        Extract features from encoder on given split ('train','val','test').
        Saves a dict {"features": Tensor[N,C], "keys": List[str]} to output_path.
        """
        if split not in self.data_cfg:
            raise ValueError(f"Unknown split '{split}'; available: {list(self.data_cfg)}")
        shards = self.data_cfg[split]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        p = Path(shards)
        pattern = str(p / "*.tar") if p.is_dir() else shards

        ds = (
            wds.WebDataset(
                pattern,
                handler=wds.warn_and_continue,
                shardshuffle=False,
                empty_check=False,
            )
            .to_tuple("jpg", "__key__")
            .map_tuple(
                lambda jpg_bytes: T.ToTensor()(Image.open(io.BytesIO(jpg_bytes)).convert("RGB")),
                lambda key: key
            )
        )

        use_cuda = (self.device.type == "cuda")
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if use_cuda else 0,
            pin_memory=use_cuda,
        )

        self.encoder.eval()
        with torch.no_grad():
            feats_dict = extract_features(self.encoder, loader, self.device)

        torch.save(feats_dict, output_path)
        print(f"✅ SimCLR features ({split}) saved → {output_path}")
