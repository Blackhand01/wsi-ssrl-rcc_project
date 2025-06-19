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


def build_simclr_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    """
    WebDataset loader per SimCLR: decodifica PIL, due viste augmentate.
    """
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
    use_cuda = device.type == "cuda"
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
    Genera due viste augmentate per SimCLR.
    """
    def __init__(self, patch_size: int, aug_cfg: Dict[str, Any]) -> None:
        self.patch_size = patch_size
        self.aug_cfg = aug_cfg
        self.transform = self._build_transform()

    def _build_transform(self) -> T.Compose:
        aug = [T.RandomResizedCrop(self.patch_size)]
        if self.aug_cfg.get("horizontal_flip", False):
            aug.append(T.RandomHorizontalFlip())
        if self.aug_cfg.get("rotation"):
            angles = self.aug_cfg["rotation"]
            aug.append(T.RandomChoice([T.RandomRotation((a, a)) for a in angles]))
        if cj := self.aug_cfg.get("color_jitter", {}):
            aug.append(T.ColorJitter(**cj))
        if self.aug_cfg.get("grayscale", False):
            aug.append(T.RandomGrayscale(p=0.2))
        if self.aug_cfg.get("gaussian_blur", False):
            aug.append(T.GaussianBlur(kernel_size=3))
        aug.append(T.ToTensor())
        return T.Compose(aug)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next((v for v in sample.values() if hasattr(v, "convert")), None)
        if not hasattr(img, "convert"):
            raise RuntimeError("No image found in SimCLR sample")
        return self.transform(img), self.transform(img)


class NTXentLoss(nn.Module):
    """
    NT-Xent loss stabile, basata su log-sum-exp.
    """
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # 2N x D
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature  # 2N x 2N

        # Positive similarities
        pos = torch.cat([
            torch.diag(sim, N),
            torch.diag(sim, -N)
        ], dim=0)  # 2N

        # Mask out self-similarities
        mask = ~torch.eye(2 * N, device=sim.device).bool()
        neg = sim.masked_select(mask).view(2 * N, 2 * N - 1)

        neg_logsumexp = torch.logsumexp(neg, dim=1)
        loss = -(pos - neg_logsumexp).mean()
        return loss


@register_trainer("simclr")
class SimCLRTrainer(BaseTrainer):
    """
    Trainer SimCLR con batch-wise training e NT-Xent loss stabile.
    """
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)
        self._init_training_params(model_cfg)
        self.device = self._init_device()
        self._init_paths(data_cfg)
        self._init_dataloader()
        self._init_model_and_optimizer(model_cfg)
        self._init_tracking()

    def _init_training_params(self, model_cfg: Dict[str, Any]) -> None:
        t = model_cfg["training"]
        self.epochs = int(t["epochs"])
        self.batch_size = int(t["batch_size"])
        self.lr = float(t["learning_rate"])
        self.weight_decay = float(t["weight_decay"])
        self.temperature = float(t.get("temperature", 0.5))
        self.patch_size = int(model_cfg.get("patch_size", 224))
        self.aug_cfg = model_cfg.get("augmentation", {})

    def _init_device(self) -> torch.device:
        return choose_device()

    def _init_paths(self, data_cfg: Dict[str, Any]) -> None:
        self.train_pattern = str(Path(data_cfg["train"]))
        self.num_train = count_samples(Path(self.train_pattern))
        self.batches_train = math.ceil(self.num_train / self.batch_size)

    def _init_dataloader(self) -> None:
        self.train_loader = build_simclr_loader(
            shards_pattern=self.train_pattern,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )

    def _init_model_and_optimizer(self, model_cfg: Dict[str, Any]) -> None:
        backbone_name = model_cfg.get("backbone", "resnet18").lower()
        base = create_backbone(backbone_name, num_classes=0, pretrained=False)
        D = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base.to(self.device)
        self.projector = nn.Sequential(
            nn.Linear(D, model_cfg.get("proj_dim", 128)),
            nn.ReLU(),
            nn.Linear(model_cfg.get("proj_dim", 128), model_cfg.get("proj_dim", 128)),
        ).to(self.device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.projector.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = NTXentLoss(self.temperature)

    def _init_tracking(self) -> None:
        self.best_epoch = 0
        self.best_loss = float("inf")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        x1, x2 = batch
        x1, x2 = x1.to(self.device), x2.to(self.device)
        self.encoder.train()
        self.projector.train()
        self.optimizer.zero_grad()
        h1, h2 = self.encoder(x1), self.encoder(x2)
        z1, z2 = self.projector(h1), self.projector(h2)
        loss = self.criterion(z1, z2)
        loss.backward()
        self.optimizer.step()
        return float(loss.item()), x1.size(0)

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            ckpt_dir = Path(self.train_pattern).parent / "simclr" / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir=ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=torch.nn.Sequential(self.encoder, self.projector),
                optimizer=self.optimizer,
                class_to_idx={}, model_cfg=self.model_cfg, data_cfg=self.data_cfg,
            )

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss
    
    def extract_features_to(self, output_path: str) -> None:
        """
        Estrae le feature usando il backbone e le salva in output_path.
        """

        def _make_inference_loader():
          ds = (
              wds.WebDataset(self.train_pattern)
              .decode("pil")
              .map(lambda sample: {
                  "img": T.ToTensor()(
                      next((v for k, v in sample.items() if isinstance(v, Image.Image)), None).convert("RGB")
                  ),
                  "key": sample["__key__"] + "." + next((k for k in sample.keys() if k.endswith(".jpg")), "")

              })

          )
          return DataLoader(ds, batch_size=self.batch_size, shuffle=False)


        dataloader = _make_inference_loader()
        feats = extract_features(self.encoder, dataloader, self.device)
        torch.save(feats, output_path)
        self.logger.info(f"✅ Feature salvate in {output_path}")