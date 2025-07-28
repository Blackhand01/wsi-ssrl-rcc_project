from __future__ import annotations
"""
MoCo v2 trainer – versione corretta
-----------------------------------
* Augmentations da YAML (flip, jitter, blur, grayscale)
* Momentum separati: moco_momentum (EMA) vs opt_momentum (SGD)
* AMP opzionale (use_amp) con GradScaler
* Cosine LR + warm-up
* **FIX**: non chiama più len(loader) su un IterableDataset
"""

import math, io
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds
from PIL import Image
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from utils.training_utils import (
    BaseTrainer, choose_device, save_checkpoint, register_trainer, create_backbone,
)
from utils.training_utils.model_utils import NTXentLoss
from .extract_features import extract_features

# ----------------------------------------------------------------------------- #
# 1. Augmentation pipeline                                                      #
# ----------------------------------------------------------------------------- #
class _MoCoPreproc:
    """Produce due viste augmentate per MoCo v2."""

    def __init__(self, patch_size: int, aug_cfg: Dict[str, Any]) -> None:
        self.patch_size = patch_size
        self.aug_cfg = aug_cfg or {}
        self.transform = self._build_transform()

    def _cj_params(self) -> Tuple[float, float, float, float]:
        cj = self.aug_cfg.get("color_jitter", {})
        return (
            cj.get("brightness", 0.4),
            cj.get("contrast",   0.4),
            cj.get("saturation", 0.4),
            cj.get("hue",        0.1),
        )

    def _build_transform(self) -> T.Compose:
        tr: List[nn.Module] = [T.RandomResizedCrop(self.patch_size)]
        if self.aug_cfg.get("horizontal_flip", False):
            tr.append(T.RandomHorizontalFlip())
        tr.append(T.RandomApply([T.ColorJitter(*self._cj_params())], p=0.8))
        gray_p = float(self.aug_cfg.get("grayscale", 0.2))
        tr.append(T.RandomGrayscale(p=gray_p))
        if self.aug_cfg.get("gaussian_blur", False):
            tr.append(T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5))
        tr.append(T.ToTensor())
        return T.Compose(tr)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)), None
        )
        if img is None:
            raise RuntimeError("MoCoPreproc: no image in WebDataset sample")
        return self.transform(img), self.transform(img)

# ----------------------------------------------------------------------------- #
# 2. DataLoader helper                                                          #
# ----------------------------------------------------------------------------- #
def build_moco_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    ds = (
        wds.WebDataset(
            shards_pattern, handler=wds.warn_and_continue,
            shardshuffle=False, empty_check=False
        )
        .decode("pil")
        .map(_MoCoPreproc(patch_size, aug_cfg))
    )
    use_cuda = device.type == "cuda"
    p = Path(shards_pattern)
    pattern = str(p / "*.tar") if p.is_dir() else shards_pattern
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

# ----------------------------------------------------------------------------- #
# 3. Trainer                                                                    #
# ----------------------------------------------------------------------------- #
@register_trainer("moco_v2")
class MoCoV2Trainer(BaseTrainer):
    """Momentum Contrast v2 trainer."""

    # ---------------------------- init --------------------------------------- #
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)
        self.device = choose_device()
        self._init_params(model_cfg)
        self._init_model_and_optimizer(model_cfg)
        self._init_tracking()

    def _init_params(self, cfg: Dict[str, Any]) -> None:
        t = cfg["training"]
        self.epochs        = int(t["epochs"])
        self.batch_size    = int(t["batch_size"])
        self.lr            = float(t["learning_rate"])
        self.weight_decay  = float(t["weight_decay"])
        self.moco_momentum = float(t.get("momentum", 0.99))   # EMA
        self.opt_momentum  = float(t.get("opt_momentum", 0.9))# SGD
        self.temperature   = float(t.get("temperature", 0.2))
        self.queue_size    = int(t.get("queue_size", 1024))
        self.patch_size    = int(cfg.get("patch_size", 224))
        self.proj_dim      = int(cfg.get("proj_dim", 256))
        self.aug_cfg       = cfg.get("augmentation", {})
        self.lr_schedule   = t.get("lr_schedule", None)
        self.warmup_epochs = int(t.get("warmup_epochs", 0))
        self.use_amp       = bool(t.get("use_amp", True)) and self.device.type == "cuda"
        self.scaler        = GradScaler(enabled=self.use_amp)

    # --------------------- model / optimizer / scheduler --------------------- #
    def _init_model_and_optimizer(self, cfg: Dict[str, Any]) -> None:
        backbone = cfg.get("backbone", "resnet18").lower()

        # --- query encoder --------------------------------------------------- #
        base_q = create_backbone(backbone, num_classes=0, pretrained=False)
        feat_dim = base_q.fc.in_features
        base_q.fc = nn.Identity()
        self.encoder_q = base_q.to(self.device)
        self.projector_q = nn.Sequential(
            nn.Linear(feat_dim, self.proj_dim), nn.ReLU(), nn.Linear(self.proj_dim, self.proj_dim)
        ).to(self.device)

        # --- key encoder (EMA) ---------------------------------------------- #
        base_k = create_backbone(backbone, num_classes=0, pretrained=False)
        base_k.fc = nn.Identity()
        self.encoder_k = base_k.to(self.device)
        self.projector_k = nn.Sequential(
            nn.Linear(feat_dim, self.proj_dim), nn.ReLU(), nn.Linear(self.proj_dim, self.proj_dim)
        ).to(self.device)

        # sync & freeze key
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        for p in list(self.encoder_k.parameters()) + list(self.projector_k.parameters()):
            p.requires_grad = False
        self.encoder_k.eval(); self.projector_k.eval()

        # queue
        self.encoder_q.register_buffer(
            "queue", F.normalize(torch.randn(self.queue_size, self.proj_dim, device=self.device), dim=1)
        )
        self.encoder_q.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long, device=self.device))

        # optimizer
        params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        if cfg["training"].get("optimizer", "sgd").lower() == "sgd":
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=self.opt_momentum, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        # loss & scheduler
        self.criterion = NTXentLoss(self.temperature)
        self.scheduler = None
        if self.lr_schedule == "cosine":
            def lr_lambda(ep: int) -> float:
                if ep < self.warmup_epochs:
                    return (ep + 1) / max(1, self.warmup_epochs)
                prog = (ep - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
                return 0.5 * (1.0 + math.cos(math.pi * prog))
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _init_tracking(self) -> None:
        self.best_epoch = 0
        self.best_loss  = float("inf")

    # --------------------------- dataloader ---------------------------------- #
    def build_loader(self, split: str) -> DataLoader:
        loader = build_moco_loader(
            shards_pattern=self.data_cfg[split],
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )
        # WebDataset è Iterable: la lunghezza non è definita
        self.batches_train = None
        return loader

    # --------------------------- momentum update ----------------------------- #
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data.mul_(self.moco_momentum).add_(q.data, alpha=1.0 - self.moco_momentum)
        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data.mul_(self.moco_momentum).add_(q.data, alpha=1.0 - self.moco_momentum)

    # ------------------------------ train step ------------------------------- #
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        q_img, k_img = [x.to(self.device) for x in batch]
        self.encoder_q.train(); self.projector_q.train()
        self.encoder_k.eval();  self.projector_k.eval()
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            q = F.normalize(self.projector_q(self.encoder_q(q_img)), dim=1)
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k = F.normalize(self.projector_k(self.encoder_k(k_img)), dim=1)

            pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
            neg = torch.matmul(q, self.encoder_q.queue.t())
            logits = torch.cat([pos, neg], dim=1) / self.temperature
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
            loss = F.cross_entropy(logits, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # queue
        bs = k_img.size(0)
        ptr = int(self.encoder_q.queue_ptr)
        end = ptr + bs
        if end <= self.queue_size:
            self.encoder_q.queue[ptr:end] = k.detach()
        else:
            first = self.queue_size - ptr
            self.encoder_q.queue[ptr:] = k[:first].detach()
            self.encoder_q.queue[: end % self.queue_size] = k[first:].detach()
        self.encoder_q.queue_ptr[0] = end % self.queue_size

        return float(loss.item()), bs

    # --------------------------- epoch callbacks ----------------------------- #
    def post_epoch(self, epoch: int, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss, self.best_epoch = loss, epoch
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=torch.nn.Sequential(self.encoder_q, self.projector_q),
                optimizer=self.optimizer,
                metadata={"model_cfg": self.model_cfg, "data_cfg": self.data_cfg},
            )
        if self.scheduler is not None:
            self.scheduler.step()

    # ----------------------------- misc API ---------------------------------- #
    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    def get_resume_model_and_optimizer(self) -> Tuple[Any, Any]:
        return torch.nn.Sequential(self.encoder_q, self.projector_q), self.optimizer

    # --------------------------- feature export ------------------------------ #
    def extract_features_to(self, output_path: Path | str, split: str = "train") -> None:
        if split not in self.data_cfg:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self.data_cfg)}")
        output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
        p = Path(self.data_cfg[split]); pattern = str(p / "*.tar") if p.is_dir() else str(p)

        ds = (
            wds.WebDataset(pattern, handler=wds.warn_and_continue, shardshuffle=False, empty_check=False)
            .to_tuple("jpg", "__key__")
            .map_tuple(lambda b: T.ToTensor()(Image.open(io.BytesIO(b)).convert("RGB")), lambda k: k)
        )
        loader = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if self.device.type == "cuda" else 0,
            pin_memory=self.device.type == "cuda",
        )

        self.encoder_q.eval()
        with torch.no_grad():
            feats = extract_features(self.encoder_q, loader, self.device)
        torch.save(feats, output_path)
        print(f"✅ MoCo v2 features ({split}) saved → {output_path}")
