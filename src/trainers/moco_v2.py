from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Tuple, List

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


def build_moco_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    """
    DataLoader for MoCo-style contrastive learning with WebDataset.
    """
    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=1000,
            empty_check=False,
        )
        .decode("pil")
        .map(MoCoPreprocSample(patch_size, aug_cfg))
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


class MoCoPreprocSample:
    """
    Generate query and key augmentations for MoCo.
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
        # Strong color jitter and blur as in MoCo v2
        aug.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
        aug.append(T.RandomGrayscale(p=0.2))
        aug.append(T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5))
        aug.append(T.ToTensor())
        return T.Compose(aug)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next((v for v in sample.values() if hasattr(v, "convert")), None)
        if not hasattr(img, "convert"):
            raise RuntimeError("No image found in MoCo sample")
        return self.transform(img), self.transform(img)


@register_trainer("moco_v2")
class MoCoV2Trainer(BaseTrainer):
    """
    MoCo v2 Trainer with contrastive loss and momentum encoder.
    """
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)
        tcfg = model_cfg["training"]
        self.epochs = int(tcfg.get("epochs", 100))
        self.batch_size = int(tcfg.get("batch_size", 128))
        self.lr = float(tcfg["learning_rate"])
        self.weight_decay = float(tcfg["weight_decay"])
        self.momentum = float(tcfg.get("momentum", 0.999))
        self.temperature = float(tcfg.get("temperature", 0.2))
        self.queue_size = int(tcfg.get("queue_size", 1024))
        self.patch_size = int(model_cfg.get("patch_size", 224))
        self.proj_dim = int(model_cfg.get("proj_dim", 256))
        self.aug_cfg = model_cfg.get("augmentation", {})

        self.device = choose_device()
        self.train_pattern = str(Path(data_cfg["train"]))
        self.ckpt_dir = Path(self.train_pattern).parent / "checkpoints"
        self.num_train = count_samples(Path(self.train_pattern))
        self.batches_train = math.ceil(self.num_train / self.batch_size)

        self.train_loader = build_moco_loader(
            shards_pattern=self.train_pattern,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )

        # Query Encoder
        self.encoder_q = create_backbone(model_cfg["backbone"], num_classes=0, pretrained=False)
        D = self.encoder_q.fc.in_features
        self.encoder_q.fc = nn.Identity()
        self.encoder_q = self.encoder_q.to(self.device)

        self.projector_q = nn.Sequential(
            nn.Linear(D, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        ).to(self.device)

        # Key Encoder
        self.encoder_k = create_backbone(model_cfg["backbone"], num_classes=0, pretrained=False)
        self.encoder_k.fc = nn.Identity()
        self.encoder_k = self.encoder_k.to(self.device)

        self.projector_k = nn.Sequential(
            nn.Linear(D, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        ).to(self.device)

        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False
        # Ensure BN stats are frozen for key encoder
        self.encoder_k.eval()
        self.projector_k.eval()

        # Queue
        self.queue = nn.functional.normalize(torch.randn(self.queue_size, self.proj_dim), dim=1).to(self.device)
        self.queue_ptr = torch.zeros(1, dtype=torch.long).to(self.device)


        opt_name = model_cfg["training"].get("optimizer", "sgd").lower()
        params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        if opt_name == "sgd":
            self.optimizer = optim.SGD(params, lr=self.lr,
                                       momentum=0.9, weight_decay=self.weight_decay)
        elif opt_name == "adam":
            self.optimizer = optim.Adam(params, lr=self.lr,
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
        # Cosine annealing LR schedule with linear warmup
        self.lr_schedule = tcfg.get("lr_schedule", "cosine")
        # Warmup epochs: 10% of total, max 10
        self.warmup_epochs = min(10, int(0.1 * self.epochs))
        total_steps = self.epochs * self.batches_train
        warmup_steps = self.warmup_epochs * self.batches_train
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.best_epoch = 0
        self.best_loss = float("inf")

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        q_imgs, k_imgs = batch
        q_imgs, k_imgs = q_imgs.to(self.device), k_imgs.to(self.device)

        self.encoder_q.train()
        self.projector_q.train()
        self.encoder_k.eval()
        self.projector_k.eval()
        self.optimizer.zero_grad()

        q_feats = self.projector_q(self.encoder_q(q_imgs))  # NxD
        q_feats = nn.functional.normalize(q_feats, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_feats = self.projector_k(self.encoder_k(k_imgs))
            k_feats = nn.functional.normalize(k_feats, dim=1)

        # Compute logits
        logits_pos = torch.einsum('nc,nc->n', [q_feats, k_feats]).unsqueeze(-1)  # Nx1
        logits_neg = torch.mm(q_feats, self.queue.t())  # NxK
        logits = torch.cat([logits_pos, logits_neg], dim=1)  # Nx(1+K)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(self.device)

        logits /= self.temperature
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        # Update queue (circular, wrap-around, bugfix)
        batch_size = k_feats.shape[0]
        ptr = int(self.queue_ptr)
        end = ptr + batch_size
        if end <= self.queue_size:
            self.queue[ptr:end] = k_feats.detach()
        else:
            first = self.queue_size - ptr
            self.queue[ptr:] = k_feats[:first].detach()
            self.queue[:end % self.queue_size] = k_feats[first:].detach()
        self.queue_ptr[0] = end % self.queue_size

        return float(loss.item()), q_imgs.size(0)

    def post_epoch(self, epoch: int, epoch_loss: float) -> None:
        # Non salvare checkpoint prima della fine del warm-up
        if epoch < self.warmup_epochs:
            return

        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.best_epoch = epoch
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=torch.nn.Sequential(self.encoder_q, self.projector_q),
                optimizer=self.optimizer,
                class_to_idx={},
                model_cfg=self.model_cfg,
                data_cfg=self.data_cfg,
            )

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    def extract_features_to(self, output_path: str) -> None:
        self.encoder_q.eval()
        def _make_loader():
            ds = (
                wds.WebDataset(self.train_pattern)
                .decode("pil")
                .map(lambda s: {
                    "img": T.ToTensor()(next((v for k, v in s.items() if isinstance(v, Image.Image)), None).convert("RGB")),
                    "key": s["__key__"] + "." + next((k for k in s.keys() if k.endswith(".jpg")), "")
                })
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        dataloader = _make_loader()
        feats = extract_features(self.encoder_q, dataloader, self.device)
        torch.save(feats, output_path)
        self.logger.info(f"✅ Features saved to {output_path}")