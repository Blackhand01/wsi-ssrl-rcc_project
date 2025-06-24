from __future__ import annotations
import itertools
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
# Utility functions
#######################################################################

def _default_permutations(n_perm: int = 30, grid: int = 3) -> List[Tuple[int, ...]]:
    """Return a list of Jigsaw permutations with large Hamming distance."""
    random.seed(0)
    all_perms = list(itertools.permutations(range(grid ** 2)))
    perms: List[Tuple[int, ...]] = []
    while len(perms) < n_perm and all_perms:
        cand = random.choice(all_perms)
        if all(sum(ci != pi for ci, pi in zip(cand, p)) >= 5 for p in perms):
            perms.append(cand)
        all_perms.remove(cand)
    return perms


def _make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")

#######################################################################
# Dataset & DataLoader
#######################################################################

class JigsawPreprocSample:
    """Split an image into tiles, permute them, return tiles tensor + perm index."""

    def __init__(self, patch_size: int, perms: List[Tuple[int, ...]]):
        self.patch_size = patch_size
        self.perms      = perms
        self.grid       = int(math.sqrt(len(perms[0])))
        self.tile_size  = patch_size // self.grid

        self.pre_aug = T.Compose([
            T.RandomResizedCrop(patch_size),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        self.to_tensor = T.ToTensor()

    def __call__(self, sample: dict):
        img = sample.get("jpg") or next((v for v in sample.values() if isinstance(v, Image.Image)), None)
        if not isinstance(img, Image.Image):
            raise RuntimeError("No image found in Jigsaw sample")

        img = self.pre_aug(img.convert("RGB"))

        # split into tiles
        tiles: List[Image.Image] = []
        for i in range(self.grid):
            for j in range(self.grid):
                left  = j * self.tile_size
                upper = i * self.tile_size
                tile = img.crop((left, upper, left + self.tile_size, upper + self.tile_size))
                tiles.append(tile)

        perm_idx = random.randint(0, len(self.perms) - 1)
        perm     = self.perms[perm_idx]
        perm_tiles = [tiles[p] for p in perm]
        tiles_tensor = torch.stack([self.to_tensor(t) for t in perm_tiles])  # (9,C,H,W)
        return tiles_tensor, perm_idx


def build_jigsaw_loader(
    shards_pattern: str,
    patch_size: int,
    batch_size: int,
    device: torch.device,
    n_perm: int,
) -> DataLoader:
    perms = _default_permutations(n_perm)
    ds = (
        wds.WebDataset(shards_pattern, handler=wds.warn_and_continue, shardshuffle=1000, empty_check=False)
        .decode("pil")
        .map(JigsawPreprocSample(patch_size, perms))
    )
    use_cuda = device.type == "cuda"
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4 if use_cuda else 0, pin_memory=use_cuda, drop_last=True)

#######################################################################
# Trainer
#######################################################################

@register_trainer("jigsaw")
class JigsawTrainer(BaseTrainer):
    """Self‑supervised **Jigsaw** trainer (predict tile permutation)."""

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        super().__init__(model_cfg, data_cfg)

        # ---------------- Hyper‑parameters ----------------
        t = model_cfg["training"]
        self.epochs        = int(t["epochs"])
        self.batch_size    = int(t["batch_size"])
        self.lr            = float(t["learning_rate"])
        self.weight_decay  = float(t["weight_decay"])
        self.optimizer_name= t.get("optimizer", "adam")
        self.patch_size    = int(model_cfg.get("patch_size", 224))
        self.n_perm        = int(model_cfg.get("n_permutations", 30))
        self.hidden_dim    = int(model_cfg.get("hidden_dim", 1024))

        # ---------------- Device & data ----------------
        self.device        = choose_device()
        self.train_pattern = str(Path(data_cfg["train"]))
        self.num_train     = count_samples(Path(self.train_pattern))
        self.batches_train = math.ceil(self.num_train / self.batch_size)

        self.train_loader  = build_jigsaw_loader(
            shards_pattern=self.train_pattern,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            n_perm=self.n_perm,
        )

        # ---------------- Model ----------------
        backbone_name = model_cfg.get("backbone", "resnet18").lower()
        base = create_backbone(backbone_name, num_classes=0, pretrained=False)
        dim_feats = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base.to(self.device)

        self.head = nn.Sequential(
            nn.Linear(dim_feats * 9, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.n_perm),
        ).to(self.device)

        # ---------------- Optimizer & Loss ----------------
        self.optimizer = _make_optimizer(self.optimizer_name, list(self.encoder.parameters()) + list(self.head.parameters()), self.lr, self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        # ---------------- Tracking ----------------
        self.best_epoch = 0
        self.best_loss  = float("inf")

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(self, batch):
        tiles, perm_idx = batch
        B, N, C, H, W = tiles.shape
        tiles   = tiles.view(B * N, C, H, W).to(self.device)
        perm_idx= perm_idx.to(self.device)

        self.encoder.train(); self.head.train()
        self.optimizer.zero_grad()

        feats  = self.encoder(tiles)          # (B*N, D)
        feats  = feats.view(B, -1)            # (B, D*9)
        logits = self.head(feats)             # (B, n_perm)
        loss   = self.criterion(logits, perm_idx)
        loss.backward(); self.optimizer.step()
        return float(loss.item()), B

    def post_epoch(self, epoch: int, epoch_loss: float):
        if epoch_loss < self.best_loss:
            self.best_loss, self.best_epoch = epoch_loss, epoch
            ckpt_dir = Path(self.train_pattern).parent / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir=ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=torch.nn.Sequential(self.encoder, self.head),
                optimizer=self.optimizer,
                class_to_idx={},
                model_cfg=self.model_cfg,
                data_cfg=self.data_cfg,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    # ------------------------------------------------------------------
    # Feature extraction (tile-level)
    # ------------------------------------------------------------------

    def extract_features_to(self, output_path: str):
        """Extract tile‑level features with *encoder* and save to *output_path*."""
        from .extract_features import extract_features  # avoid circular imports

        def _make_loader():
            ds = (
                wds.WebDataset(self.train_pattern)
                .decode("pil")
                .map(lambda sample: {
                    "img": T.ToTensor()(next((v for k, v in sample.items() if isinstance(v, Image.Image)), None).convert("RGB")),
                    "key": sample["__key__"] + "." + next((k for k in sample.keys() if k.endswith(".jpg")), "")
                })
            )
            return DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        self.encoder.eval()
        dl = _make_loader()
        feats = extract_features(self.encoder, dl, self.device)
        torch.save(feats, output_path)
        print(f"✅ Jigsaw features saved to {output_path}")
