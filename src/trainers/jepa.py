# trainers/jepa.py

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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


# ------------------------------------------------------------------------------
# Dataset helper for JEPA: local patch + context patch
# ------------------------------------------------------------------------------
class JEPADataPrep:
    """Generate one local and one context patch per sample for JEPA."""

    def __init__(self, patch_size: int, context_size: int, aug_cfg: Dict[str, Any]) -> None:
        self.patch_size = patch_size
        self.context_size = context_size
        self.aug_cfg = aug_cfg
        self.local_transform   = self._build_transform(self.patch_size)
        self.context_transform = self._build_transform(self.context_size)

    def _build_transform(self, size: int) -> T.Compose:
        aug: List[nn.Module] = [T.RandomResizedCrop(size)]
        if self.aug_cfg.get("horizontal_flip", False):
            aug.append(T.RandomHorizontalFlip())
        if cj := self.aug_cfg.get("color_jitter", {}):
            aug.append(T.ColorJitter(**cj))
        if self.aug_cfg.get("grayscale", False):
            aug.append(T.RandomGrayscale(p=0.2))
        if self.aug_cfg.get("gaussian_blur", False):
            aug.append(T.GaussianBlur(kernel_size=3))
        aug.append(T.ToTensor())
        return T.Compose(aug)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)), None
        )
        if img is None:
            raise RuntimeError(f"JEPA sample has no image: keys={list(sample.keys())}")
        img = img.convert("RGB")
        return self.local_transform(img), self.context_transform(img)


def build_jepa_loader(
    shards_pattern: str,
    patch_size: int,
    context_size: int,
    batch_size: int,
    device: torch.device,
    aug_cfg: Dict[str, Any],
) -> DataLoader:
    """Return a DataLoader yielding (local_patch, context_patch) tuples."""
    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=False,   # inferenza → niente shuffle
            empty_check=False
        )
        .decode("pil")
        .map(JEPADataPrep(patch_size, context_size, aug_cfg))
    )
    use_cuda = (device.type == "cuda")
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


# ------------------------------------------------------------------------------
# JEPA Trainer
# ------------------------------------------------------------------------------
@register_trainer("jepa")
class JEPATrainer(BaseTrainer):
    """
    JEPA self-supervised trainer.
    Trains encoder + head + projector to predict context embedding from local patch.
    """

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
        self.patch_size   = int(cfg.get("patch_size", 224))
        self.context_size = int(cfg.get("context_size", 224))
        self.hidden_dim   = int(cfg.get("hidden_dim", 256))
        self.temperature  = float(t.get("temperature", 0.5))
        self.aug_cfg      = cfg.get("augmentation", {})

    def build_loader(self, split: str) -> DataLoader:
        pattern = self.data_cfg[split]
        loader = build_jepa_loader(
            shards_pattern=pattern,
            patch_size=self.patch_size,
            context_size=self.context_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )
        try:
            self.batches_train = len(loader)
        except Exception:
            self.batches_train = None
        return loader

    def _init_model_and_optimizer(self, cfg: Dict[str, Any]) -> None:
        backbone_name = cfg.get("backbone", "resnet18").lower()
        base = create_backbone(backbone_name, num_classes=0, pretrained=False)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()

        self.encoder   = base.to(self.device)
        self.head      = nn.Sequential(
            nn.Linear(feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, feat_dim),
        ).to(self.device)
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

    def _init_tracking(self) -> None:
        self.best_epoch = 0
        self.best_loss  = float("inf")

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        local, context = batch
        local, context = local.to(self.device), context.to(self.device)
        self.encoder.train(); self.head.train(); self.projector.train()
        self.optimizer.zero_grad()

        h_local   = self.encoder(local)
        h_context = self.encoder(context)
        z1 = self.head(h_local)
        z2 = self.projector(h_context)

        # contrastive NT-Xent loss
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        mask = ~torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        neg = sim.masked_select(mask).view(2 * N, 2 * N - 1)
        loss = -(pos - torch.logsumexp(neg, dim=1)).mean()

        loss.backward()
        self.optimizer.step()
        return float(loss.item()), local.size(0)

    def validate_epoch(self) -> Tuple[float, float]:
        raise NotImplementedError("JEPATrainer has no validation loop")

    def post_epoch(self, epoch: int, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss, self.best_epoch = loss, epoch
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=torch.nn.Sequential(self.encoder, self.head, self.projector),
                optimizer=self.optimizer,
                metadata={"model_cfg": self.model_cfg, "data_cfg": self.data_cfg},
            )

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    def get_resume_model_and_optimizer(self):
        return torch.nn.Sequential(self.encoder, self.head, self.projector), self.optimizer

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
        print(f"✅ JEPA features ({split}) saved → {output_path}")
