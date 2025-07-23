# trainers/moco_v2.py

from __future__ import annotations
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
from utils.training_utils.model_utils import NTXentLoss
from .extract_features import extract_features


# ------------------------------------------------------------------------------
# MoCo v2 augmentations
# ------------------------------------------------------------------------------
class _MoCoPreproc:
    """Generate two augmented views per image for MoCo v2."""
    def __init__(self, patch_size: int, aug_cfg: Dict[str, Any]) -> None:
        self.patch_size = patch_size
        self.aug_cfg = aug_cfg
        self.transform = self._build_transform()

    def _build_transform(self) -> T.Compose:
        aug: List[nn.Module] = [T.RandomResizedCrop(self.patch_size)]
        if self.aug_cfg.get("horizontal_flip", False):
            aug.append(T.RandomHorizontalFlip())
        # strong color jitter + blur as in MoCo v2
        aug.append(T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
        aug.append(T.RandomGrayscale(p=0.2))
        aug.append(T.RandomApply([T.GaussianBlur(23, sigma=(0.1, 2.0))], p=0.5))
        aug.append(T.ToTensor())
        return T.Compose(aug)

    def __call__(self, sample: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        img = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)), None
        )
        if img is None:
            raise RuntimeError("No image found in WebDataset sample")
        return self.transform(img), self.transform(img)


def build_moco_loader(
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
            shardshuffle=False,   # inferenza → niente shuffle
            empty_check=False
        )
        .decode("pil")
        .map(_MoCoPreproc(patch_size, aug_cfg))
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
# MoCo v2 Trainer
# ------------------------------------------------------------------------------
@register_trainer("moco_v2")
class MoCoV2Trainer(BaseTrainer):
    """
    MoCo v2 self-supervised trainer.
    Trains query & key encoders with momentum and a dynamic negative queue.
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
        self.momentum    = float(t.get("momentum", 0.999))
        self.temperature = float(t.get("temperature", 0.2))
        self.queue_size  = int(t.get("queue_size", 1024))
        self.patch_size  = int(cfg.get("patch_size", 224))
        self.proj_dim    = int(cfg.get("proj_dim", 256))
        self.aug_cfg     = cfg.get("augmentation", {})

    def _init_model_and_optimizer(self, cfg: Dict[str, Any]) -> None:
        backbone = cfg.get("backbone", "resnet18").lower()

        # Query encoder + projector
        base_q = create_backbone(backbone, num_classes=0, pretrained=False)
        D = base_q.fc.in_features
        base_q.fc = nn.Identity()
        self.encoder_q   = base_q.to(self.device)
        self.projector_q = nn.Sequential(
            nn.Linear(D, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        ).to(self.device)

        # Key encoder + projector (momentum)
        base_k = create_backbone(backbone, num_classes=0, pretrained=False)
        base_k.fc = nn.Identity()
        self.encoder_k   = base_k.to(self.device)
        self.projector_k = nn.Sequential(
            nn.Linear(D, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim),
        ).to(self.device)

        # initialize key weights & freeze
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        for param in self.encoder_k.parameters():   param.requires_grad = False
        for param in self.projector_k.parameters(): param.requires_grad = False
        self.encoder_k.eval(); self.projector_k.eval()

        # dynamic queue: buffer *già* sul device corretto
        self.encoder_q.register_buffer(
            "queue",
            nn.functional.normalize(
                torch.randn(self.queue_size, self.proj_dim, device=self.device), dim=1
            ),
        )
        self.encoder_q.register_buffer(
            "queue_ptr",
            torch.zeros(1, dtype=torch.long, device=self.device),
        )

        # optimizer on query side
        params = list(self.encoder_q.parameters()) + list(self.projector_q.parameters())
        optim_name = cfg["training"].get("optimizer", "sgd").lower()
        if optim_name == "sgd":
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

        self.criterion = NTXentLoss(self.temperature)

    def _init_tracking(self) -> None:
        self.best_epoch = 0
        self.best_loss  = float("inf")

    def build_loader(self, split: str) -> DataLoader:
        loader = build_moco_loader(
            shards_pattern=self.data_cfg[split],
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            aug_cfg=self.aug_cfg,
        )
        try:
            self.batches_train = len(loader)
        except:
            self.batches_train = None
        return loader

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        # momentum update: key = m*key + (1-m)*query
        for q, k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            k.data = k.data * self.momentum + q.data * (1. - self.momentum)
        for q, k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            k.data = k.data * self.momentum + q.data * (1. - self.momentum)

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, int]:
        q_imgs, k_imgs = [x.to(self.device) for x in batch]
        self.encoder_q.train();   self.projector_q.train()
        self.encoder_k.eval();    self.projector_k.eval()
        self.optimizer.zero_grad()

        # queries
        q = F.normalize(self.projector_q(self.encoder_q(q_imgs)), dim=1)
        # updated keys
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = F.normalize(self.projector_k(self.encoder_k(k_imgs)), dim=1)

        # logits: pos vs neg
        pos    = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        neg    = torch.matmul(q, self.encoder_q.queue.t())
        logits = torch.cat([pos, neg], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)

        loss = F.cross_entropy(logits, labels)
        loss.backward(); self.optimizer.step()

        # dequeue & enqueue
        batch_size = k_imgs.size(0)
        ptr = int(self.encoder_q.queue_ptr)
        end = ptr + batch_size
        if end <= self.queue_size:
            self.encoder_q.queue[ptr:end] = k.detach()
        else:
            first = self.queue_size - ptr
            self.encoder_q.queue[ptr:]  = k[:first].detach()
            self.encoder_q.queue[: end % self.queue_size] = k[first:].detach()
        self.encoder_q.queue_ptr[0] = end % self.queue_size

        return float(loss.item()), batch_size

    def validate_epoch(self) -> Tuple[float, float]:
        raise NotImplementedError("MoCoV2Trainer does not use validation")

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

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_loss

    def get_resume_model_and_optimizer(self) -> Tuple[Any, Any]:
        return torch.nn.Sequential(self.encoder_q, self.projector_q), self.optimizer

    def extract_features_to(
        self,
        output_path: Path | str,
        split: str = "train",
    ) -> None:
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

        self.encoder_q.eval()
        with torch.no_grad():
            feats_dict = extract_features(self.encoder_q, loader, self.device)

        torch.save(feats_dict, output_path)
        print(f"✅ MoCo v2 features ({split}) saved → {output_path}")
