from __future__ import annotations
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.models as models
import tarfile
import math

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Set of all known class labels for the RCC dataset.
KNOWN_LABELS = {"ccRCC", "pRCC", "CHROMO", "ONCO", "not_tumor"}


# -----------------------------------------------------------------------------
# Trainer Registry System
# -----------------------------------------------------------------------------
"""
trainer_registry
----------------
Implements a registry for trainer classes and a base trainer interface.
"""

from typing import Type

TRAINER_REGISTRY: Dict[str, Type] = {}

def register_trainer(name: str):
    """
    Decorator to register a trainer class under a given name.
    """
    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator

class BaseTrainer:
    """
    Abstract base class for all trainers.
    """
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg

    def train(self):
        raise NotImplementedError("Trainer must implement train()")


# -----------------------------------------------------------------------------
# Device Selection
# -----------------------------------------------------------------------------
def choose_device() -> torch.device:
    """
    Selects the best available device for training:
    - CUDA if available
    - Apple MPS if available
    - Otherwise, CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Data Transforms and Label Parsing
# -----------------------------------------------------------------------------
def default_transforms(patch_size: int, augment: bool) -> T.Compose:
    """
    Builds a torchvision transformation pipeline for image preprocessing.
    Includes resizing, cropping, tensor conversion, and optional augmentation.
    """
    transforms: List[Any] = []
    if augment:
        transforms += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomChoice([
                T.RandomRotation((0, 0)),
                T.RandomRotation((90, 90)),
                T.RandomRotation((180, 180)),
                T.RandomRotation((270, 270)),
            ]),
        ]
    transforms += [
        T.Resize(patch_size),
        T.CenterCrop(patch_size),
        T.ToTensor(),
    ]
    return T.Compose(transforms)

def parse_label_from_key(key: str) -> str:
    """
    Extracts the class label from a WebDataset sample key.
    Handles the special case for 'not_tumor'.
    """
    stem = Path(key).stem
    parts = stem.split("_")
    if parts[:2] == ["not", "tumor"]:
        return "not_tumor"
    return parts[0]

class PreprocSample:
    """
    Callable class to preprocess a WebDataset sample.
    Converts an image to a tensor and maps its label to an integer index.
    """
    def __init__(self,
                 class_to_idx: Dict[str,int],
                 patch_size: int,
                 augment: bool):
        self.class_to_idx = class_to_idx
        self.tfms = default_transforms(patch_size, augment)

    def __call__(self, sample: Dict[str,Any]) -> Tuple[torch.Tensor,int]:
        key = sample.get("__key__", "<unknown>")
        img = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)),
            None
        )
        if not isinstance(img, Image.Image):
            raise RuntimeError(f"No image found in sample '{key}'")

        label_str = parse_label_from_key(key)
        if label_str not in self.class_to_idx:
            raise RuntimeError(f"Unknown label '{label_str}' in sample '{key}'")
        return self.tfms(img), self.class_to_idx[label_str]


# -----------------------------------------------------------------------------
# Class Discovery & Sample Counting
# -----------------------------------------------------------------------------
def discover_classes(train_dir: Path) -> Dict[str,int]:
    """
    Scans training .tar shards to find all known labels.
    Returns a mapping from label string to integer index.
    """
    shards = list(train_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {train_dir}")
    found: set[str] = set()
    for tar in shards:
        with tarfile.open(tar) as tf:
            for member in tf.getmembers():
                if member.isfile() and member.name.lower().endswith(".jpg"):
                    lbl = parse_label_from_key(Path(member.name).stem)
                    if lbl in KNOWN_LABELS:
                        found.add(lbl)
        if found == KNOWN_LABELS:
            break
    if not found:
        raise RuntimeError("No valid labels found in training shards")
    return {c: i for i, c in enumerate(sorted(found))}

def count_samples(pattern: Path) -> int:
    """
    Counts the total number of .jpg files across all .tar shards in the given folder.
    """
    folder = pattern.parent
    total = 0
    for tar in folder.glob("*.tar"):
        with tarfile.open(tar) as tf:
            total += sum(1 for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".jpg"))
    return total


# -----------------------------------------------------------------------------
# DataLoader Factory
# -----------------------------------------------------------------------------
def build_loader(shards_pattern: str,
                 class_to_idx: Dict[str,int],
                 patch_size: int,
                 batch_size: int,
                 device: torch.device,
                 augment: bool) -> DataLoader:
    """
    Creates a DataLoader for a WebDataset of image patches.
    Applies preprocessing and optional augmentation.
    """
    ds = (
        wds.WebDataset(
            shards_pattern,
            handler=wds.warn_and_continue,
            shardshuffle=1000 if augment else 0,
            empty_check=False,
        )
        .decode("pil")
        .map(PreprocSample(class_to_idx, patch_size, augment))
    )
    if augment:
        ds = ds.shuffle(1000)
    use_cuda = (device.type == "cuda")
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
    )


# -----------------------------------------------------------------------------
# Backbone Factory
# -----------------------------------------------------------------------------
def create_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Instantiates a ResNet backbone (resnet18 or resnet50) and replaces its final FC layer.
    """
    if name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
    elif name == "resnet18":
        model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise ValueError(f"Unsupported backbone '{name}'")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# -----------------------------------------------------------------------------
# Checkpointing Utilities
# -----------------------------------------------------------------------------
def save_checkpoint(ckpt_dir: Path,
                    prefix: str,
                    *,
                    epoch: int,
                    best: bool,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    class_to_idx: Dict[str,int],
                    model_cfg: Dict[str,Any],
                    data_cfg: Dict[str,Any]) -> None:
    """
    Saves model and optimizer state to disk.
    Uses a naming convention: <timestamp>_<prefix>_best.pt or <timestamp>_<prefix>_epochXXX.pt.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tag = "best" if best else f"epoch{epoch:03d}"
    filename = f"{timestamp}_{prefix}_{tag}.pt"
    path = ckpt_dir / filename
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
    }, path)
    print(f"Checkpoint saved → {path}")

def load_checkpoint(ckpt_path: Path,
                    model: nn.Module,
                    optimizer: Optional[optim.Optimizer] = None) -> Dict[str, Any]:
    """
    Loads model (and optionally optimizer) state from a checkpoint file.
    Returns the checkpoint dictionary.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optim_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state_dict"])
    print(f"Loaded checkpoint '{ckpt_path}' (epoch {ckpt.get('epoch', -1)})")
    return ckpt

def get_latest_checkpoint(ckpt_dir: Path, prefix: str) -> Path:
    """
    Returns the path of the most recent checkpoint file matching the given prefix.
    """
    files = list(ckpt_dir.glob(f"{prefix}_*.pt"))
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


# -----------------------------------------------------------------------------
# Training Report Generation
# -----------------------------------------------------------------------------
def save_report_md(history: List[Dict[str, float]],
                   best_epoch: int,
                   best_acc: float,
                   model_cfg: Dict[str, Any],
                   class_to_idx: Dict[str, int],
                   num_train: int,
                   num_val: int,
                   out_root: Path) -> None:
    """
    Writes a detailed Markdown training report to:
    report/training/YYYY-MM-DD_HHMMSS_<model_name>.md

    Includes configuration, class mapping, dataset sizes, training history, and best model info.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    model_name = model_cfg.get("name", "model")
    out_dir = out_root / "report" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{timestamp}_{model_name}.md"

    backbone  = model_cfg["backbone"]
    pretrained= model_cfg.get("pretrained", False)
    patch     = model_cfg.get("patch_size", model_cfg["training"]["batch_size"])
    tcfg      = model_cfg["training"]
    epochs    = tcfg["epochs"]
    batch     = tcfg["batch_size"]
    lr        = tcfg["learning_rate"]
    wd        = tcfg["weight_decay"]
    optimizer = tcfg["optimizer"]

    lines: List[str] = [
        f"# Training Report",
        f"- Date: {timestamp}",
        "",
        "## Configuration",
        f"- Backbone: **{backbone}**  (pretrained={pretrained})",
        f"- Patch size: {patch}  |  Batch size: {batch}",
        f"- Epochs: {epochs}  |  LR: {lr:.1e}  |  WD: {wd:.1e}  |  Optimizer: {optimizer}",
        "",
        "## Classes",
    ]
    for cls, idx in sorted(class_to_idx.items(), key=lambda x: x[1]):
        lines.append(f"- {cls}: {idx}")
    lines += [
        "",
        "## Dataset",
        f"- Train samples: {num_train}",
        f"- Val   samples: {num_val}",
        "",
        "## History",
        "|Epoch|Train Loss|Train Acc|Val Loss|Val Acc|",
        "|:---:|:--------:|:-------:|:------:|:-----:|",
    ]
    for h in history:
        lines.append(
            f"| {h['epoch']} "
            f"| {h['train_loss']:.4f} "
            f"| {h['train_acc']:.3f} "
            f"| {h['val_loss']:.4f} "
            f"| {h['val_acc']:.3f} |"
        )
    lines += [
        "",
        "## Best Model",
        f"- Epoch: **{best_epoch}**",
        f"- Val Accuracy: **{best_acc:.3f}**",
        ""
    ]

    report_path.write_text("\n".join(lines))
    print(f"Training report saved → {report_path}")
