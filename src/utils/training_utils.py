# src/utils/training_utils.py
# -*- coding: utf-8 -*-
"""
training_utils
--------------
Funzioni riutilizzabili per tutti i Trainer:
- gestione device
- trasformazioni e preprocessing
- WebDataset loader
- checkpointing e report markdown
"""

from __future__ import annotations
import logging
import tarfile
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import torch
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from PIL import Image

import torchvision.models as models

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
KNOWN_LABELS = {"ccRCC", "pRCC", "CHROMO", "ONCO", "not_tumor"}

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def choose_device() -> torch.device:
    """Sceglie tra CUDA, MPS (Apple) o CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Trasformazioni e parsing etichette
# ---------------------------------------------------------------------------
def default_transforms(patch_size: int, augment: bool) -> T.Compose:
    """Costruisce la pipeline di trasformazioni: resize, crop, tensor, + augment."""
    tfms: List[Any] = []
    if augment:
        tfms += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomChoice([
                T.RandomRotation((0, 0)),
                T.RandomRotation((90, 90)),
                T.RandomRotation((180, 180)),
                T.RandomRotation((270, 270)),
            ]),
        ]
    tfms += [
        T.Resize(patch_size),
        T.CenterCrop(patch_size),
        T.ToTensor(),
    ]
    return T.Compose(tfms)

def parse_label_from_key(key: str) -> str:
    """Estrae l’etichetta dal nome del file (__key__ di WebDataset)."""
    stem = Path(key).stem
    parts = stem.split("_")
    if parts[:2] == ["not", "tumor"]:
        return "not_tumor"
    return parts[0]

class PreprocSample:
    """Transform callable per WebDataset.map: da sample dict a (tensor, label_idx)."""
    def __init__(self,
                 class_to_idx: Dict[str,int],
                 patch_size: int,
                 augment: bool):
        self.class_to_idx = class_to_idx
        self.tfms = default_transforms(patch_size, augment)

    def __call__(self, sample: Dict[str,Any]) -> Tuple[torch.Tensor,int]:
        key = sample.get("__key__", "<unknown>")
        # cerco il campo 'jpg' oppure la prima PIL.Image
        img = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)),
            None
        )
        if not isinstance(img, Image.Image):
            raise RuntimeError(f"No image found in sample '{key}'")

        lbl_str = parse_label_from_key(key)
        if lbl_str not in self.class_to_idx:
            raise RuntimeError(f"Unknown label '{lbl_str}' in sample '{key}'")

        return self.tfms(img), self.class_to_idx[lbl_str]

# ---------------------------------------------------------------------------
# Classes discovery & sample counting
# ---------------------------------------------------------------------------
def discover_classes(train_dir: Path) -> Dict[str,int]:
    """
    Scansiona tutti i .tar in train_dir e raccoglie le etichette valide.
    Ritorna dict label->indice.
    """
    shards = list(train_dir.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {train_dir}")
    found: set[str] = set()
    for tar in shards:
        with tarfile.open(tar) as tf:
            for m in tf:
                if m.isfile() and m.name.lower().endswith(".jpg"):
                    lbl = parse_label_from_key(Path(m.name).stem)
                    if lbl in KNOWN_LABELS:
                        found.add(lbl)
        if found == KNOWN_LABELS:
            break
    if not found:
        raise RuntimeError("No valid labels found in training shards")
    return {c: i for i, c in enumerate(sorted(found))}

def count_samples(pattern: Path) -> int:
    """
    Conta i file .jpg in tutti i tar dello stesso folder di 'pattern'.
    """
    folder = pattern.parent
    total = 0
    for tar in folder.glob("*.tar"):
        with tarfile.open(tar) as tf:
            total += sum(
                1
                for m in tf.getmembers()
                if m.isfile() and m.name.lower().endswith(".jpg")
            )
    return total

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
def build_loader(shards_pattern: str,
                 class_to_idx: Dict[str,int],
                 patch_size: int,
                 batch_size: int,
                 device: torch.device,
                 augment: bool) -> DataLoader:
    """
    Crea un DataLoader su WebDataset:
    - shards_pattern: es. "data/.../train/patches-*.tar"
    - class_to_idx: mappa label->indice
    - patch_size, batch_size, device
    - augment: se True applica shuffle+augment
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

# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------
def create_backbone(name: str, num_classes: int, pretrained: bool) -> torch.nn.Module:
    """
    Istanzia un backbone ResNet e sostituisce l'ultimo FC con num_classes.
    """
    if name == "resnet50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        )
    elif name == "resnet18":
        m = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
    else:
        raise ValueError(f"Unsupported backbone '{name}'")
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m

# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(ckpt_dir: Path,
                    prefix: str,
                    *,
                    epoch: int,
                    best: bool,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    class_to_idx: Dict[str,int],
                    model_cfg: Dict[str,Any],
                    data_cfg: Dict[str,Any]) -> None:
    """
    Salva i pesi e lo stato ottimizzatore in ckpt_dir con nome:
    {prefix}_{best|epochXXX}.pt
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = "best" if best else f"epoch{epoch:03d}"
    name = f"{prefix}_{tag}.pt"
    path = ckpt_dir / name
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "model_cfg": model_cfg,
        "data_cfg": data_cfg,
    }, path)
    LOGGER.info("💾 Checkpoint saved → %s", path)

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def save_model_doc(history: List[Dict[str, float]],
                   best_epoch: int,
                   best_acc: float,
                   model_cfg: Dict[str,Any],
                   class_to_idx: Dict[str,int],
                   num_train: int,
                   num_val: int,
                   out_dir: Path) -> None:
    """
    Scrive un report markdown completo in out_dir/model_report.md
    con configurazione, cronologia e miglior epoca.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "model_report.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # estraggo da model_cfg
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
        f"- Date: {now}",
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
    LOGGER.info("📄 Report written → %s", report_path)
