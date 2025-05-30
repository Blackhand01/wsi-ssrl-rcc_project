# src/trainers/supervised.py
# -*- coding: utf-8 -*-
"""
SupervisedTrainer
-----------------
Addestramento CNN supervisionato sui WebDataset RCC.

Author: Stefano Roy Bisignano – refactor 2025-05
"""

from __future__ import annotations

import logging
import math
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.training_utils import (
    # registry & base
    BaseTrainer,
    register_trainer,
    # device & transforms
    choose_device,
    KNOWN_LABELS,
    default_transforms,
    parse_label_from_key,
    PreprocSample,
    # data indexing
    discover_classes,
    count_samples,
    build_loader,
    # model / checkpoint / report
    create_backbone,
    save_checkpoint,
    save_model_doc,
)

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Trainer                                                                     #
# --------------------------------------------------------------------------- #
@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """Trainer per classificazione supervisionata dei sottotipi RCC."""

    # --------------------------------------------------------------------- #
    # Costruzione                                                           #
    # --------------------------------------------------------------------- #
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        super().__init__(model_cfg, data_cfg)

        # ── Hparams ──────────────────────────────────────────────────────
        t = model_cfg["training"]
        self.epochs         = int(t["epochs"])
        self.batch_size     = int(t["batch_size"])
        self.lr             = float(t["learning_rate"])
        self.weight_decay   = float(t["weight_decay"])
        self.optimizer_name = t["optimizer"].lower()
        self.patch_size     = int(model_cfg.get("patch_size", 224))
        self.pretrained     = bool(model_cfg.get("pretrained", False))
        self.backbone_name  = model_cfg["backbone"].lower()

        # ── Device ───────────────────────────────────────────────────────
        self.device = choose_device()
        LOGGER.info("► Device selected: %s", self.device)

        # ── Dataset paths ────────────────────────────────────────────────
        self.train_pattern = Path(data_cfg["train"])
        self.val_pattern   = Path(data_cfg["val"])
        train_dir          = self.train_pattern.parent
        self.dataset_root  = train_dir.parent       # per checkpoints & reports

        # ── Scoperta classi ──────────────────────────────────────────────
        self.class_to_idx = self._discover_classes(train_dir)
        LOGGER.info("► Classes discovered: %s", self.class_to_idx)

        # ── Conteggi dataset ─────────────────────────────────────────────
        self.num_train_samples = count_samples(self.train_pattern)
        self.num_val_samples   = count_samples(self.val_pattern)
        self.num_train_batches = math.ceil(self.num_train_samples / self.batch_size)
        self.num_val_batches   = math.ceil(self.num_val_samples   / self.batch_size)

        LOGGER.info(
            "► Dataset size – train: %d (%d batches) | val: %d (%d batches)",
            self.num_train_samples, self.num_train_batches,
            self.num_val_samples,   self.num_val_batches,
        )

        # ── DataLoaders ──────────────────────────────────────────────────
        self.train_loader = build_loader(
            shards_pattern=str(self.train_pattern),
            class_to_idx=self.class_to_idx,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            augment=True,
        )
        self.val_loader = build_loader(
            shards_pattern=str(self.val_pattern),
            class_to_idx=self.class_to_idx,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            device=self.device,
            augment=False,
        )

        # ── Modello & ottimizzatore ──────────────────────────────────────
        self.model = self._create_model(len(self.class_to_idx)).to(self.device)
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer '{self.optimizer_name}'")

        self.criterion = nn.CrossEntropyLoss()

        # ── Tracking cronologia ──────────────────────────────────────────
        self.history: List[Dict[str, float]] = []
        self.best_epoch, self.best_epoch_acc = 0, 0.0

    # --------------------------------------------------------------------- #
    # Training loop                                                         #
    # --------------------------------------------------------------------- #
    def train(self) -> None:
        LOGGER.info(
            "► Training start → epochs=%d | bs=%d | lr=%.2e | wd=%.2e | opt=%s",
            self.epochs, self.batch_size, self.lr, self.weight_decay, self.optimizer_name,
        )

        epoch_bar = tqdm(
            range(1, self.epochs + 1),
            desc="Epoch",
            unit="epoch",
            position=0,
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            tr_loss, tr_acc = self._run_epoch(epoch, training=True)
            v_loss, v_acc   = self._run_epoch(epoch, training=False)

            # log & store
            self.history.append(
                dict(epoch=epoch,
                     train_loss=tr_loss, train_acc=tr_acc,
                     val_loss=v_loss,   val_acc=v_acc)
            )
            if v_acc > self.best_epoch_acc:
                self.best_epoch_acc, self.best_epoch = v_acc, epoch
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    class_to_idx=self.class_to_idx,
                    model_cfg=self.model_cfg,
                    data_cfg=self.data_cfg,
                    ckpt_dir=self.dataset_root / "checkpoints",
                    trainer_name=self.__class__.__name__,
                    best=True,
                )

            epoch_bar.set_postfix(
                tloss=f"{tr_loss:.4f}", tacc=f"{tr_acc:.3f}",
                vloss=f"{v_loss:.4f}", vacc=f"{v_acc:.3f}",
            )

        LOGGER.info("► Done. Best val_acc=%.3f @ epoch %d",
                    self.best_epoch_acc, self.best_epoch)

        # report markdown
        save_model_doc(
            history=self.history,
            best_epoch=self.best_epoch,
            best_acc=self.best_epoch_acc,
            model_cfg=self.model_cfg,
            class_to_idx=self.class_to_idx,
            num_train=self.num_train_samples,
            num_val=self.num_val_samples,
            out_dir=self.dataset_root / "checkpoints",
        )

    # ------------------------------------------------------------------ #
    # Helper methods specific to this trainer                            #
    # ------------------------------------------------------------------ #
    def _discover_classes(self, train_dir: Path) -> Dict[str, int]:
        """Scansiona i tar di training per ottenere l’insieme delle label."""
        shard_paths = list(train_dir.glob("*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No .tar shards found in {train_dir}")

        classes: set[str] = set()
        for tar in shard_paths:
            with tarfile.open(tar) as tf:
                for m in tf:
                    if m.isfile() and m.name.lower().endswith(".jpg"):
                        lbl = parse_label_from_key(Path(m.name).stem)
                        if lbl in KNOWN_LABELS:
                            classes.add(lbl)
            if classes == KNOWN_LABELS:
                break

        if not classes:
            raise RuntimeError("No valid classes discovered in training shards")

        return {c: i for i, c in enumerate(sorted(classes))}

    def _create_model(self, num_classes: int) -> nn.Module:
        """Istanzia il backbone scelto e sostituisce l’ultimo livello FC."""
        if self.backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
            )
        elif self.backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
            )
        else:
            raise ValueError(f"Unsupported backbone '{self.backbone_name}'")
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        return backbone

    # ------------------------------------------------------------------ #
    # Epoch routine                                                      #
    # ------------------------------------------------------------------ #
    def _run_epoch(self, epoch: int, *, training: bool) -> Tuple[float, float]:
        phase         = "Train" if training else "Val"
        loader        = self.train_loader if training else self.val_loader
        total_batches = self.num_train_batches if training else self.num_val_batches

        bar = tqdm(
            loader,
            desc=f"{phase} E{epoch}/{self.epochs}",
            total=total_batches,
            unit="batch",
            position=1,
            leave=False,
            dynamic_ncols=True,
        )

        self.model.train(training)
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in bar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)

            if training:
                self.optimizer.zero_grad()

            outputs = self.model(imgs)
            loss    = self.criterion(outputs, labels)

            if training:
                loss.backward()
                self.optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            correct      += (outputs.argmax(dim=1) == labels).sum().item()
            total        += bs

            bar.set_postfix(
                avg_loss=f"{running_loss/total:.4f}",
                avg_acc=f"{correct/total:.3f}",
            )

        if total == 0:
            raise RuntimeError(f"No samples processed in {phase} phase")

        return running_loss / total, correct / total
