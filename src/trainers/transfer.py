from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder

from utils.training_utils import (
    create_backbone,
    BaseTrainer,
    choose_device,
    discover_classes,
    count_samples,
    build_loader,
    save_checkpoint,
    register_trainer,
)

#######################################################################
# Trainer
#######################################################################

@register_trainer("transfer")
class TransferTrainer(BaseTrainer):
    """Fine‑tune only the classifier head on a frozen pretrained backbone."""

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        super().__init__(model_cfg, data_cfg)
        self._init_training_params(model_cfg)
        self.device = choose_device()
        self._init_paths(data_cfg)
        self._init_dataloaders()
        self._init_model_and_optimizer(model_cfg)
        self._init_tracking()

    def _init_training_params(self, model_cfg: Dict[str, Any]):
        t = model_cfg["training"]
        self.epochs        = int(t["epochs"])
        self.batch_size    = int(t["batch_size"])
        self.lr            = float(t["learning_rate"])
        self.weight_decay  = float(t["weight_decay"])
        self.optimizer_name= t.get("optimizer", "adam").lower()
        self.patch_size    = int(model_cfg.get("patch_size", 224))

    def _init_paths(self, data_cfg: Dict[str, Any]):
        self.train_pattern = Path(data_cfg["train"])
        self.val_pattern   = Path(data_cfg["val"])
        data_root          = self.train_pattern.parent
        self.output_root   = data_root.parent / "transfer"
        self.ckpt_dir      = self.output_root / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.class_to_idx  = discover_classes(data_root)
        self.label_encoder = LabelEncoder().fit(list(self.class_to_idx.keys()))
        self.num_train     = count_samples(self.train_pattern)
        self.num_val       = count_samples(self.val_pattern)
        self.batches_train = math.ceil(self.num_train / self.batch_size)
        self.batches_val   = math.ceil(self.num_val   / self.batch_size)

    def _init_dataloaders(self):
        loader_kwargs = dict(patch_size=self.patch_size, batch_size=self.batch_size, device=self.device)
        self.train_loader = build_loader(str(self.train_pattern), self.class_to_idx, augment=True, **loader_kwargs)
        self.val_loader   = build_loader(str(self.val_pattern),   self.class_to_idx, augment=False, **loader_kwargs)

    def _init_model_and_optimizer(self, model_cfg: Dict[str, Any]):
        self.model = create_backbone(model_cfg["backbone"].lower(), num_classes=len(self.class_to_idx), pretrained=True).to(self.device)

        # Freeze the entire backbone
        for p in self.model.parameters():
            p.requires_grad = False
        # Unfreeze classifier (fc) layer
        for p in self.model.fc.parameters():
            p.requires_grad = True

        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.fc.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        self.criterion = nn.CrossEntropyLoss()

    def _init_tracking(self):
        self.history: List[Dict[str, float]] = []
        self.best_epoch = 0
        self.best_val_acc = 0.0

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_step(self, imgs: torch.Tensor, labels: torch.Tensor):
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        self.model.train(); self.optimizer.zero_grad()
        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        loss.backward(); self.optimizer.step()
        preds   = outputs.argmax(dim=1)
        correct = int((preds == labels).sum().item())
        return outputs, float(loss.item()), correct, labels.size(0)

    def validate_epoch(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss    = self.criterion(outputs, labels)
                running_loss += float(loss.item()) * labels.size(0)
                correct      += int((outputs.argmax(1) == labels).sum().item())
                total        += labels.size(0)
        return running_loss / total, correct / total

    def post_epoch(self, epoch: int, val_acc: float):
        self.history.append({"epoch": epoch, "val_acc": val_acc})
        if val_acc > self.best_val_acc:
            self.best_val_acc, self.best_epoch = val_acc, epoch
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=self.model,
                optimizer=self.optimizer,
                class_to_idx=self.class_to_idx,
                model_cfg=self.model_cfg,
                data_cfg=self.data_cfg,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_val_acc