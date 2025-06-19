from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.training_utils import (
    BaseTrainer,
    create_backbone,
    choose_device,
    discover_classes,
    count_samples,
    build_loader,
    save_checkpoint,
    register_trainer,
)

@register_trainer("transfer")
class TransferTrainer(BaseTrainer):
    """
    Trainer for transfer learning using a pretrained backbone.

    Freezes all backbone layers and trains only the final fully-connected layer.
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any]
    ) -> None:
        super().__init__(model_cfg, data_cfg)
        self._init_training_params(model_cfg)
        self.device = choose_device()
        self._init_paths(data_cfg)
        self._init_dataloaders()
        self._init_model_and_optimizer(model_cfg)
        self._init_tracking()

    def _init_training_params(self, model_cfg: Dict[str, Any]) -> None:
        tcfg = model_cfg["training"]
        self.epochs = int(tcfg["epochs"])
        self.batch_size = int(tcfg["batch_size"])
        self.lr = float(tcfg["learning_rate"])
        self.weight_decay = float(tcfg["weight_decay"])
        self.patch_size = int(model_cfg.get("patch_size", 224))

    def _init_paths(self, data_cfg: Dict[str, Any]) -> None:
        self.train_pattern = Path(data_cfg["train"])
        self.val_pattern = Path(data_cfg["val"])
        data_root = self.train_pattern.parent
        self.output_root = data_root.parent / "transfer"
        self.class_to_idx = discover_classes(data_root)
        self.num_train = count_samples(self.train_pattern)
        self.num_val = count_samples(self.val_pattern)
        self.batches_train = math.ceil(self.num_train / self.batch_size)
        self.batches_val = math.ceil(self.num_val / self.batch_size)

    def _init_dataloaders(self) -> None:
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

    def _init_model_and_optimizer(self, model_cfg: Dict[str, Any]) -> None:
        # Load pretrained backbone
        self.model = create_backbone(
            model_cfg["backbone"].lower(),
            num_classes=len(self.class_to_idx),
            pretrained=True,
        ).to(self.device)
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze final classifier layer
        for param in self.model.fc.parameters():
            param.requires_grad = True
        # Optimizer only on classifier parameters
        self.optimizer = optim.Adam(
            self.model.fc.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def _init_tracking(self) -> None:
        self.history: List[Dict[str, float]] = []
        self.best_epoch = 0
        self.best_val_acc = 0.0

    def train_step(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float, int, int]:
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(imgs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        preds = outputs.argmax(dim=1)
        correct = int((preds == labels).sum().item())
        bs = labels.size(0)
        return outputs, float(loss.item()), correct, bs

    def validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                bs = labels.size(0)
                running_loss += float(loss.item()) * bs
                correct += int((outputs.argmax(1) == labels).sum().item())
                total += bs
        return running_loss / total, correct / total

    def post_epoch(self, epoch: int, val_metric: float) -> None:
        self.history.append({"epoch": epoch, "val_acc": val_metric})
        if val_metric > self.best_val_acc:
            self.best_val_acc = val_metric
            self.best_epoch = epoch
            save_checkpoint(
                ckpt_dir=self.output_root / "checkpoints",
                prefix=self.__class__.__name__, epoch=epoch, best=True,
                model=self.model, optimizer=self.optimizer,
                class_to_idx=self.class_to_idx,
                model_cfg=self.model_cfg, data_cfg=self.data_cfg
            )

    def summary(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_val_acc
