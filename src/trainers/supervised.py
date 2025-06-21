from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """
    Trainer for supervised RCC subtype classification using WebDataset.

    Methods
    -------
    __init__(model_cfg, data_cfg)
        Initializes the trainer, model, optimizer, dataloaders, etc.
    train_step(imgs, labels)
        Performs a single training step (forward, backward, optimizer step).
    validate_epoch()
        Runs validation over the entire validation set and returns loss and accuracy.
    post_epoch(epoch, val_metric)
        Handles post-epoch logic: checkpointing and history update.
    summary()
        Returns the best epoch and best validation accuracy.
    """

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        data_cfg: Dict[str, Any]
    ) -> None:
        super().__init__(model_cfg, data_cfg)
        self._init_training_params(model_cfg)
        self.device = self._init_device()
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
        self.optimizer_name = tcfg.get("optimizer", "adam").lower()
        self.patch_size = int(model_cfg.get("patch_size", 224))

    def _init_device(self) -> torch.device:
        return choose_device()

    def _init_paths(self, data_cfg: Dict[str, Any]) -> None:
        train_pattern = Path(data_cfg["train"])
        val_pattern = Path(data_cfg["val"])
        self.train_pattern = train_pattern
        self.val_pattern = val_pattern
        data_root = train_pattern.parent
        self.output_root = data_root.parent
        self.class_to_idx = discover_classes(data_root)
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.class_to_idx.keys()))
        self.num_train = count_samples(train_pattern)
        self.num_val = count_samples(val_pattern)
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
        self.model = create_backbone(
            model_cfg["backbone"].lower(),
            num_classes=len(self.class_to_idx),
            pretrained=model_cfg.get("pretrained", False),
        ).to(self.device)
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=self.lr,
                momentum=0.9, weight_decay=self.weight_decay
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
        """
        Single training step: forward, backward, update.
        Returns: outputs, loss_value, correct_count, batch_size
        """
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
        """
        Runs validation over entire val set.
        Returns: avg_loss, avg_accuracy
        """
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
        """
        Post-epoch: save checkpoint if val accuracy improved, update history.
        """
        self.history.append({"epoch": epoch, "val_acc": val_metric})
        if val_metric > self.best_val_acc:
            self.best_val_acc = val_metric
            self.best_epoch = epoch
            save_checkpoint(
                ckpt_dir=self.output_root / "supervised" / "checkpoints",
                prefix=self.__class__.__name__, epoch=epoch, best=True,
                model=self.model, optimizer=self.optimizer,
                class_to_idx=self.class_to_idx,
                model_cfg=self.model_cfg, data_cfg=self.data_cfg
            )

    def summary(self) -> Tuple[int, float]:
        """
        Returns: best_epoch, best_val_acc
        """
        return self.best_epoch, self.best_val_acc
