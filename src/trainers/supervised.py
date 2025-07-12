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
    BaseTrainer,
    create_backbone,
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
    Supervised Trainer for RCC subtype classification (WebDataset).
    """

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]) -> None:
        super().__init__(model_cfg, data_cfg)
        self._init_training_params(model_cfg)
        self.device = choose_device()
        self._init_paths(data_cfg)
        self._init_dataloaders()
        self._init_model_and_optimizer(model_cfg)
        self._init_tracking()

    def _init_training_params(self, model_cfg: Dict[str, Any]) -> None:
        t = model_cfg["training"]
        self.epochs        = int(t["epochs"])
        self.batch_size    = int(t["batch_size"])
        self.lr            = float(t["learning_rate"])
        self.weight_decay  = float(t["weight_decay"])
        self.optimizer_name= t.get("optimizer", "adam").lower()
        self.patch_size    = int(model_cfg.get("patch_size", 224))

    def _init_paths(self, data_cfg: Dict[str, Any]) -> None:
        self.train_pattern = Path(data_cfg["train"])
        self.val_pattern   = Path(data_cfg["val"])
        data_root = self.train_pattern.parent
        self.class_to_idx = discover_classes(data_root)
        self.label_encoder = LabelEncoder().fit(list(self.class_to_idx.keys()))
        self.num_train = count_samples(self.train_pattern)
        self.num_val   = count_samples(self.val_pattern)
        self.batches_train = math.ceil(self.num_train / self.batch_size)
        self.batches_val   = math.ceil(self.num_val   / self.batch_size)

    def _init_dataloaders(self) -> None:
        # Corretto: usa chiamata posizionale come definito in data_utils.build_loader
        self.train_loader = build_loader(
            str(self.train_pattern),   # shards
            self.class_to_idx,         # class_to_idx
            self.patch_size,           # patch_size
            self.batch_size,           # batch_size
            self.device,               # device
            True                       # augment (solo train)
        )
        self.val_loader = build_loader(
            str(self.val_pattern),     # shards
            self.class_to_idx,         # class_to_idx
            self.patch_size,           # patch_size
            self.batch_size,           # batch_size
            self.device,               # device
            False                      # no augment su val
        )

    def _init_model_and_optimizer(self, model_cfg: Dict[str, Any]) -> None:
        self.model = create_backbone(
            model_cfg["backbone"].lower(),
            num_classes=len(self.class_to_idx),
            pretrained=model_cfg.get("pretrained", False),
        ).to(self.device)

        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:  # sgd
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )

        self.criterion = nn.CrossEntropyLoss()

    def _init_tracking(self) -> None:
        self.history      : List[Dict[str, float]] = []
        self.best_epoch   = 0
        self.best_val_acc = 0.0

    def train_step(
        self,
        imgs: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float, int, int]:
        """
        Single training step.
        Returns: outputs, loss_value, correct_count, batch_size
        """
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(imgs)
        loss    = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        preds   = outputs.argmax(dim=1)
        correct = int((preds == labels).sum().item())
        bs      = labels.size(0)
        return outputs, float(loss.item()), correct, bs

    def validate_epoch(self) -> Tuple[float, float]:
        """
        Full-validation loop.
        Returns: avg_loss, avg_accuracy
        """
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs = self.model(imgs)
                loss    = self.criterion(outputs, labels)
                bs      = labels.size(0)

                running_loss += float(loss.item()) * bs
                correct      += int((outputs.argmax(1) == labels).sum().item())
                total        += bs

        return running_loss / total, correct / total

    def post_epoch(self, epoch: int, val_acc: float) -> None:
        """
        After each epoch: checkpoint if improved, update history.
        """
        self.history.append({"epoch": epoch, "val_acc": val_acc})
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                ckpt_dir=self.ckpt_dir,
                prefix=self.__class__.__name__,
                epoch=epoch,
                best=True,
                model=self.model,
                optimizer=self.optimizer,
                metadata={
                    "class_to_idx": self.class_to_idx,
                    "model_cfg":    self.model_cfg,
                    "data_cfg":     self.data_cfg,
                },
            )

    def summary(self) -> Tuple[int, float]:
        """
        Return best epoch and corresponding validation accuracy.
        """
        return self.best_epoch, self.best_val_acc

    def get_resume_model_and_optimizer(self) -> Tuple[Any, Any]:
        """
        Interface required by launch_training to resume from checkpoint.
        """
        return self.model, self.optimizer

    def extract_features_to(
        self,
        output_path: Path | str,
        split: str = "train"
    ) -> None:
        """
        Stub per coerenza di interfaccia: non estraiamo feature per SL.
        """
        raise NotImplementedError("SupervisedTrainer does not support feature extraction")
