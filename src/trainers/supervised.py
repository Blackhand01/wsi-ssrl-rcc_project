import logging
import math
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
import webdataset as wds
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from launch_training import BaseTrainer, register_trainer  # noqa: E402

LOGGER = logging.getLogger(__name__)
KNOWN_LABELS = {"ccRCC", "pRCC", "CHROMO", "ONCO", "not_tumor"}


def _default_transforms(patch_size: int, augment: bool) -> T.Compose:
    """
    Build default image transformations: resize, crop, to-tensor, with optional augmentation.
    """
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


def _parse_label_from_key(key: str) -> str:
    """
    Parse class label from WebDataset key (filename stem).
    """
    stem = Path(key).stem
    parts = stem.split("_")
    if parts[0] == "not" and len(parts) >= 3 and parts[1] == "tumor":
        return "not_tumor"
    return parts[0]


class _PreprocSample:
    """
    Apply transforms to incoming samples: extracts key and PIL.Image,
    computes label index and returns (tensor, label_idx).
    """
    def __init__(self, class_to_idx: Dict[str, int], patch_size: int, augment: bool):
        self.class_to_idx = class_to_idx
        self.tfms = _default_transforms(patch_size, augment)

    def __call__(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        key = sample["__key__"]
        img = sample.get("jpg", None)
        if img is None or not isinstance(img, Image.Image):
            img = next((v for v in sample.values() if isinstance(v, Image.Image)), None)
        if img is None:
            raise RuntimeError(f"No image found in sample {key}")

        lbl_str = _parse_label_from_key(key)
        lbl = self.class_to_idx.get(lbl_str)
        if lbl is None:
            raise RuntimeError(f"Unrecognized label '{lbl_str}' for sample {key}")

        return self.tfms(img), lbl


@register_trainer("supervised")
class SupervisedTrainer(BaseTrainer):
    """
    Trainer for supervised classification of RCC subtypes using WebDataset shards.
    """
    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        super().__init__(model_cfg, data_cfg)

        # Hyperparameters
        t = model_cfg["training"]
        self.epochs = int(t["epochs"])
        self.batch_size = int(t["batch_size"])
        self.lr = float(t["learning_rate"])
        self.weight_decay = float(t["weight_decay"])
        self.optimizer_name = str(t["optimizer"]).lower()
        self.patch_size = int(model_cfg.get("patch_size", 224))
        self.pretrained = bool(model_cfg.get("pretrained", False))
        self.backbone_name = str(model_cfg["backbone"]).lower()

        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        LOGGER.info("Using device: %s", self.device)

        # Data patterns
        self.train_pattern = data_cfg["train"]
        self.val_pattern = data_cfg["val"]
        train_dir = Path(self.train_pattern).parent
        self.dataset_root = train_dir.parent

        # Discover classes
        self.class_to_idx = self._discover_classes(train_dir)
        LOGGER.info("Discovered classes: %s", self.class_to_idx)

        # Count samples and batches
        self.num_train_samples = self._count_samples(self.train_pattern)
        self.num_val_samples = self._count_samples(self.val_pattern)
        self.num_train_batches = math.ceil(self.num_train_samples / self.batch_size)
        self.num_val_batches = math.ceil(self.num_val_samples / self.batch_size)

        # Create data loaders
        self.train_loader = self._make_loader(self.train_pattern, augment=True)
        self.val_loader = self._make_loader(self.val_pattern, augment=False)

        # Build model and optimizer
        self.model = self._create_model(len(self.class_to_idx)).to(self.device)
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer '{self.optimizer_name}'")
        self.criterion = nn.CrossEntropyLoss()

    def train(self) -> None:
        """
        Run training loop with improved terminal visualization and save documentation.
        """
        epoch_bar = tqdm(
            range(1, self.epochs + 1),
            desc="Epoch",
            unit="epoch",
            position=0,
            leave=True,
            dynamic_ncols=True,
            ncols=100,
            colour="green",
        )
        best_acc = 0.0

        for epoch in epoch_bar:
            tr_loss, tr_acc = self._run_epoch(epoch, training=True)
            v_loss, v_acc = self._run_epoch(epoch, training=False)

            epoch_bar.set_postfix(
                train_loss=f"{tr_loss:.4f}",
                train_acc=f"{tr_acc:.3f}",
                val_loss=f"{v_loss:.4f}",
                val_acc=f"{v_acc:.3f}"
            )

            if v_acc > best_acc:
                best_acc = v_acc
                self._save_checkpoint(epoch, best=True)

        LOGGER.info("🏅 Best validation accuracy: %.3f", best_acc)
        self._save_model_doc(best_acc)

    def _discover_classes(self, train_dir: Path) -> Dict[str, int]:
        """
        Scan shards to discover available class labels.
        """
        shard_paths = list(train_dir.glob("*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shards found in {train_dir}")
        classes = set()
        for tar in shard_paths:
            with tarfile.open(tar) as t:
                for m in t:
                    if m.isfile() and m.name.lower().endswith(".jpg"):
                        lbl = _parse_label_from_key(Path(m.name).stem)
                        if lbl in KNOWN_LABELS:
                            classes.add(lbl)
            if len(classes) == len(KNOWN_LABELS):
                break
        if not classes:
            raise RuntimeError("No valid classes found in training shards")
        return {c: i for i, c in enumerate(sorted(classes))}

    def _count_samples(self, shards_pattern: str) -> int:
        """
        Count number of .jpg samples across all shard files.
        """
        path = Path(shards_pattern)
        tar_files = list(path.parent.glob("*.tar"))
        total = 0
        for tar in tar_files:
            with tarfile.open(tar) as tf:
                total += sum(1 for m in tf if m.isfile() and m.name.lower().endswith(".jpg"))
        return total

    def _make_loader(self, shards_pattern: str, augment: bool) -> DataLoader:
        """
        Create DataLoader from WebDataset shards with preprocessing.
        """
        if not list(Path(shards_pattern).parent.glob("*.tar")):
            raise FileNotFoundError(f"No .tar shards found for pattern '{shards_pattern}'")

        ds = (
            wds.WebDataset(
                shards_pattern,
                handler=wds.warn_and_continue,
                shardshuffle=1000 if augment else 0,
                empty_check=False,
            )
            .decode("pil")
        )
        if augment:
            ds = ds.shuffle(1000)

        preproc = _PreprocSample(self.class_to_idx, self.patch_size, augment)
        ds = ds.map(preproc)

        is_cuda = (self.device.type == "cuda")
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4 if is_cuda else 0,
            pin_memory=is_cuda,
        )

    def _create_model(self, num_classes: int) -> nn.Module:
        """
        Instantiate model backbone and replace final layer.
        """
        if self.backbone_name == "resnet50":
            m = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V2 if self.pretrained else None
            )
        elif self.backbone_name == "resnet18":
            m = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
            )
        else:
            raise ValueError(f"Unsupported backbone '{self.backbone_name}'")
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    def _run_epoch(self, epoch: int, *, training: bool) -> Tuple[float, float]:
        """
        Execute one epoch of training or validation.
        """
        loader = self.train_loader if training else self.val_loader
        phase = "Train" if training else "Val"
        total_batches = self.num_train_batches if training else self.num_val_batches

        batch_bar = tqdm(
            loader,
            desc=f"{phase} E{epoch}/{self.epochs}",
            total=total_batches,
            unit="batch",
            position=1,
            leave=True,
            dynamic_ncols=True,
            ncols=100,
            colour="blue",
        )

        self.model.train(training)
        total = correct = 0
        running_loss = 0.0

        for imgs, labels in batch_bar:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            if training:
                self.optimizer.zero_grad()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, labels)
            if training:
                loss.backward()
                self.optimizer.step()

            bs = labels.size(0)
            total += bs
            running_loss += loss.item() * bs
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            avg_loss = running_loss / total
            avg_acc = correct / total
            batch_bar.set_postfix(
                avg_loss=f"{avg_loss:.4f}",
                avg_acc=f"{avg_acc:.3f}"
            )

        if total == 0:
            split = "train" if training else "val"
            raise RuntimeError(f"No samples processed for split {split}: check your data")

        return running_loss / total, correct / total

    def _save_checkpoint(self, epoch: int, *, best: bool = False) -> None:
        """
        Save model checkpoint to disk.
        """
        ckpt_dir = self.dataset_root / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        name = "supervised_best.pt" if best else f"supervised_epoch{epoch:03d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "class_to_idx": self.class_to_idx,
            },
            ckpt_dir / name,
        )
        LOGGER.info("💾 Saved checkpoint: %s", name)

    def _save_model_doc(self, best_acc: float) -> None:
        """
        Generate a markdown documentation file summarizing model config and results.
        """
        ckpt_dir = self.dataset_root / "checkpoints"
        doc_path = ckpt_dir / "model_doc.md"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = [
            "# Model Documentation",
            f"- Date: {now}",
            f"- Backbone: {self.backbone_name}",
            f"- Pretrained: {self.pretrained}",
            f"- Patch size: {self.patch_size}",
            f"- Epochs: {self.epochs}",
            f"- Batch size: {self.batch_size}",
            f"- Learning rate: {self.lr}",
            f"- Weight decay: {self.weight_decay}",
            f"- Optimizer: {self.optimizer_name}",
            f"- Classes: {self.class_to_idx}",
            f"- Best validation accuracy: {best_acc:.3f}",
        ]
        doc_path.write_text("\n".join(content))
        LOGGER.info("📄 Saved model documentation: %s", doc_path)
