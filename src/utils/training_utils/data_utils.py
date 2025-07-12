from pathlib import Path
from typing import Any, Dict, Tuple, List
import torch
import joblib
import tarfile, math, webdataset as wds
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import LabelEncoder

# — Constants —
KNOWN_LABELS = {"ccRCC", "pRCC", "CHROMO", "ONCO", "not_tumor"}

# --- Feature/classifier I/O ---

def save_features(features: torch.Tensor, keys: List[str], path: Path) -> None:
    """Save a dict {"features": Tensor[N,D], "keys": List[str]} to a .pt file."""
    torch.save({"features": features, "keys": keys}, path)


def load_features(path: Path) -> Tuple[torch.Tensor, List[str]]:
    """Load features and keys from a .pt file produced by save_features."""
    data = torch.load(path)
    return data["features"], data["keys"]


def save_classifier(clf: Any, label_encoder: Any, path: Path) -> None:
    """Serialize classifier and label encoder into a joblib file."""
    joblib.dump({"model": clf, "le": label_encoder}, path)


def load_classifier(path: Path) -> Tuple[Any, Any]:
    """Load classifier and label encoder from joblib file."""
    data = joblib.load(path)
    model = data.get("model")
    le    = data.get("le") or data.get("label_encoder")
    if le is None:
        raise KeyError(f"No label encoder found in {path} (keys: {list(data.keys())})")
    return model, le

# --- Label extraction utility ---

def extract_labels_from_keys(keys: List[str], le: LabelEncoder) -> np.ndarray:
    """
    Extract integer-encoded labels from a list of feature keys using a LabelEncoder.
    Each key is expected to start with one of the classes known by the LabelEncoder,
    followed by an underscore.

    Args:
        keys (List[str]): List of key strings, e.g. "ccRCC_...".
        le   (LabelEncoder): Fitted LabelEncoder with known classes.

    Returns:
        np.ndarray: Array of integer-encoded labels corresponding to each key.

    Raises:
        ValueError: If any key does not start with a known class prefix.
    """
    classes = list(le.classes_)
    extracted: List[str] = []
    for key in keys:
        matched = False
        for cls in classes:
            if key.startswith(f"{cls}_"):
                extracted.append(cls)
                matched = True
                break
        if not matched:
            raise ValueError(
                f"Cannot extract label from key: {key!r}. Expected prefixes: {classes}"
            )
    return le.transform(extracted)

# — Transforms & parsing —

def default_transforms(patch_size: int, augment: bool) -> T.Compose:
    ops = []
    if augment:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    return T.Compose(ops + [
        T.Resize(patch_size),
        T.CenterCrop(patch_size),
        T.ToTensor()
    ])


def parse_label_from_filename(filename: str) -> str:
    """Estrae la label dal nome file (senza __key__), es. ccRCC_HP01_123_456.jpg"""
    parts = Path(filename).stem.split("_")
    label = "not_tumor" if parts[:2] == ["not", "tumor"] else parts[0]
    if label not in KNOWN_LABELS:
        raise ValueError(
            f"[parse_label_from_filename] Label '{label}' estratta da '{filename}' "
            f"non è tra quelle attese: {sorted(KNOWN_LABELS)}.\n"
            "Controlla il formato del filename e la lista delle classi."
        )
    return label


class PreprocSample:
    """Applica transform e parsing della label partendo dal nome del file."""
    def __init__(self, class_to_idx: Dict[str,int], patch_size: int, augment: bool):
        self.tfms = default_transforms(patch_size, augment)
        self.cls2idx = class_to_idx

    def __call__(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """
        Apply transforms and parse label from the WebDataset sample.
        Expects:
          - sample["jpg"]: the decoded PIL image
          - sample["__key__"]: the original filename without extension
        """
        # 1) extract PIL image
        img = sample.get("jpg") or next(
            (v for v in sample.values() if isinstance(v, Image.Image)),
            None
        )
        if img is None:
            raise ValueError("No image found in sample; expected key 'jpg' or a PIL.Image")
        # 2) reconstruct filename from __key__
        key = sample.get("__key__")
        if key is None:
            raise ValueError("No '__key__' found in sample to reconstruct filename")
        filename = f"{key}.jpg"
        # 3) parse and validate label
        label_str = parse_label_from_filename(filename)
        if label_str not in self.cls2idx:
            raise ValueError(f"Label '{label_str}' not in class->idx mapping")
        label_idx = self.cls2idx[label_str]
        # 4) apply transforms
        img = img.convert("RGB")
        tensor = self.tfms(img)
        return tensor, label_idx

# — Class discovery & counting —

def discover_classes(train_dir: Path) -> Dict[str, int]:
    found = set()
    for tar in train_dir.glob("*.tar"):
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".jpg"):
                    lbl = parse_label_from_filename(Path(m.name).name)
                    if lbl in KNOWN_LABELS:
                        found.add(lbl)
        if found == KNOWN_LABELS:
            break
    return {c: i for i, c in enumerate(sorted(found))}


def count_samples(shard: Path) -> int:
    total = 0
    for tar in shard.parent.glob("*.tar"):
        with tarfile.open(tar) as tf:
            total += sum(1 for m in tf.getmembers() if m.isfile() and m.name.endswith(".jpg"))
    return total

# — DataLoader factory —
def build_loader(shards: str, class_to_idx: Dict[str,int],
                 patch_size: int, batch_size: int,
                 device, augment: bool) -> DataLoader:
    """
    Generic loader for supervised or patch-based tasks.
    Assumes images have informative filenames (e.g. ccRCC_...jpg).
    """
    ds = (
        wds.WebDataset(shards)
        .decode("pil")
        .map(PreprocSample(class_to_idx, patch_size, augment))
    )
    use_cuda = device.type == "cuda"
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda
    )