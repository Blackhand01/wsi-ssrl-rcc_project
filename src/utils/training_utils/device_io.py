# src/utils/training_utils/device_io.py

from __future__ import annotations
import torch, joblib, json
from pathlib import Path
from typing import Optional, Any, Dict

# — Device selection —
def choose_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# — Checkpointing —
def save_checkpoint(ckpt_dir: Path, prefix: str, *,
                    epoch: int, best: bool,
                    model: torch.nn.Module,
                    optimizer: Any,
                    metadata: Dict[str, Any] | None = None,
                    **legacy_kwargs):
    """
    Salva un checkpoint PyTorch, coerente con la struttura esperimenti:

    es: simclr_bestepoch002_fold0.pt

    - `prefix` viene definito a monte come: f"{model_name}_fold{i}"
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    meta: Dict[str, Any] = dict(metadata or {})
    meta.update(legacy_kwargs)

    fname = f"{prefix}_bestepoch{epoch:03d}.pt"
    path  = ckpt_dir / fname

    for f in ckpt_dir.glob(f"{prefix}_bestepoch*.pt"):
        try: f.unlink()
        except OSError: pass

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            **meta,
        },
        path,
    )


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: Optional[Any] = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optim" in ckpt:
        optimizer.load_state_dict(ckpt["optim"])
    return ckpt


def get_latest_checkpoint(ckpt_dir: Path, ext: str = ".pt") -> Optional[Path]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        ckpt_dir.glob(f"*bestepoch*{ext}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


# — File / JSON / Joblib I/O —
def save_json(data: Dict, path: Path) -> Path:
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    return path

def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)

def save_joblib(obj: Any, path: Path) -> Path:
    path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(obj, path)
    return path
