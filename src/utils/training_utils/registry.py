"""
Registry + classe base comune per *tutti* i trainer.

• `TRAINER_REGISTRY`       – dict «nome → class»
• `@register_trainer`      – decorator per l’autoregistrazione
• `BaseTrainer`            – API minima uniforme + stub utili

Il design è intenzionalmente leggero: le sottoclassi implementano solo ciò
che serve davvero (train_step, validate_epoch, ecc.) mentre la logica
comune/di fallback è già qui.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Type

import torch

# --------------------------------------------------------------------------- #
# 1)           REGISTRY                                                       #
# --------------------------------------------------------------------------- #
TRAINER_REGISTRY: Dict[str, Type] = {}

def register_trainer(name: str):
    """Decorator: registra la classe trainer nel dizionario globale."""
    def _decorator(cls: Type):
        TRAINER_REGISTRY[name] = cls
        cls.NAME = name          # comodo averlo anche come attributo
        return cls
    return _decorator


# --------------------------------------------------------------------------- #
# 2)           BASE TRAINER                                                   #
# --------------------------------------------------------------------------- #
class BaseTrainer:
    """
    Interfaccia (quasi) astratta, con molte *no-op* già pronte:

    • __init__               – salva cfg, da usare o estendere
    • train_step / validate  – da implementare nelle sottoclassi
    • get_resume_model_and_optimizer – fallback, torna (self.model, self.optimizer)
    • build_loader           – usa factory standard da data_utils.py
    """

    # Etichetta di default: le sottoclassi possono ridefinirla (es. "ssl" / "sl")
    TYPE: str = "generic"

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        self.model_cfg: Dict[str, Any] = model_cfg
        self.data_cfg: Dict[str, Any] = data_cfg
        self.ckpt_dir: Path | None = None  # impostata dal notebook

    # ----------------------------- Train Step (required)
    def train_step(self, *batch) -> Tuple[float, int]:
        raise NotImplementedError

    # ----------------------------- Validation (optional)
    def validate_epoch(self) -> Tuple[float, float]:
        raise NotImplementedError(f"{self.__class__.__name__}.validate_epoch() non implementato.")

    # ----------------------------- Early stopping / Checkpoint saving
    def post_epoch(self, epoch: int, metric: float):
        raise NotImplementedError(f"{self.__class__.__name__}.post_epoch() non implementato.")

    # ----------------------------- Resume
    def get_resume_model_and_optimizer(self):
        model = getattr(self, "model", None) or getattr(self, "encoder", None)
        optim = getattr(self, "optimizer", None)
        if model is None or optim is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.get_resume_model_and_optimizer() non implementato correttamente."
            )
        return model, optim

    def summary(self) -> Tuple[int, float]:
        raise NotImplementedError(f"{self.__class__.__name__}.summary() non implementato.")

    # ----------------------------- Default loader logic
    def build_loader(self, split: str):
        """
        Default loader: supervised/SL/SSL patch-based pipeline.
        Usa la factory build_loader da data_utils.py, con parametri dal config.
        """
        from utils.training_utils.data_utils import build_loader, discover_classes

        is_ssl = self.model_cfg.get("type") == "ssl"
        patch_size = int(self.model_cfg.get("patch_size", 224))
        batch_size = int(self.model_cfg["training"]["batch_size"])
        augment = bool(self.model_cfg.get("augmentation", False))
        device = getattr(self, "device", None)

        if not is_ssl:
            class_to_idx = discover_classes(Path(self.data_cfg["train"]).parent)
        else:
            class_to_idx = None

        return build_loader(
            shards=self.data_cfg[split],
            class_to_idx=class_to_idx,
            patch_size=patch_size,
            batch_size=batch_size,
            device=device,
            augment=augment
        )


# --------------------------------------------------------------------------- #
# 3)           UTILITY (facilita l’import * in __init__ dei trainer)         #
# --------------------------------------------------------------------------- #
__all__ = [
    "TRAINER_REGISTRY",
    "register_trainer",
    "BaseTrainer",
]
