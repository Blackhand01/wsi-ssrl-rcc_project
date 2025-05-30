#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_training.py

Wrapper utility that launches training for one (or all) vision models defined
in a YAML configuration file. Trainers are discovered dynamically from the
trainers/ package via a global registry.

Examples
--------
python src/launch_training.py -c config/training.yaml -m supervised
python src/launch_training.py --config config/training.yaml --model moco_v2 --verbose
python src/launch_training.py -c config/training.yaml -m all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import yaml

# ---------------------------------------------------------------------------
# Ensure this module is reachable as "launch_training" (for trainer imports)
# ---------------------------------------------------------------------------
sys.modules.setdefault("launch_training", sys.modules[__name__])

# ---------------------------------------------------------------------------
# TRAINER REGISTRY
# ---------------------------------------------------------------------------
TRAINER_REGISTRY: Dict[str, type] = {}


def register_trainer(name: str):
    """
    Decorator to register a Trainer class in the global registry.

    Parameters
    ----------
    name : str
        Key that must match an entry in the 'models' section of the YAML file.
    """

    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# BASE TRAINER — must be defined *before* importing trainers.*
# ---------------------------------------------------------------------------
class BaseTrainer:
    """
    Minimal common interface inherited by all concrete trainers.

    Each subclass should override ``train()`` and usually ``__init__``.
    """

    def __init__(self, model_cfg: Dict[str, Any], data_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.logger = logging.getLogger(self.__class__.__name__)

    def train(self):
        """Run the training loop (to be implemented in subclasses)."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# IMPORT ALL TRAINERS (so that @register_trainer decorators run)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent            # .../src
PROJECT_ROOT = SCRIPT_DIR.parent                        # wsi-ssrl-rcc_project
sys.path.insert(0, str(PROJECT_ROOT))                  # ensure import path

# pylint: disable=unused-import, wrong-import-position
import trainers.simclr      # registers 'simclr'
import trainers.moco_v2     # registers 'moco_v2'
import trainers.rotation    # registers 'rotation'
import trainers.jigsaw      # registers 'jigsaw'
import trainers.supervised  # registers 'supervised'
import trainers.transfer    # registers 'transfer'
# pylint: enable=unused-import, wrong-import-position

# ---------------------------------------------------------------------------
# CORE LAUNCHER
# ---------------------------------------------------------------------------
def launch_training(cfg: Dict[str, Any], model_to_run: str):
    """
    Launch training for a single model or for all models in the configuration.

    Parameters
    ----------
    cfg : Dict[str, Any]
        Parsed YAML configuration (must contain 'data' and 'models').
    model_to_run : str
        Name of the model to train (must match a key in cfg['models'])
        or 'all' to train every available model.
    """
    data_cfg = cfg.get("data", {})
    models_cfg = cfg.get("models", {})

    if not models_cfg:
        raise ValueError("No 'models' section found in configuration.")

    # decide which models to iterate on
    if model_to_run.lower() == "all":
        target_models = models_cfg.items()
    else:
        if model_to_run not in models_cfg:
            raise KeyError(
                f"Model '{model_to_run}' not found in YAML. "
                f"Available models: {list(models_cfg.keys())}"
            )
        target_models = [(model_to_run, models_cfg[model_to_run])]

    for model_name, model_cfg in target_models:
        if model_name not in TRAINER_REGISTRY:
            raise KeyError(
                f"No trainer registered for model '{model_name}'. "
                f"Registered trainers: {list(TRAINER_REGISTRY.keys())}"
            )

        trainer_cls = TRAINER_REGISTRY[model_name]
        trainer = trainer_cls(model_cfg, data_cfg)

        logging.info("🚀 Launching training for model: %s", model_name)
        trainer.train()
        logging.info("✅ Completed training for model: %s", model_name)


# ---------------------------------------------------------------------------
# CLI PARSING
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch training for a specific model (or all models) "
        "defined in a YAML configuration file."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/training.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Name of the model to train as listed in the YAML ('supervised', "
        "'simclr', ...), or 'all' to train every model.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging."
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# MAIN ENTRYPOINT
# ---------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)

    # configure root logger
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logging.error("Configuration file not found: %s", cfg_path)
        sys.exit(1)

    logging.info("📄 Loading configuration from %s", cfg_path)
    with cfg_path.open("r") as f:
        cfg_dict = yaml.safe_load(f)

    launch_training(cfg_dict, args.model)


if __name__ == "__main__":
    main()
