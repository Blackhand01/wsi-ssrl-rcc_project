#!/usr/bin/env python3
"""
generate_splits_by_patient.py

Reads metadata.csv, extracts unique patient IDs and randomly assigns
each patient to train, val or test according to given proportions.
Outputs a JSON file mapping patient_id -> split.
"""

import argparse
import json
import random
import pandas as pd
from pathlib import Path

def generate_splits(patient_ids, fractions, seed=None):
    """
    patient_ids: list of unique IDs
    fractions: dict e.g. {'train': 0.8, 'val': 0.1, 'test': 0.1}
    seed: int for reproducibility
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("Fractions must sum to 1.")

    # Shuffle reproducibly
    rng = random.Random(seed)
    ids = patient_ids.copy()
    rng.shuffle(ids)

    n = len(ids)
    splits = {}
    # compute boundaries
    train_end = int(fractions['train'] * n)
    val_end   = train_end + int(fractions['val'] * n)

    for idx, pid in enumerate(ids):
        if idx < train_end:
            splits[pid] = 'train'
        elif idx < val_end:
            splits[pid] = 'val'
        else:
            splits[pid] = 'test'
    return splits

def main(metadata_csv: Path, out_json: Path,
         train_frac: float, val_frac: float, test_frac: float,
         seed: int):
    # Load metadata
    df = pd.read_csv(metadata_csv)
    patients = sorted(df['patient_id'].unique().tolist())

    # Generate splits
    fractions = {'train': train_frac, 'val': val_frac, 'test': test_frac}
    splits = generate_splits(patients, fractions, seed)

    # Save JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(splits, indent=2))
    print(f"✅ Saved splits for {len(patients)} patients to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate patient-wise train/val/test splits JSON"
    )
    parser.add_argument(
        "--metadata", required=True,
        help="Path to metadata CSV file (must contain 'patient_id' column)"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSON path for splits_by_patient.json"
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8,
        help="Fraction of patients for training (default: 0.8)"
    )
    parser.add_argument(
        "--val_frac", type=float, default=0.1,
        help="Fraction of patients for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.1,
        help="Fraction of patients for test (default: 0.1)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Validate sum of fractions
    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-6:
        parser.error(f"train_frac+val_frac+test_frac must sum to 1. Got {total:.3f}")

    main(
        metadata_csv=Path(args.metadata),
        out_json=Path(args.out),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )
