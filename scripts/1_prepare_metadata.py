#!/usr/bin/env python3
"""
1_prepare_metadata.py

Prepare a unified metadata table for RCC dataset:
  - loads mapping JSONs (ccRCC, pRCC, CHROMO, ONCO)
  - loads dataset statistics
  - checks that mapping sizes match stats
  - builds one row per WSI + annotations/ROIs
  - writes out a CSV file for downstream training pipeline
  - uses relative source directories based on config/base_dir
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import yaml
import pandas as pd

def load_json(path: Path):
    return json.loads(path.read_text())

def resolve_config(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    base_dir = Path(cfg["base_dir"]).expanduser()
    def R(p: str) -> Path:
        return Path(str(p).replace("${base_dir}", str(base_dir))).expanduser()
    paths = {k: R(v) for k, v in cfg["paths"].items()}
    return base_dir, paths

def check_equal(name: str, actual: int, expected: int):
    if actual != expected:
        raise ValueError(f"Stat mismatch for {name}: mapping has {actual}, stats expect {expected}")

def rel(path: Path, base: Path):
    return str(path.relative_to(base).as_posix())

def main(cfg_file, mapdir, out_csv):
    base_dir, paths = resolve_config(Path(cfg_file))

    # load mappings and stats
    cc_map     = load_json(Path(mapdir)/"ccRCC_mapping.json")
    pr_map     = load_json(Path(mapdir)/"pRCC_mapping.json")
    chromo_map = load_json(Path(mapdir)/"CHROMO_patient_mapping.json")
    onco_map   = load_json(Path(mapdir)/"ONCO_patient_mapping.json")
    stats      = load_json(Path(mapdir)/"rcc_dataset_stats.json")

    # consistency checks
    check_equal("ccRCC", len(cc_map), stats["ccRCC"]["n_wsis"])
    check_equal("pRCC", len(pr_map), stats["pRCC"]["n_wsis"])
    actual_ch = sum(len(v["wsi_files"]) for v in chromo_map.values())
    check_equal("CHROMO", actual_ch, stats["CHROMO"]["n_wsis"])
    actual_on = sum(len(v["wsi_files"]) for v in onco_map.values())
    check_equal("ONCO", actual_on, stats["ONCO"]["n_wsis"])

    rows = []

    # ccRCC: distinguo tra ccRCC vs pre/ccRCC
    for wsi, xmls in cc_map.items():
        # determina la cartella di origine
        wsi_path_std = paths["ccrcc_wsi"]/wsi
        wsi_path_pre = paths["pre_ccrcc_wsi"]/wsi
        if wsi_path_std.exists():
            src = rel(paths["ccrcc_wsi"], base_dir)
        elif wsi_path_pre.exists():
            src = rel(paths["pre_ccrcc_wsi"], base_dir)
        else:
            src = ""  # non dovrebbe succedere

        rows.append({
            "subtype":         "ccRCC",
            "patient":         wsi.split(".")[0],
            "wsi_filename":    wsi,
            "annotation_xml":  ";".join(xmls),
            "num_annotations": len(xmls),
            "roi_files":       "",
            "num_rois":        0,
            "source_dir":      src
        })

    # pRCC: distinguo tra pRCC vs pre/pRCC
    for wsi, xmls in pr_map.items():
        wsi_path_std = paths["prcc_wsi"]/wsi
        wsi_path_pre = paths["pre_prcc_wsi"]/wsi
        if wsi_path_std.exists():
            src = rel(paths["prcc_wsi"], base_dir)
        elif wsi_path_pre.exists():
            src = rel(paths["pre_prcc_wsi"], base_dir)
        else:
            src = ""

        rows.append({
            "subtype":         "pRCC",
            "patient":         wsi.split(".")[0],
            "wsi_filename":    wsi,
            "annotation_xml":  ";".join(xmls),
            "num_annotations": len(xmls),
            "roi_files":       "",
            "num_rois":        0,
            "source_dir":      src
        })

    # CHROMO: tutte da paths["chromo_wsi"]
    for patient, maps in chromo_map.items():
        roi_list = maps["roi_files"]
        for wsi in maps["wsi_files"]:
            rows.append({
                "subtype":         "CHROMO",
                "patient":         patient,
                "wsi_filename":    wsi,
                "annotation_xml":  "",
                "num_annotations": 0,
                "roi_files":       ";".join(roi_list),
                "num_rois":        len(roi_list),
                "source_dir":      rel(paths["chromo_wsi"], base_dir)
            })

    # ONCO: tutte da paths["onco_wsi"]
    for patient, maps in onco_map.items():
        roi_list = maps["roi_files"]
        for wsi in maps["wsi_files"]:
            rows.append({
                "subtype":         "ONCO",
                "patient":         patient,
                "wsi_filename":    wsi,
                "annotation_xml":  "",
                "num_annotations": 0,
                "roi_files":       ";".join(roi_list),
                "num_rois":        len(roi_list),
                "source_dir":      rel(paths["onco_wsi"], base_dir)
            })

    # salva il CSV
    df = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"INFO: ✅ Consolidated metadata saved to {out_csv}")
    print(f"      Total entries: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build unified metadata CSV for RCC dataset"
    )
    parser.add_argument("--cfg",    required=False, default="config/preprocessing.yaml",
                        help="Path to YAML configuration")
    parser.add_argument("--mapdir", required=False, default="data/processed/mapping",
                        help="Directory with mapping JSONs and stats")
    parser.add_argument("--out",    required=False, default="data/processed/metadata.csv",
                        help="Output metadata CSV file")
    args = parser.parse_args()
    main(args.cfg, args.mapdir, args.out)
