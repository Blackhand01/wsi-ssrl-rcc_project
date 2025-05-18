#!/usr/bin/env python3
# =============================================================================
# Build mapping JSON files for the RCC WSI dataset
#   • ccRCC_mapping.json            (WSI .scn  → list[.xml])
#   • pRCC_mapping.json
#   • CHROMO_patient_mapping.json   (patient → {wsi_files, roi_files})
#   • ONCO_patient_mapping.json
# It also saves rcc_dataset_stats.json and prints any source files
# that were not mapped (slides lacking XML or patient association).
# =============================================================================

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

# --------------------------------------------------------------------------- #
# Helper functions                                                            #
# --------------------------------------------------------------------------- #

def mapping_scn_xml(xml_dirs, wsi_dirs):
    """Return dict{wsi_filename: list[xml_filename]} for (pre)ccRCC / (pre)pRCC."""
    xml_index = defaultdict(list)

    # Index every XML by its *full* stem (incl. “-1”, “-2”, …)
    for xml_dir in xml_dirs:
        for xml in Path(xml_dir).glob("*.xml"):
            xml_index[xml.stem].append(xml.name)

    mapping = {}
    for wsi_dir in wsi_dirs:
        for scn in Path(wsi_dir).glob("*.scn"):
            mapping[scn.name] = xml_index.get(scn.stem, [])
    return mapping


def parse_correspondence(xlsx_path: Path):
    """Excel sheet ⇒ dict{id_number (str) → patient_id (str)}."""
    df = pd.read_excel(xlsx_path)
    id2pat = {}
    for _, row in df.iterrows():
        rng = str(row["ID"])
        pat = str(row["PATIENT"])
        if "-" in rng:           #  “9-14” range notation
            a, b = map(int, rng.split("-"))
            for k in range(a, b + 1):
                id2pat[str(k)] = pat
        else:
            id2pat[rng] = pat
    return id2pat


def build_patient_roi_map(ann_dir: Path, xlsx_path: Path):
    """Return dict{patient → list[roi_filename]} for .svs/.tif ROIs."""
    id2pat = parse_correspondence(xlsx_path)
    pat2roi = defaultdict(list)

    for fp in ann_dir.glob("*.*"):
        if fp.suffix.lower() not in (".svs", ".tif"):
            continue
        m = re.match(r"(\d+)", fp.stem)  # ROI filenames start with numeric ID
        if m:
            patient = id2pat.get(m.group(1))
            if patient:
                pat2roi[patient].append(fp.name)
    return pat2roi


def build_patient_wsi_map(wsi_dir: Path, patient_ids):
    """Return dict{patient → list[Path]} for original .svs WSI slides."""
    pat2wsi = defaultdict(list)
    for slide in Path(wsi_dir).glob("*.svs"):
        for pid in patient_ids:
            if pid in slide.stem:
                pat2wsi[pid].append(slide)  # keep Path, useful for reporting
    return pat2wsi

# --------------------------------------------------------------------------- #
# Statistics                                                                  #
# --------------------------------------------------------------------------- #

def compute_stats(cc_map, pr_map, chromo_map, onco_map):
    stats = {}

    # ccRCC
    n_cc_wsis = len(cc_map)
    n_cc_xmls = sum(len(v) for v in cc_map.values())
    stats["ccRCC"] = dict(n_wsis=n_cc_wsis,
                          n_xml_annotations=n_cc_xmls,
                          n_patients=len({f.split(".")[0] for f in cc_map}))

    # pRCC
    n_pr_wsis = len(pr_map)
    n_pr_xmls = sum(len(v) for v in pr_map.values())
    stats["pRCC"] = dict(n_wsis=n_pr_wsis,
                         n_xml_annotations=n_pr_xmls,
                         n_patients=len({f.split(".")[0] for f in pr_map}))

    # CHROMO
    n_ch_pat = len(chromo_map)
    n_ch_wsis = sum(len(v["wsi_files"]) for v in chromo_map.values())
    n_ch_rois = sum(len(v["roi_files"]) for v in chromo_map.values())
    stats["CHROMO"] = dict(n_patients=n_ch_pat,
                           n_wsis=n_ch_wsis,
                           n_roi_files=n_ch_rois)

    # ONCO
    n_on_pat = len(onco_map)
    n_on_wsis = sum(len(v["wsi_files"]) for v in onco_map.values())
    n_on_rois = sum(len(v["roi_files"]) for v in onco_map.values())
    stats["ONCO"] = dict(n_patients=n_on_pat,
                         n_wsis=n_on_wsis,
                         n_roi_files=n_on_rois)

    # totals
    stats["ALL"] = dict(total_wsis=n_cc_wsis + n_pr_wsis + n_ch_wsis + n_on_wsis,
                        total_xmls=n_cc_xmls + n_pr_xmls,
                        total_roi_files=n_ch_rois + n_on_rois,
                        total_patients=stats["ccRCC"]["n_patients"]
                                       + stats["pRCC"]["n_patients"]
                                       + n_ch_pat + n_on_pat)
    return stats

# --------------------------------------------------------------------------- #
# Reporting utilities                                                         #
# --------------------------------------------------------------------------- #

def report_unmapped_scn(cc_map, wsi_dirs):
    """Print absolute path of any .scn slide with zero XML annotations."""
    missing = [fname for fname, xmls in cc_map.items() if not xmls]
    if not missing:
        return
    print("⚠️  ccRCC slides without any XML:")
    for fname in missing:
        for d in wsi_dirs:
            p = Path(d) / fname
            if p.exists():
                print("   ", p.resolve())

def report_unmapped_svs(label, wsi_map, wsi_dir):
    """Print absolute path of .svs slides not linked to any patient."""
    mapped = {path for lst in wsi_map.values() for path in lst}
    missing = [slide for slide in Path(wsi_dir).glob("*.svs") if slide not in mapped]
    if not missing:
        return
    print(f"⚠️  {label} WSI without mapping:")
    for slide in missing:
        print("   ", slide.resolve())

# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main(cfg_path, out_dir):
    # Load YAML config
    cfg = yaml.safe_load(open(cfg_path))
    base = Path(cfg["base_dir"]).expanduser()

    def R(path):  # quick resolver helper
        return Path(str(path).replace("${base_dir}", str(base))).expanduser()

    p = cfg["paths"]  # alias
    ccrcc_wsi, pre_ccrcc_wsi = R(p["ccrcc_wsi"]), R(p["pre_ccrcc_wsi"])
    prcc_wsi,  pre_prcc_wsi  = R(p["prcc_wsi"]),  R(p["pre_prcc_wsi"])
    ccrcc_xml, pre_ccrcc_xml = R(p["ccrcc_xml"]), R(p["pre_ccrcc_xml"])
    prcc_xml,  pre_prcc_xml  = R(p["prcc_xml"]),  R(p["pre_prcc_xml"])
    chromo_wsi, onco_wsi     = R(p["chromo_wsi"]), R(p["onco_wsi"])
    chromo_ann, onco_ann     = R(p["chromo_ann"]), R(p["onco_ann"])

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ---------- ccRCC & pRCC (flat mapping) ----------------------------------
    cc_map = mapping_scn_xml([ccrcc_xml, pre_ccrcc_xml],
                             [ccrcc_wsi, pre_ccrcc_wsi])
    pr_map = mapping_scn_xml([prcc_xml, pre_prcc_xml],
                             [prcc_wsi, pre_prcc_wsi])

    json.dump(cc_map, open(out_path / "ccRCC_mapping.json", "w"), indent=2)
    json.dump(pr_map, open(out_path / "pRCC_mapping.json", "w"), indent=2)

    # ---------- CHROMO (structured mapping) ----------------------------------
    chromo_roi   = build_patient_roi_map(chromo_ann, chromo_ann /
                                         "CHROMO_patients_correspondence.xlsx")
    chromo_wsi_map = build_patient_wsi_map(chromo_wsi, chromo_roi.keys())

    chromo_map = {pat: {"roi_files": chromo_roi[pat],
                        "wsi_files": [x.name for x in chromo_wsi_map[pat]]}
                  for pat in chromo_roi}
    json.dump(chromo_map, open(out_path / "CHROMO_patient_mapping.json", "w"),
              indent=2)

    # ---------- ONCO (structured mapping) ------------------------------------
    onco_roi   = build_patient_roi_map(onco_ann, onco_ann /
                                       "ONCO_patients_correspondence.xlsx")
    onco_wsi_map = build_patient_wsi_map(onco_wsi, onco_roi.keys())

    onco_map = {pat: {"roi_files": onco_roi[pat],
                      "wsi_files": [x.name for x in onco_wsi_map[pat]]}
                for pat in onco_roi}
    json.dump(onco_map, open(out_path / "ONCO_patient_mapping.json", "w"),
              indent=2)

    # ---------- Statistics ----------------------------------------------------
    stats = compute_stats(cc_map, pr_map, chromo_map, onco_map)
    json.dump(stats, open(out_path / "rcc_dataset_stats.json", "w"), indent=2)

    # ---------- Unmapped reports ---------------------------------------------
    report_unmapped_scn(cc_map, [ccrcc_wsi, pre_ccrcc_wsi])
    report_unmapped_svs("CHROMO", chromo_wsi_map, chromo_wsi)
    report_unmapped_svs("ONCO",   onco_wsi_map,   onco_wsi)

    # ---------- Summary -------------------------------------------------------
    print("\n✅ Mapping JSON files saved in", out_path.resolve())
    print("\n📊 Dataset statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

# --------------------------------------------------------------------------- #
# Entry-point                                                                 #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate RCC mapping JSON files, stats, and unmapped report"
    )
    parser.add_argument("--cfg", required=False, default="config/preprocessing.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--out", required=False, default="data/processed/mapping",
                        help="Output directory for JSON files")
    args = parser.parse_args()
    main(args.cfg, args.out)
