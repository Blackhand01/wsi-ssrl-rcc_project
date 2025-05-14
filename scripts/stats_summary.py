import json
from collections import defaultdict
from pathlib import Path

def load_mapping(path: Path, structured=False):
    with open(path) as f:
        data = json.load(f)

    files = []
    if structured:  # for ONCO and CHROMO
        for patient, content in data.items():
            for f in content.get("wsi_files", []):
                files.append((f, "wsi", patient))
            for f in content.get("roi_files", []):
                files.append((f, "roi", patient))
    else:  # for ccRCC and pRCC
        for wsi, xmls in data.items():
            files.append((wsi, "wsi", len(xmls)))
            for xml in xmls:
                files.append((xml, "roi", wsi))
    return data, files


def compute_basic_stats(mapping: dict, structured=False):
    stats = {}
    if structured:
        stats["n_patients"] = len(mapping)
        stats["n_roi_files"] = sum(len(d.get("roi_files", [])) for d in mapping.values())
        stats["n_wsi_files"] = sum(len(d.get("wsi_files", [])) for d in mapping.values())
        stats["mean_rois_per_patient"] = round(stats["n_roi_files"] / stats["n_patients"], 2)
        stats["mean_rois_per_wsi"] = round(stats["n_roi_files"] / stats["n_wsi_files"], 2)
    else:
        stats["n_wsis"] = len(mapping)
        stats["n_annotated_wsis"] = sum(1 for v in mapping.values() if v)
        stats["n_unannotated_wsis"] = stats["n_wsis"] - stats["n_annotated_wsis"]
        stats["n_roi_files"] = sum(len(v) for v in mapping.values())
        stats["mean_rois_per_wsi"] = round(stats["n_roi_files"] / stats["n_wsis"], 2)
    return stats


def compute_file_stats(mapping: dict, subtype: str, structured=False):
    file_stats = []
    if structured:
        for patient, files in mapping.items():
            for f in files.get("wsi_files", []):
                file_stats.append({"filename": f, "type": "wsi", "patient": patient, "subtype": subtype})
            for f in files.get("roi_files", []):
                file_stats.append({"filename": f, "type": "roi", "patient": patient, "subtype": subtype})
    else:
        for wsi, xmls in mapping.items():
            file_stats.append({
                "filename": wsi,
                "type": "wsi",
                "subtype": subtype,
                "n_annotations": len(xmls),
                "annotated": bool(xmls)
            })
            for xml in xmls:
                file_stats.append({
                    "filename": xml,
                    "type": "roi",
                    "subtype": subtype,
                    "parent_wsi": wsi
                })
    return file_stats


def main():
    BASE = Path("data/organized")
    OUT = Path("data/stats/rcc_dataset_stats.json")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    files_all = []
    global_stats = {}

    result = {}

    for subtype, fname, is_structured in [
        ("ccRCC", "ccRCC_mapping.json", False),
        ("pRCC", "pRCC_mapping.json", False),
        ("ONCO", "ONCO_patient_mapping.json", True),
        ("CHROMO", "CHROMO_patient_mapping.json", True)
    ]:
        mapping, file_entries = load_mapping(BASE / fname, structured=is_structured)
        stats = compute_basic_stats(mapping, structured=is_structured)
        file_stats = compute_file_stats(mapping, subtype, structured=is_structured)
        result[subtype] = {
            "summary": stats,
            "files": file_stats
        }
        files_all.extend(file_stats)

    # Statistiche cumulative
    result["ALL"] = {
        "summary": {
            "n_total_files": len(files_all),
            "n_wsis": sum(1 for f in files_all if f["type"] == "wsi"),
            "n_rois": sum(1 for f in files_all if f["type"] == "roi"),
            "n_annotated_wsis": sum(1 for f in files_all if f.get("annotated") is True),
            "n_unannotated_wsis": sum(1 for f in files_all if f.get("annotated") is False),
        },
        "files": files_all
    }

    # Salva
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Statistiche salvate in: {OUT}")


if __name__ == "__main__":
    main()
