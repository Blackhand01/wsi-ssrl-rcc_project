import re
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict


def parse_range_to_ids(id_str: str) -> list[int]:
    """Parsa range tipo '1-3' in [1, 2, 3]"""
    if "-" in id_str:
        start, end = map(int, id_str.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(id_str)]


def mapping_scn_xml(wsi_dirs, xml_dirs):
    """Crea mapping da .scn a .xml per ccRCC e pRCC"""
    mapping = defaultdict(list)
    xml_index = defaultdict(list)

    for xml_dir in xml_dirs:
        for xml_file in Path(xml_dir).glob("*.xml"):
            key = xml_file.stem.split("-")[0]
            xml_index[key].append(xml_file.name)

    for wsi_dir in wsi_dirs:
        for scn_file in Path(wsi_dir).glob("*.scn"):
            base = scn_file.stem
            mapping[scn_file.name] = xml_index.get(base, [])

    return mapping


def map_roi_to_patient(annotations_dir: Path, excel_path: Path) -> dict[str, list[str]]:
    """Crea mapping patient → ROI file (.svs e .tif in Annotations folder)"""
    df = pd.read_excel(excel_path)
    patient_to_rois = defaultdict(list)
    id_to_patient = {}

    for _, row in df.iterrows():
        ids = parse_range_to_ids(str(row["ID"]))
        patient = str(row["PATIENT"])
        for id_ in ids:
            id_to_patient[str(id_)] = patient

    for roi_file in annotations_dir.glob("*"):
        if roi_file.suffix.lower() not in [".svs", ".tif"]:
            continue
        match = re.match(r"(\d+)", roi_file.stem)
        if match:
            roi_id = match.group(1)
            patient = id_to_patient.get(roi_id)
            if patient:
                patient_to_rois[patient].append(roi_file.name)
            else:
                print(f"⚠️  ROI ID {roi_id} (file: {roi_file.name}) non mappato a nessun paziente.")
    return patient_to_rois


def map_wsi_to_patient(wsi_dir: Path, patient_ids: list[str]) -> dict[str, list[str]]:
    """Crea mapping patient → WSI file (in ONCOCYTOMA / CHROMO)"""
    patient_to_wsi = defaultdict(list)
    for wsi_file in wsi_dir.glob("*.svs"):
        for pid in patient_ids:
            if pid in wsi_file.stem:
                patient_to_wsi[pid].append(wsi_file.name)
    return patient_to_wsi


def generate_onco_chromo_mapping(wsi_dir: Path, annotations_dir: Path, excel_path: Path) -> dict:
    """Genera mapping strutturato: patient → {roi_files, wsi_files}"""
    roi_mapping = map_roi_to_patient(annotations_dir, excel_path)
    patient_ids = list(roi_mapping.keys())
    wsi_mapping = map_wsi_to_patient(wsi_dir, patient_ids)

    combined = {
        patient: {
            "roi_files": roi_mapping.get(patient, []),
            "wsi_files": wsi_mapping.get(patient, [])
        }
        for patient in patient_ids
    }
    return combined


def save_json(mapping: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Saved: {path}")


if __name__ == "__main__":
    BASE = Path("~/Library/CloudStorage/GoogleDrive-stefano2001roy@gmail.com/Il mio Drive/Colab_Notebooks/RCC_WSIs").expanduser()
    OUT = Path("data/organized")

    # ccRCC
    cc_map = mapping_scn_xml(
        wsi_dirs=[BASE / "ccRCC", BASE / "pre" / "ccRCC"],
        xml_dirs=[BASE / "ccRCC" / "ccRCC_xml", BASE / "pre" / "ccRCC" / "pre_ccRCC_xml"]
    )
    save_json(cc_map, OUT / "ccRCC_mapping.json")

    # pRCC
    prcc_map = mapping_scn_xml(
        wsi_dirs=[BASE / "pRCC", BASE / "pre" / "pRCC"],
        xml_dirs=[BASE / "pRCC" / "pRCC_xml", BASE / "pre" / "pRCC" / "pre_pRCC_xml"]
    )
    save_json(prcc_map, OUT / "pRCC_mapping.json")

    # ONCO
    onco_map = generate_onco_chromo_mapping(
        wsi_dir=BASE / "ONCOCYTOMA",
        annotations_dir=BASE / "Annotations_onco",
        excel_path=BASE / "Annotations_onco" / "ONCO_patients_correspondence.xlsx"
    )
    save_json(onco_map, OUT / "ONCO_patient_mapping.json")

    # CHROMO
    chromo_map = generate_onco_chromo_mapping(
        wsi_dir=BASE / "CHROMO",
        annotations_dir=BASE / "Annotations_chromo",
        excel_path=BASE / "Annotations_chromo" / "CHROMO_patients_correspondence.xlsx"
    )
    save_json(chromo_map, OUT / "CHROMO_patient_mapping.json")
