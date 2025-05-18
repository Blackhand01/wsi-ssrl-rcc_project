# 📌 Mapping File Description and RCC Dataset Statistics

### 🧬 Introduction

The RCC (Renal Cell Carcinoma) dataset consists of histopathological Whole Slide Images (WSIs) divided into 4 neoplastic subtypes:

* **clear cell RCC** (`ccRCC`)
* **papillary RCC** (`pRCC`)
* **chromophobe RCC** (`CHROMO`)
* **oncocytoma** (`ONCO`)

WSIs are provided in `.scn` or `.svs` formats, while annotations can be:

* `.xml` files (for `ccRCC` and `pRCC`)
* ROI files (`.svs` or `.tif`) (for `ONCO` and `CHROMO`), stored in *Annotations\_* folders and mapped using Excel files.

---

### 📁 JSON Mapping Files

#### 🔹 Flat JSON – `ccRCC`, `pRCC`

| File                 | Description                                       |
| -------------------- | ------------------------------------------------- |
| `ccRCC_mapping.json` | Mapping `WSI (.scn)` → list of `.xml` annotations |
| `pRCC_mapping.json`  | Mapping `WSI (.scn)` → list of `.xml` annotations |

Format:

```json
{
  "HP19.754.A5.ccRCC.scn": ["HP19.754.A5.ccRCC.xml"],
  "HP19.7840.A1.ccRCC.scn": ["HP19.7840.A1.ccRCC.xml", "HP19.7840.A1.ccRCC-1.xml"]
}
```

> The folders `pre/ccRCC` and `pre/pRCC` are treated as equivalent and included in the mapping.

---

#### 🔸 Structured JSON – `ONCO`, `CHROMO`

| File                          | Description                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------ |
| `ONCO_patient_mapping.json`   | Mapping `patient ID` → `{ wsi_files, roi_files }` from `ONCOCYTOMA` and `Annotations_onco` |
| `CHROMO_patient_mapping.json` | Similar mapping from `CHROMO` and `Annotations_chromo`                                     |

Format:

```json
{
  "HP20.2506": {
    "roi_files": ["13.tif", "12.svs"],
    "wsi_files": ["HP20.2506_1339.svs", "HP20.2506_1342.svs"]
  }
}
```

> ROI files are cropped areas from original WSI slides, selected to represent tissue-homogeneous regions.

---

### 📊 Dataset Statistics

#### File: `rcc_dataset_stats.json`

This file includes:

* Per-subtype statistics
* File-level metadata for all WSI and ROI files
* Cumulative global statistics

Example:

```json
{
  "ccRCC": {
    "summary": {
      "n_wsis": 108,
      "n_annotated_wsis": 105,
      "n_roi_files": 115,
      "mean_rois_per_wsi": 1.06
    },
    "files": [
      {
        "filename": "HP19.754.A5.ccRCC.scn",
        "type": "wsi",
        "annotated": true,
        "n_annotations": 1
      },
      ...
    ]
  },
  ...
  "ALL": {
    "summary": {
      "n_total_files": 943,
      "n_wsis": 173,
      "n_rois": 173
    }
  }
}
```

---

### 🧱 Flat vs Structured Mapping

| Type       | Subtypes         | Primary Key           | Annotations                | Notes                                                    |
| ---------- | ---------------- | --------------------- | -------------------------- | -------------------------------------------------------- |
| Flat       | `ccRCC`, `pRCC`  | WSI filename (`.scn`) | list of `.xml` files       | Standalone XML annotations per WSI                       |
| Structured | `ONCO`, `CHROMO` | Patient ID            | `{ roi_files, wsi_files }` | ROI `.svs`/`.tif` mapped via Excel correspondence tables |

---
