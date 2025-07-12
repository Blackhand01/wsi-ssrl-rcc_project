# %% [markdown]
# ### 🧪 Extracting Patches into WebDataset Format 
# 
# This notebook extracts image patches from RCC WSIs based on coordinates provided in a `.parquet` file generated in Stage 2. Patches are saved in `WebDataset` format (`.tar`) for efficient downstream training. The output uses **flat, keyless JPEG entries** (no `__key__`), ensuring full sample accessibility during downstream loading.
# 
# The notebook performs the following steps:
# 
# 1. **Install required dependencies**, including `openslide`, `webdataset`, and compression tools.
# 2. **Load configuration from YAML**, resolving environment paths and parameters like `patch_size`, `random_seed`, and active stage.
# 3. **Locate and load the latest dataset folder**, automatically or manually, and read the `patch_df.parquet` file.
# 4. **Initialize WebDataset shard writers** for each split (`train`, `val`, `test`) with a configurable shard size.
# 5. **Iterate over unique sources (WSI and ROI)**, copy them locally, open them with `OpenSlide`, and extract patches as specified in `patch_df`.
# 6. **Filter out non-informative patches**, using grayscale standard deviation as a quality check.
# 7. **Save valid patches** into `.tar` shards as JPEGs, without keys or metadata — each image gets a flat incremental name (`000000.jpg`, `000001.jpg`, …).
# 8. **Log errors and summary**, writing a JSON error report if needed.
# 9. **Close shard writers and compute statistics**, including the number of patches per split and per subtype.
# 10. **Display visual summary** with sampled thumbnails for each class/split.
# 
# ---
# 
# ### 📄 `3-extract_patches.ipynb` – Documentation Table
# 
# | #  | **Section (Markdown Title)**  | **Main Content (Documented Classes/Functions)**                                                | **Output**                 |
# | -- | ----------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------- |
# | 1  | **Install & Setup**           | System and pip installs for OpenSlide and WebDataset                                           | system packages            |
# | 2  | **Load Configuration**        | `ConfigLoader`, `PathResolver` resolve paths and environment from `preprocessing.yaml`         | `cfg`, `PATHS`             |
# | 3  | **Select Dataset**            | Load latest or manually defined `dataset_{uuid}` folder and load `.parquet` file               | `patch_df`, `unique_wsis`  |
# | 4  | **Shard Initialization**      | Create empty `ShardWriter` objects for `train`, `val`, `test`, one per split                   | `writers` dictionary       |
# | 5  | **Informative Patch Check**   | `is_patch_informative`: filters out blank/black patches                                        | utility function           |
# | 6  | **Patch Extraction Loop**     | Iterate over sources, copy WSI to local SSD, open, extract and filter patches, write to `.tar` | WebDataset shards (`.tar`) |
# | 7  | **Error Handling**            | Logs missing/corrupted slides or failed reads into a `.json` error file                        | `extract_errors.json`      |
# | 8  | **Close Writers & Summary**   | Closes all writers, prints total saved patches and logs if errors occurred                     | summary stats              |
# | 9  | **Shard Statistics Overview** | Prints stats per shard: size, patch count, number of valid shards                              | per-split `.tar` summary   |
# | 10 | **Per-Class Stats**           | Stats for each split (`train`, `val`, `test`) with per-subtype patch counts                    | printed overview           |
# | 11 | **Patch Thumbnails Preview**  | Randomly loads JPEGs from shards and shows a grid of sampled patches per class/split           | matplotlib image grid      |
# 

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
import subprocess, sys, json, time, shutil, random, yaml, io, os, glob, tarfile
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# System packages
print("[SETUP] Installing OpenSlide system libs…")
subprocess.run("apt-get -qq update", shell=True, check=True)
subprocess.run("apt-get -qq install -y openslide-tools libopenslide-dev", shell=True, check=True)

# Python wheels
print("[SETUP] Installing Python wheels…")
subprocess.run(
    "pip install -qq --upgrade openslide-python openslide-bin "
    "webdataset tqdm matplotlib zarr 'numcodecs<0.8.0'",
    shell=True,
    check=True,
)

# Clean stray ~orch dirs (PyTorch leftovers if present)

# %%
# Cell 1 – Configuration & Paths (new YAML schema – no resumable logic)

YAML_PATH = Path("/content/drive/MyDrive/ColabNotebooks/wsi-ssrl-rcc_project/config/preprocessing.yaml")
if not YAML_PATH.exists():
    sys.exit(f"[FATAL] YAML config not found → {YAML_PATH}")

# ------------------------------------------------------------------#
# 1) ConfigLoader: resolves ${RESOLVED_BASE_DIR} and ${base.*}      #
# ------------------------------------------------------------------#
class ConfigLoader:
    def __init__(self, path: Path):
        raw = yaml.safe_load(path.read_text())
        self.base_root = self._select_env(raw['environment'])
        self.cfg = self._substitute(raw)

    def _select_env(self, env):
        colab, local = Path(env['colab']), Path(env['local'])
        if colab.exists(): return colab
        if local.exists(): return local
        sys.exit("[FATAL] No environment path found in YAML")

    def _substitute(self, cfg_raw):
        import copy
        cfg = copy.deepcopy(cfg_raw)
        # resolve base section
        base = {k: v.replace('${RESOLVED_BASE_DIR}', str(self.base_root))
                for k, v in cfg['base'].items()}
        cfg['base'] = base
        # placeholder map
        ph = {'${RESOLVED_BASE_DIR}': str(self.base_root),
              **{f'${{base.{k}}}': v for k, v in base.items()}}
        def repl(o):
            if isinstance(o, str):
                for k, v in ph.items(): o = o.replace(k, v)
            elif isinstance(o, dict):
                o = {k: repl(v) for k, v in o.items()}
            elif isinstance(o, list):
                o = [repl(v) for v in o]
            return o
        for sec in ['data_paths', 'stage_overrides', 'patching_defaults', 'split_by_patient']:
            cfg[sec] = repl(cfg[sec])
        return cfg

# ------------------------------------------------------------------#
# 2) PathResolver: maps data_paths → dataraw_root                   #
# ------------------------------------------------------------------#
class PathResolver:
    def __init__(self, cfg):
        b, dp = cfg['base'], cfg['data_paths']
        self.project_root  = Path(b['project_root'])
        self.raw_root      = Path(b['dataraw_root'])      # <- RAW slides
        self.metadata_root = Path(b['metadata_root'])
        self.patch_df_dir  = self.project_root / "data" / "processed"  # where the parquet will be saved

        def raw(p):  # convert path under data_root ⇒ dataraw_root
            return self.raw_root / Path(p).relative_to(Path(b['data_root']))

        # ccRCC / pRCC + pre
        self.ccrcc_wsi      = raw(dp['ccRCC']['wsi'])
        self.pre_ccrcc_wsi  = raw(dp['ccRCC']['pre']['wsi'])
        self.prcc_wsi       = raw(dp['pRCC']['wsi'])
        self.pre_prcc_wsi   = raw(dp['pRCC']['pre']['wsi'])
        # CHROMO / ONCO
        self.chromo_wsi     = raw(dp['CHROMO']['wsi'])
        self.onco_wsi       = raw(dp['ONCO']['wsi'])

# ------------------------------------------------------------------#
# 3) Initialization                                                 #
# ------------------------------------------------------------------#
CFG   = ConfigLoader(YAML_PATH).cfg
PATHS = PathResolver(CFG)

# Select stage (debug / training) from YAML
stage_key   = 'debug' if CFG['stage_overrides']['debug']['sampling']['enabled'] else 'training'
stage_cfg   = CFG['stage_overrides'][stage_key]

PATCH_SIZE  = stage_cfg['patching']['patch_size']
RANDOM_SEED = stage_cfg['patching']['random_seed']
SHARD_SIZE  = 5_000
random.seed(RANDOM_SEED)

print(f"[CFG] Base root       : {PATHS.project_root}")
print(f"[CFG] RAW slides root : {PATHS.raw_root}")
print(f"[CFG] Stage           : {stage_key}")
print(f"[CFG] Patch size      : {PATCH_SIZE}")

# %%
# Cell 2 – Locate the latest dataset_{uuid} and load its parquet
import os, glob, pandas as pd
from pathlib import Path
import sys

# --- Configuration ---
# Set to True to automatically select the most recently modified dataset directory.
# Set to False to manually specify the directory name below.
AUTO_SELECT_LATEST_DATASET = True  # Set to True or False

# If AUTO_SELECT_LATEST_DATASET is False, specify the directory name here.
MANUAL_DATASET_NAME = "dataset_9f30917e"
# --------------------

# 1) Processed dataset root folder
proc_dir = PATHS.project_root / "data" / "processed"

# 2) Select dataset directory based on configuration
if AUTO_SELECT_LATEST_DATASET:
    ds_dirs = [p for p in proc_dir.glob("dataset_*") if p.is_dir()]
    if not ds_dirs:
        sys.exit("[FATAL] No dataset_{uuid} directory found – run 2_generate_patch_df first")
    ds_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    dataset_dir = ds_dirs[0]
    print(f"[DATA] Automatically selected latest dataset directory: {dataset_dir.name}")
else:
    dataset_dir = proc_dir / MANUAL_DATASET_NAME
    if not dataset_dir.is_dir():
        sys.exit(f"[FATAL] Manually specified dataset directory not found: {dataset_dir}")
    print(f"[DATA] Using manually specified dataset directory: {dataset_dir.name}")

# 3) Inside it, locate all .parquet files and sort them by last modification time
parquet_files = list(dataset_dir.glob("*.parquet"))
if not parquet_files:
    sys.exit(f"[FATAL] No .parquet in {dataset_dir}")

parquet_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
PATCH_DF_PATH = parquet_files[0]
print(f"[DATA] Loading patch_df from: {PATCH_DF_PATH.name}")

# 4) Load the DataFrame
patch_df = pd.read_parquet(PATCH_DF_PATH)
print(f"[DATA] patch_df rows: {len(patch_df):,}")

# 5) Build the list of unique sources (WSI + ROI files)
unique_wsis = sorted(
    set(
        patch_df['wsi_path'].dropna().tolist() +
        patch_df['roi_file'].dropna().tolist()
    )
)
print(f"[DATA] unique sources to process: {len(unique_wsis):,}")

# %%
# Cell 3 – Safe Shard Writers Initialization (train/val/test)
import webdataset as wds
import sys
import os

# Final path: {project_root}/data/processed/dataset_{uuid}/webdataset/{split}/patches-0000.tar
WDATASET_DIR = dataset_dir / "webdataset"
splits = ["train", "val", "test"]

print(f"[INFO] Checking WebDataset shards at: {WDATASET_DIR}")

# 1. Check if non-empty shards already exist
shards_found = False
for sp in splits:
    split_dir = WDATASET_DIR / sp
    existing = list(split_dir.glob("patches-*.tar"))
    for tar in existing:
        if tar.stat().st_size > 0:
            print(f"❌ Found existing shard: {tar} ({tar.stat().st_size/1024**2:.2f} MB)")
            shards_found = True

if shards_found:
    print("\n[ABORT] At least one shard already exists and is non-empty.")
    print("        Rename or delete existing files before re-running this cell.")
    sys.exit(1)

# 2. Create empty directories for each split if no shards were found
for sp in splits:
    (WDATASET_DIR / sp).mkdir(parents=True, exist_ok=True)

# 3. Initialize shard writers
writers = {
    sp: wds.ShardWriter(
        str(WDATASET_DIR / sp / "patches-%04d.tar"),
        maxcount=SHARD_SIZE
    )
    for sp in splits
}

print("[INFO] Shard writers ready (empty).")

# %%
# Cell 3 bis – Helper: discard “black” / non-informative patches
import numpy as np
from PIL import Image

def is_patch_informative(pil_img: Image.Image, thresh: int = 10) -> bool:
    """
    Returns True if the grayscale standard deviation is > thresh.
    Used to avoid saving black or low-information patches.
    """
    gray = pil_img.convert("L")
    return np.array(gray).std() > thresh

# %%
# Cell X – Print shard file sizes for each split
import tarfile, os, glob

for split in ["train", "val", "test"]:
    tar_paths = sorted(glob.glob(str(WDATASET_DIR / split / "*.tar")))
    for tp in tar_paths:
        size_mb = os.path.getsize(tp) / (1024**2)
        print(f"{Path(tp).name:<25} → {size_mb:6.2f} MB")


# %%
# Cell 4 – Patch extraction (all WSI/ROI with bounding box & filtering)
import openslide, io, shutil
from tqdm.auto import tqdm
from PIL import Image

# Local cache for WSI files
WSI_LOCAL_DIR = Path("/content/WSI_cache")
WSI_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# --- DEV option: limit number of patches per WSI (only in debug stage) ---
MAX_PATCH_PER_WSI = 10 if stage_key == "debug" else None

total_patches = 0
error_log     = []

for wsi_path in tqdm(unique_wsis, desc="Processing source"):
    src = Path(wsi_path)
    print(f"\n[INFO] Processing source: {src.name}")
    if not src.exists():
        print("  [WARN] source not found on disk – skipped")
        error_log.append({"src": wsi_path, "error": "not_found"})
        continue

    # 1) copy to local SSD
    dst = WSI_LOCAL_DIR / src.name
    if not dst.exists():
        try:
            shutil.copy(src, dst)
        except Exception as e:
            print(f"  [ERROR] copy failed: {e}")
            error_log.append({"src": wsi_path, "error": f"copy_failed: {e}"})
            continue

    # 2) open the slide
    try:
        slide = openslide.OpenSlide(str(dst))
    except Exception as e:
        print(f"  [ERROR] cannot open slide: {e}")
        error_log.append({"src": wsi_path, "error": f"open_failed: {e}"})
        continue

    W, H = slide.dimensions
    sub_df = patch_df[
        (patch_df["wsi_path"] == wsi_path) |
        (patch_df["roi_file"]  == wsi_path)
    ]

    if sub_df.empty:
        slide.close()
        dst.unlink(missing_ok=True)
        continue

    # shuffle and optionally limit for debug
    sub_df = sub_df.sample(frac=1, random_state=RANDOM_SEED)
    if MAX_PATCH_PER_WSI:
        sub_df = sub_df.head(MAX_PATCH_PER_WSI)

    patch_cnt = 0
    for _, row in sub_df.iterrows():
        x, y = int(row["x"]), int(row["y"])

        # bounding-box check
        if x + PATCH_SIZE > W or y + PATCH_SIZE > H:
            continue

        # read region and convert to RGB
        try:
            img = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
        except Exception:
            continue

        # discard black / low-information patches
        if not is_patch_informative(img):
            continue

        # write to .tar: no explicit key → WebDataset will assign flat names
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        writers[row["split"]].write({"jpg": buf.getvalue()})

        patch_cnt     += 1
        total_patches += 1

    print(f"  [OK] saved {patch_cnt} patches from {src.name}")
    slide.close()
    dst.unlink(missing_ok=True)

print(f"\n✅ Patches successfully saved this run: {total_patches:,}")
if error_log:
    err_file = dataset_dir / "extract_errors.json"
    err_file.write_text(json.dumps(error_log, indent=2))
    print(f"⚠️  Errors logged → {err_file}")


# %%
# Cell 5 – Close writers & summary

for w in writers.values():
    w.close()

print(f"[SUMMARY] WebDataset dir → {WDATASET_DIR}")
print(f"[SUMMARY] Total patches  → {total_patches}")

if error_log:
    err_path = dataset_dir / "extract_errors.json"
    err_path.write_text(json.dumps(error_log, indent=2))
    print(f"[SUMMARY] Errors logged → {err_path}")
else:
    print("[SUMMARY] No errors 🎉")


# %%
# Cell 6 – WebDataset statistics: shard overview
import os, glob, tarfile

print("\n[STATS] WebDataset overview")

tar_paths = sorted(glob.glob(str(WDATASET_DIR / "*" / "patches-*.tar")))
print(f"• Shard files found : {len(tar_paths)}")

total_samples, valid_shards = 0, 0

for tp in tar_paths:
    size_mb = os.path.getsize(tp) / (1024**2)
    try:
        with tarfile.open(tp, "r") as tar:
            n_members  = len(tar.getmembers())
            n_samples  = n_members // 1           # 1 file (.jpg) per sample
            total_samples += n_samples
            valid_shards  += 1
            print(f"  - {Path(tp).name:<20} → {n_samples:5d} samples, {size_mb:6.2f} MB")
    except Exception as e:
        print(f"  - {Path(tp).name:<20} → ERROR ({e})")

print(f"\n• Valid shards          : {valid_shards}/{len(tar_paths)}")
print(f"• Total samples saved   : {total_samples:,}")

# – Detailed statistics per split and class
import os, tarfile, json
from collections import defaultdict, Counter
from pathlib import Path

print("\n[STATS • DETAILED] Split / subtype overview")

split_stats     = defaultdict(lambda: {"shards":0, "samples":0, "size_mb":0.0,
                                       "per_class": Counter()})
total_size_mb   = 0.0
total_samples   = 0
total_shards    = 0

for split_dir in (WDATASET_DIR / "train", WDATASET_DIR / "val", WDATASET_DIR / "test"):
    split = split_dir.name
    for tar_path in sorted(split_dir.glob("patches-*.tar")):
        size_mb = os.path.getsize(tar_path) / (1024**2)
        try:
            with tarfile.open(tar_path, "r") as tar:
                keys = [m.name for m in tar.getmembers() if m.isfile() and m.name.endswith(".jpg")]
                n_samples = len(keys)
                # extract subtype from the saved key (format: '{subtype}_{patient}_{x}_{y}.jpg')
                for k in keys:
                    subtype = Path(k).stem.split("_", 1)[0]
                    split_stats[split]["per_class"][subtype] += 1
        except Exception as e:
            print(f"  ⚠️  Could not open {tar_path.name}: {e}")
            continue

        # aggregate stats
        split_stats[split]["shards"]  += 1
        split_stats[split]["samples"] += n_samples
        split_stats[split]["size_mb"] += size_mb

        total_shards   += 1
        total_samples  += n_samples
        total_size_mb  += size_mb

# ── Pretty print ───────────────────────────────────────────────────────────────
for split, info in split_stats.items():
    print(f"\n🔹 Split: {split}")
    print(f"   • shards      : {info['shards']}")
    print(f"   • samples     : {info['samples']:,}")
    print(f"   • size (MB)   : {info['size_mb']:.2f}")
    print("   • per class   :")
    for cls, cnt in info["per_class"].items():
        print(f"       - {cls:8}: {cnt:,}")

print("\n==============================")
print(f"TOTAL shards    : {total_shards}")
print(f"TOTAL samples   : {total_samples:,}")
print(f"TOTAL size (GB) : {total_size_mb/1024:.2f} GB")
print("==============================")

# %%
# Cell 7 – Display thumbnails per class (20 patches per class) directly from .tar files on Drive
import tarfile
import random
from io import BytesIO
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# 1) Parameters
CLASSES     = ["CHROMO", "ONCO", "ccRCC", "pRCC", "not_tumor"]
SPLITS      = ["train", "val", "test"]
N_PER_CLASS = 25

# 2) WebDataset directory
WDATASET_DIR = dataset_dir / "webdataset"
print(f"[INFO] WebDataset dir: {WDATASET_DIR}")

# 3) Collect thumbnails per class
random.seed(RANDOM_SEED)
thumbs = {cls: [] for cls in CLASSES}

for split in SPLITS:
    for tar_path in sorted((WDATASET_DIR / split).glob("patches-*.tar")):
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile() and m.name.endswith(".jpg")]
            random.shuffle(members)

            for m in members:
                subtype = Path(m.name).stem.rsplit("_", 3)[0]
                if subtype in thumbs and len(thumbs[subtype]) < N_PER_CLASS:
                    img_data = tar.extractfile(m).read()
                    thumbs[subtype].append((Image.open(BytesIO(img_data)), m.name))  # add file name

            if all(len(thumbs[c]) >= N_PER_CLASS for c in CLASSES):
                break
    if all(len(thumbs[c]) >= N_PER_CLASS for c in CLASSES):
        break

# 4) Grid: 1 row per class, N columns
n_rows, n_cols = len(CLASSES), N_PER_CLASS
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 2.2, max(2, n_rows) * 2.2),
                         squeeze=False)

# 5) Show each thumbnail
for i, cls in enumerate(CLASSES):
    for j in range(n_cols):
        ax = axes[i, j]
        ax.axis("off")
        if j < len(thumbs[cls]):
            img, fname = thumbs[cls][j]
            ax.imshow(img)
            if j == 0:
                ax.set_title(cls, loc="left", fontsize=12)

# 6) Finalize layout
plt.show()


