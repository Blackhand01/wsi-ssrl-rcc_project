# %%
# ⬇️ correct_shards.py  ───────────────────────────────────────────────
"""
Convert a WebDataset shard created by the *new* pipeline into a shard that
behaves like the *old* one (no __key__, no dots inside the name).

Usage:
- Update SRC_TAR_PATH and DST_TAR_PATH below.
- Run this cell; it creates the destination shard and prints a final check.
"""

from pathlib import Path
import tarfile, io, os, shutil
from tqdm.auto import tqdm

# ────────────────────────────────────────────────────────────────────────
# 🔧 CONFIG – edit only these two paths
SRC_TAR_PATH = Path(
    "/Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project/"
    "data/processed/dataset_7b24514c/webdataset/test/patches-0000.tar"
)
DST_TAR_PATH = Path(
    "/Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project/"
    "data/processed/dataset_7b24514c/webdataset2/test/patches-0000.tar"
)
# ────────────────────────────────────────────────────────────────────────

assert SRC_TAR_PATH.exists(), f"❌ Source tar not found: {SRC_TAR_PATH}"
DST_TAR_PATH.parent.mkdir(parents=True, exist_ok=True)
if DST_TAR_PATH.exists():
    print("⚠️  Destination tar already exists → lo rinomino con .bak")
    backup_path = DST_TAR_PATH.with_suffix(".bak")
    if backup_path.exists():
        backup_path.unlink()  # elimina vecchio backup
    DST_TAR_PATH.rename(backup_path)


def sanitize(name: str) -> str:
    """
    Remove every dot (.) from basename except the final extension dot.
    Example: 'ccRCC_HP19.754_19587_111282.jpg' → 'ccRCC_HP19754_19587_111282.jpg'
    """
    stem, ext = os.path.splitext(name)
    stem = stem.replace(".", "")      # kill dots
    return f"{stem}{ext.lower()}"

with tarfile.open(SRC_TAR_PATH, "r") as src_tar, \
     tarfile.open(DST_TAR_PATH, "w") as dst_tar:

    members = [m for m in src_tar.getmembers()
               if m.isfile() and m.name.lower().endswith(".jpg")]
    print(f"📦 Source contains {len(members):,} .jpg members")

    for i, m in enumerate(tqdm(members, desc="Re-packing")):
        data = src_tar.extractfile(m).read()      # raw JPEG bytes
        new_name = sanitize(Path(m.name).name)    # flat, safe filename
        # Build new TarInfo
        ti = tarfile.TarInfo(name=new_name)
        ti.size = len(data)
        dst_tar.addfile(ti, io.BytesIO(data))

print(f"✅ Finished. New shard saved → {DST_TAR_PATH}")

# ── Quick sanity check ────────────────────────────────────────────────
import webdataset as wds
n_files = sum(1 for _ in tarfile.open(DST_TAR_PATH, "r")
              if _.isfile() and _.name.lower().endswith(".jpg"))
n_samples = sum(1 for _ in wds.WebDataset(str(DST_TAR_PATH)).decode("pil"))
print(f"🧾 JPG in tar   : {n_files:,}")
print(f"🧪 Samples read : {n_samples:,}")
assert n_files == n_samples, "‼️ mismatch – something is still wrong!"
# ────────────────────────────────────────────────────────────────────────


# %%
import webdataset as wds
# Dataset 1 (vecchia pipeline)
webdataset1 = wds.WebDataset(
    "/Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project/data/processed/dataset_7b24514c/webdataset/test/patches-0000.tar"
).decode("pil")

# Dataset 2 (nuova pipeline)
webdataset2 = wds.WebDataset(
    "/Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project/data/processed/dataset_7b24514c/webdataset2/test/patches-0000.tar"
).decode("pil")

count_1 = 0
count_2 = 0

# Conta i sample effettivamente leggibili
count_1 = sum(1 for _ in webdataset1)
count_2 = sum(1 for _ in webdataset2)
print(f"✅ Dataset webdataset/test/patches-0000.tar: {count_1} samples")
print(f"✅ Dataset webdataset2/test/patches-0000.tar: {count_2} samples")


# %%

from PIL import Image
import tarfile
import io

tar_path =     "/Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project/data/processed/dataset_7b24514c/webdataset2/test/patches-0000.tar"

bad = []

with tarfile.open(tar_path, "r") as tar:
    members = tar.getmembers()
    for member in members:
        if member.isfile() and member.name.endswith(".jpg"):
            try:
                img = Image.open(tar.extractfile(member))
                img.verify()  # checks if corrupted
            except Exception as e:
                bad.append((member.name, str(e)))

print(f"🧪 Corrupted images: {len(bad)} / {len(members)}")
if bad:
    print("🚫 Esempio:", bad[0]) 


