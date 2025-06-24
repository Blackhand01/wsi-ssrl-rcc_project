# %%
# Cell 1 – Environment Setup & Dependency Management (Colab & VSCode compatible)
import os, sys, subprocess
from pathlib import Path

# Detect Google Colab environment
IN_COLAB = Path("/content").exists()
if IN_COLAB:
    from google.colab import drive                          # type: ignore
    drive.mount("/content/drive", force_remount=False)

# Load YAML config from correct path
import yaml
CONFIG_PATH = (
    Path("/content/drive/MyDrive/Colab Notebooks/MLA_PROJECT/wsi-ssrl-rcc_project/config/training.yaml")
    if IN_COLAB
    else Path.cwd() / "config" / "training.yaml"
)
CFG = yaml.safe_load(CONFIG_PATH.read_text())

# Define PROJECT_ROOT based on current environment
colab_root = Path(CFG["env_paths"]["colab"])
local_root = Path(CFG["env_paths"]["local"])
PROJECT_ROOT = colab_root if colab_root.exists() else local_root

# Add `src/` directory to Python path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Install missing Python dependencies (only when needed)
def install_if_missing(packages):
    import importlib.util
    missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])

install_if_missing([
    "torch", "torchvision", "webdataset", "tqdm",
    "pillow", "scikit-learn", "joblib"
])


# %%
# Cell 2 – Path normalization and debug print (Colab & VSCode)
from pathlib import Path

# Normalize relative dataset paths using PROJECT_ROOT
for split in ("train", "val", "test"):
    rel_path = CFG["data"].get(split)
    if rel_path:
        CFG["data"][split] = str(PROJECT_ROOT / rel_path)

# Extract dataset ID and model output directory
DATASET_ID = CFG["data"]["dataset_id"]
MODELS_DIR = (PROJECT_ROOT / CFG["output_dir"].format(dataset_id=DATASET_ID)).resolve()

# Debug info (print only)
print("📁 Project root   :", PROJECT_ROOT)
print("📦 Dataset ID     :", DATASET_ID)
print("💾 Models dir     :", MODELS_DIR)
print("🧪 Normalized paths:")
for split in ("train", "val", "test"):
    print(f"   • {split}: {CFG['data'].get(split)}")


# %%
# Cell 3 – Dynamic import of training_utils and trainer modules
import importlib.util
import sys

# Load utils/training_utils.py dynamically
utils_path = PROJECT_ROOT / "src" / "utils" / "training_utils.py"
spec = importlib.util.spec_from_file_location("utils.training_utils", str(utils_path))
training_utils = importlib.util.module_from_spec(spec)              # type: ignore[arg-type]
assert spec and spec.loader
spec.loader.exec_module(training_utils)                             # type: ignore[assignment]
sys.modules["utils.training_utils"] = training_utils

# Import core functions from training_utils
from utils.training_utils import TRAINER_REGISTRY, load_checkpoint, get_latest_checkpoint

# Import all trainer modules (SimCLR, MoCo, etc.)
trainer_modules = [
    "trainers.simclr",
    "trainers.moco_v2",
    "trainers.rotation",
    "trainers.jigsaw",
    "trainers.supervised",
    "trainers.transfer",
]
for module_name in trainer_modules:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        importlib.import_module(module_name)

print("✅ training_utils and trainer modules loaded successfully.")


# %%
# Cell 4 – Evaluation helpers and majority voting utilities
import numpy as np
import torch, joblib, webdataset as wds
import torchvision.transforms as T
from PIL import Image
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix

# --- Utility: Extract patient ID and label from key ------------------------ #
def extract_patient_id(key: str) -> str:
    for part in key.split("_"):
        if part.startswith(("HP", "H")):
            return part
    return "UNKNOWN"

def extract_label_from_key(key: str) -> str:
    return "not_tumor" if key.startswith("not_tumor") else key.split("_")[0]

# --- Utility: Load test data from WebDataset ------------------------------- #
def make_loader(wds_path: str, batch_size: int = 64):
    dataset = (
        wds.WebDataset(wds_path, shardshuffle=False, handler=wds.warn_and_continue, empty_check=False)
        .decode("pil")
        .map(lambda sample: {
            "img": T.ToTensor()(
                next((v for k, v in sample.items() if isinstance(v, Image.Image)), None).convert("RGB")),
            "key": sample["__key__"] + "." + next((k for k in sample.keys() if k.endswith(".jpg")), "")
        })
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=0, pin_memory=True)

# --- Utility: Patient-level majority voting ------------------------------- #
def majority_vote(keys: list[str], preds: list[int], label_encoder, exclude_label: str = "not_tumor"):
    votes_by_patient = defaultdict(list)
    labels_by_patient = defaultdict(list)

    for k, p in zip(keys, preds):
        pid = extract_patient_id(k)
        label = extract_label_from_key(k)
        if label_encoder.classes_[p] != exclude_label:
            votes_by_patient[pid].append(p)
        labels_by_patient[pid].append(label)

    y_true, y_pred, valid_pids = [], [], []
    for pid, vote_list in votes_by_patient.items():
        gt_labels = [lab for lab in labels_by_patient[pid] if lab != exclude_label]
        if len(set(gt_labels)) != 1 or not vote_list:
            continue
        gt_index = label_encoder.transform([gt_labels[0]])[0]
        majority = Counter(vote_list).most_common(1)[0][0]
        y_true.append(gt_index)
        y_pred.append(majority)
        valid_pids.append(pid)

    return y_true, y_pred, valid_pids

# --- Evaluation for Self-Supervised models ------------------------------- #
def evaluate_selfsupervised(trainer, classifier_path: str, test_path: str):
    print(f"\n🧪 Evaluating Self-Supervised model: {classifier_path}")
    model_bundle = joblib.load(classifier_path)
    clf = model_bundle["model"]
    le  = model_bundle["label_encoder"]

    from trainers.extract_features import extract_features
    loader = make_loader(test_path)
    feats = extract_features(trainer.encoder, loader, trainer.device)

    X     = feats["features"].cpu().numpy()
    keys  = feats["keys"]
    preds = clf.predict(X)

    y_true, y_pred, patient_ids = majority_vote(keys, preds, le)

    if not y_true:
        print("⚠️ No evaluable patients (Self-Supervised)")
        return

    _print_report(y_true, y_pred, le, patient_ids)

# --- Evaluation for Supervised and Transfer models ------------------------ #
def evaluate_supervised(trainer, test_path: str):
    print(f"\n🧪 Evaluating Supervised/Transfer model...")
    loader = make_loader(test_path)
    model = trainer.model.to(trainer.device).eval()
    le = trainer.label_encoder

    preds, keys = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["img"].to(trainer.device))
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            keys.extend(batch["key"])

    y_true, y_pred, patient_ids = majority_vote(keys, preds, le)

    if not y_true:
        print("⚠️ No evaluable patients (Supervised/Transfer)")
        return

    _print_report(y_true, y_pred, le, patient_ids)

# --- Report printing ------------------------------------------------------ #
def _print_report(y_true, y_pred, label_encoder, patient_ids):
    classes = [c for c in label_encoder.classes_ if c != "not_tumor"]
    print("\n📊 Classification report (majority voting per patient):")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("📉 Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"✅ Total evaluated patients: {len(y_true)}")

    print("\n🧾 Per-patient results:")
    for pid, t, p in zip(patient_ids, y_true, y_pred):
        true_label = label_encoder.inverse_transform([t])[0]
        pred_label = label_encoder.inverse_transform([p])[0]
        print(f"• Patient {pid}: predicted = {pred_label} | true = {true_label}")


# %%
# Cell 5 – Evaluation loop for all models
run_model = CFG.get("run_model", "all").lower()
models_cfg = CFG["models"]
tasks = models_cfg.items() if run_model == "all" else [(run_model, models_cfg[run_model])]

test_path = CFG["data"]["test"]

for name, model_cfg in tasks:
    print(f"\n🔍 Evaluating model: {name}")

    if name not in TRAINER_REGISTRY:
        raise KeyError(f"❌ Trainer '{name}' is not registered.")

    trainer = TRAINER_REGISTRY[name](model_cfg, CFG["data"])
    ckpt_dir = MODELS_DIR / name / "checkpoints"
    ckpt = get_latest_checkpoint(ckpt_dir, prefix=trainer.__class__.__name__)
    if ckpt is None:
        print(f"⚠️ No checkpoint found for '{name}', skipping evaluation.")
        continue

    print(f"📥 Loading checkpoint: {ckpt.name}")
    
    if name in ("supervised", "transfer"):
        # Load full supervised/transfer model
        load_checkpoint(ckpt, model=trainer.model)
        trainer.model = trainer.model.to(trainer.device)
        evaluate_supervised(trainer, test_path)
    else:
        # Load encoder + projector + external classifier
        clf_path = MODELS_DIR / name / "classifier" / f"{name}_classifier.joblib"
        if not clf_path.exists():
            print(f"⚠️ No classifier found for '{name}', skipping.")
            continue

        model = torch.nn.Sequential(trainer.encoder, trainer.projector)
        load_checkpoint(ckpt, model=model)
        trainer.encoder = model[0].to(trainer.device)
        trainer.projector = model[1].to(trainer.device)
        evaluate_selfsupervised(trainer, clf_path, test_path)


# %%
# Cell 6 – Check saved artifacts for each evaluated model
from pathlib import Path

print("\n📂 Artifacts summary:")

for name, _ in tasks:
    model_dir = MODELS_DIR / name
    if not model_dir.exists():
        continue
    print(f"\n🔎 Model: {name}")
    for sub in ["checkpoints", "features", "classifier"]:
        subdir = model_dir / sub
        if subdir.exists():
            files = list(subdir.glob("*"))
            if files:
                print(f"  📁 {sub}/")
                for f in files:
                    print(f"     - {f.name}")
            else:
                print(f"  ⚠️ {sub}/ is empty.")
        else:
            print(f"  ❌ Missing subdirectory: {sub}/")



