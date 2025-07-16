# %%
# from google.colab import drive
# drive.mount('/content/drive', force_remount=False)

!pip install --quiet torch torchvision webdataset tqdm pillow

# %%
# Cell 1 – Environment Setup & Dependencies
import os, sys, subprocess, importlib
from pathlib import Path

print("📦 [DEBUG] Avvio configurazione ambiente...")

# --- Colab detection ---------------------------------------------------------#
IN_COLAB = Path("/content").exists()
if IN_COLAB:
    print("📍 [DEBUG] Ambiente Google Colab rilevato.")
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
else:
    print("💻 [DEBUG] Ambiente locale rilevato (VSCode o simile).")

# --- Project root ------------------------------------------------------------#
ENV_PATHS = {
    "colab": "/content/drive/MyDrive/ColabNotebooks/wsi-ssrl-rcc_project",
    "local": "/Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project",
}
PROJECT_ROOT = Path(ENV_PATHS["colab" if IN_COLAB else "local"]).resolve()
sys.path.append(str(PROJECT_ROOT / "src"))
print(f"📁 [DEBUG] PROJECT_ROOT impostato a: {PROJECT_ROOT}")

# --- Dependencies (installa solo se mancano) ---------------------------------#
def _pip_install(pkgs):
    import importlib.util
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if missing:
        print(f"🔧 [DEBUG] Installazione pacchetti mancanti: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    else:
        print("✅ [DEBUG] Tutti i pacchetti richiesti sono già installati.")

_pip_install([
    "torch", "torchvision", "webdataset", "tqdm",
    "pillow", "pyyaml", "joblib"
])


# %%
# Cell 2 – SLURM Submission via SSH per locale VSCode (debug .env + rsync)
import os, subprocess, traceback
from pathlib import Path
from textwrap import dedent

# Detect VSCode vs Colab
IN_COLAB  = Path("/content").exists()
# Use explicit check for not in Colab for VSCode specific logic
IN_VSCODE = not IN_COLAB and bool(os.environ.get("VSCODE_PID"))
print(f"🚀 Detected Colab={IN_COLAB}, VSCode={IN_VSCODE}")

if IN_VSCODE:
    from dotenv import load_dotenv, find_dotenv

    # 1) Carica .env (ricerca automatica)
    dotenv_path = find_dotenv()
    if not dotenv_path:
        raise FileNotFoundError("❌ Non ho trovato alcun .env! Mettilo nella root del progetto.")
    print(f"🔍 Carico .env da {dotenv_path}")
    load_dotenv(dotenv_path, override=True)

    # 2) Controlla le env vars
    REMOTE_USER      = os.getenv("CLUSTER_USER")
    REMOTE_HOST      = os.getenv("CLUSTER_HOST")
    REMOTE_BASE_PATH = os.getenv("REMOTE_BASE_PATH")
    SBATCH_MODULE    = os.getenv("SBATCH_MODULE", "python/3.9")
    SBATCH_PARTITION = os.getenv("SBATCH_PARTITION", "global")
    MAIL_USER        = os.getenv("RESPONSABILE_EMAIL", os.getenv("MEMBER_EMAIL"))

    missing = [v for v in ("CLUSTER_USER","CLUSTER_HOST","REMOTE_BASE_PATH") if not os.getenv(v)]
    if missing:
        raise KeyError(f"🌱 Mancano queste env vars: {missing}. Controlla il .env.")

    # 3) Prepara lo script sbatch locale per la sottomissione remota
    LOCAL_SCRIPT = Path.cwd() / "hpc_submit.sh"
    print(f"   • SSH target: {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE_PATH}")

    # 4) Genera sbatch script
    header = dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=rcc_ssrl_launch
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem-per-cpu=4G
        #SBATCH --time=2:00:00
        #SBATCH --gres=gpu:1
        #SBATCH --partition={SBATCH_PARTITION}
        #SBATCH --output=%x_%j.out
        #SBATCH --mail-type=END,FAIL
        #SBATCH --mail-user={MAIL_USER}
        #SBATCH --workdir={REMOTE_BASE_PATH}

        module purge
        module load {SBATCH_MODULE}

        cd {REMOTE_BASE_PATH}
    """)
    header += f"\npython {PROJECT_ROOT}/4-launch_training.py --config config/training.yaml\n"

    LOCAL_SCRIPT.write_text(header)
    LOCAL_SCRIPT.chmod(0o755)
    print(f"📝 Wrote sbatch script: {LOCAL_SCRIPT}")

    try:
        # 5) Crea cartella remota
        subprocess.run(
            ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}", f"mkdir -p {REMOTE_BASE_PATH}"],
            check=True
        )
        print("🔄 Remote directory ensured")

        # 6) Sync progetto (esclude dati pesanti)
        subprocess.run([
            "rsync","-avz","--delete",
            "--exclude","data/processed",
            f"{PROJECT_ROOT}/",
            f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE_PATH}/"
        ], check=True)
        print("🔄 Project synchronized via rsync")

        # 7) Sottometti job
        res = subprocess.run(
            ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}",
             f"cd {REMOTE_BASE_PATH} && sbatch {LOCAL_SCRIPT.name}"],
            capture_output=True, text=True, check=True
        )
        print(f"🔍 sbatch stdout: {res.stdout.strip()}")
        print(f"📬 Job submitted: {res.stdout.strip().split()[-1]}")

    except subprocess.CalledProcessError as e:
        print("❌ SLURM submission failed:")
        print(e.stdout, e.stderr)
    except Exception:
        print("❌ Unexpected error:")
        traceback.print_exc()

else:
    print("⚠️ SLURM integration skipped: non in locale VSCode.")

# %%
# Cell 3 – Dynamic import of utils.training_utils
import sys
import importlib.util
from pathlib import Path

# 1) locate & load the module file
utils_path = PROJECT_ROOT / "src" / "utils" / "training_utils.py"
spec       = importlib.util.spec_from_file_location("utils.training_utils", str(utils_path))
utils_mod  = importlib.util.module_from_spec(spec)     # type: ignore[arg-type]
assert spec and spec.loader, f"Cannot load spec for {utils_path}"
spec.loader.exec_module(utils_mod)                     # type: ignore[assignment]
sys.modules["utils.training_utils"] = utils_mod        # register in sys.modules
print(f"[DEBUG] Loaded utils.training_utils from {utils_path}")

# 2) import what we need
from utils.training_utils import (
    TRAINER_REGISTRY,
    get_latest_checkpoint,
    load_checkpoint,
)

print("[DEBUG] Imported:")
print("  • TRAINER_REGISTRY keys:", list(TRAINER_REGISTRY.keys()))
print("  • get_latest_checkpoint →", get_latest_checkpoint)
print("  • load_checkpoint       →", load_checkpoint)


# %%
# Cell 4 – Configuration & Directory Setup (formatted and absolute paths)
import yaml
from pathlib import Path

# 1. Load config file
cfg_path   = PROJECT_ROOT / "config" / "training.yaml"
cfg        = yaml.safe_load(cfg_path.read_text())
DATASET_ID = cfg["data"]["dataset_id"]

# 2. Format and resolve absolute paths for train/val/test
for split in ("train", "val", "test"):
    rel_path = cfg["data"][split].format(dataset_id=DATASET_ID)
    abs_path = (PROJECT_ROOT / rel_path).resolve()
    cfg["data"][split] = str(abs_path)
    print(f"[DEBUG] {split.upper()} → {abs_path}")

# 3. Format and resolve models dir
rel_models = cfg["output_dir"].format(dataset_id=DATASET_ID)
MODELS_DIR = (PROJECT_ROOT / rel_models).resolve()
MODELS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[DEBUG] MODELS_DIR → {MODELS_DIR}")


# %%
# Cell 5 – Import all trainer modules with debug prints
import importlib, sys
from utils.training_utils import TRAINER_REGISTRY

trainer_mods = [
    "trainers.simclr",
    "trainers.moco_v2",
    "trainers.rotation",
    "trainers.jigsaw",
    "trainers.supervised",
    "trainers.transfer",
]

for module_name in trainer_mods:
    if module_name in sys.modules:
        print(f"[DEBUG] Reloading module: {module_name}")
        importlib.reload(sys.modules[module_name])
    else:
        print(f"[DEBUG] Importing module: {module_name}")
        importlib.import_module(module_name)

print(f"[DEBUG] Registered trainers: {list(TRAINER_REGISTRY.keys())}")


# %%
# Cell 6
def save_artifact(subdir: str, filename: str) -> Path:
    out_dir = MODELS_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename

def append_report(md_rel: Path, header: str, body: str):
    md_abs = MODELS_DIR / md_rel
    md_abs.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y‑%m‑%d %H:%M")
    with md_abs.open("a") as f:
        f.write(f"\n\n### {header}  \n*{ts}*\n\n{body}\n")


# %%
# Cell 7 – launch_training() con logging dettagliato in checkpoint_report.md
import torch
import inspect
import time
import datetime
from pathlib import Path
from utils.training_utils import (
    TRAINER_REGISTRY,
    load_checkpoint,
    get_latest_checkpoint,
)
from trainers.train_classifier import train_classifier

def launch_training(cfg: dict) -> None:
    run_model  = cfg.get("run_model", "all").lower()
    models_cfg = cfg["models"]
    tasks = models_cfg.items() if run_model == "all" else [(run_model, models_cfg[run_model])]

    for name, m_cfg in tasks:
        # --- Trainer setup ---------------------------------------------------- #
        TrainerCls = TRAINER_REGISTRY.get(name)
        if TrainerCls is None:
            raise KeyError(f"Trainer '{name}' non registrato")
        trainer = TrainerCls(m_cfg, cfg["data"])
        has_val  = hasattr(trainer, "validate_epoch")
        epochs   = int(m_cfg["training"]["epochs"])
        batch_sz = int(m_cfg["training"]["batch_size"])
        device   = getattr(trainer, "device", "cpu")

        print(f"Device: {device} 🚀  Starting training for model '{name}'")
        print(f"→ Model config: {m_cfg}")
        print(f"Epochs: {epochs} | Batch size: {batch_sz}\n")

        # --- Paths per artefatti ---------------------------------------------- #
        ckpt_dir  = MODELS_DIR / f"{name}/checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_best = ckpt_dir / f"{TrainerCls.__name__}_best.pt"

        feat_path = save_artifact(f"{name}/features",   f"{name}_features.pt")
        clf_path  = save_artifact(f"{name}/classifier", f"{name}_classifier.joblib")
        report_md = Path(f"{name}/checkpoints/checkpoint_report.md")

        # --- Checkpoint già presente? ---------------------------------------- #
        latest_ckpt = get_latest_checkpoint(ckpt_dir, prefix=TrainerCls.__name__)
        if latest_ckpt is not None:
            print(f"⏭️  Checkpoint trovato per '{name}': {latest_ckpt.name} – skip training")
            model = torch.nn.Sequential(trainer.encoder,
                                        getattr(trainer, "projector", torch.nn.Identity()))
            load_checkpoint(latest_ckpt, model=model)
            trainer.encoder = model[0].to(trainer.device)
            append_report(report_md, "Checkpoint ri-usato",
                          f"`{latest_ckpt.relative_to(MODELS_DIR)}`")
            skip_training = True
        else:
            skip_training = False

        # --- Training loop ---------------------------------------------------- #
        if not skip_training:
            total_batches = getattr(trainer, "batches_train", None) or len(trainer.train_loader)
            print(f"TOTAL BATCHES {total_batches}")

            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                running_loss, running_correct, total_samples = 0.0, 0, 0

                print(f"--- Epoch {epoch}/{epochs} ---")
                for i, batch in enumerate(trainer.train_loader, start=1):
                    sig    = inspect.signature(trainer.train_step)
                    result = trainer.train_step(batch) if len(sig.parameters) == 1 else trainer.train_step(*batch)

                    if len(result) == 4:
                        _, loss, correct, bs = result
                    else:
                        loss, bs = result
                        correct = 0

                    running_loss    += loss * bs
                    running_correct += correct
                    total_samples   += bs

                    avg_loss = running_loss / total_samples
                    avg_acc  = (running_correct / total_samples) if has_val else 0.0
                    elapsed  = time.time() - epoch_start
                    pct      = (i / total_batches) * 100
                    eta      = (elapsed / i) * (total_batches - i)

                    msg = f"  Batch {i}/{total_batches} ({pct:.1f}%) | Loss: {avg_loss:.4f}"
                    if has_val:
                        msg += f" | Acc: {avg_acc:.3f}"
                    msg += f" | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s"
                    print(msg)

                if has_val:
                    val_loss, val_acc = trainer.validate_epoch()
                    print(f"Val -> Loss: {val_loss:.4f} | Acc: {val_acc:.3f}")
                    metric = val_acc
                else:
                    val_loss = val_acc = None
                    metric = running_loss / total_samples

                trainer.post_epoch(epoch, metric)
                epoch_time = time.time() - epoch_start
                train_loss = running_loss / total_samples
                train_acc  = running_correct / total_samples if has_val else None

                print(f"Epoch {epoch} completed in {epoch_time:.1f}s\n")

                # --- Scrittura metriche in Markdown ---------------------------- #
                metrics_md = (
                    f"| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Duration |\n"
                    f"|-------|------------|-----------|----------|---------|----------|\n"
                    f"| {epoch} "
                    f"| {train_loss:.4f} "
                    f"| {train_acc:.3f} " if train_acc is not None else "| n/a "
                )
                metrics_md += (
                    f"| {val_loss:.4f} | {val_acc:.3f} " if val_loss is not None else "| n/a | n/a "
                )
                metrics_md += f"| {epoch_time:.1f}s |"

                append_report(report_md, f"Epoch {epoch} summary", metrics_md)

            # Salva checkpoint finale
            new_best = get_latest_checkpoint(ckpt_dir, prefix=TrainerCls.__name__)
            if new_best and new_best != ckpt_best:
                new_best.replace(ckpt_best)
            append_report(report_md, "Checkpoint salvato",
                          f"`{ckpt_best.relative_to(MODELS_DIR)}`")

        # --- SSL: estrazione feature + classificatore -------------------------- #
        if not has_val:
            if not feat_path.exists():
                trainer.extract_features_to(str(feat_path))
            feat_shape = tuple(torch.load(feat_path)["features"].shape)
            append_report(Path(f"{name}/features/features_report.md"),
                          "Features estratte",
                          f"`{feat_path.relative_to(MODELS_DIR)}` shape = {feat_shape}")

            print(f"🔍 Extracting & training classifier for '{name}'")
            train_classifier(str(feat_path), str(clf_path))
            append_report(Path(f"{name}/classifier/classifier_report.md"),
                          "Classifier addestrato",
                          f"`{clf_path.relative_to(MODELS_DIR)}`")

# Cell 8 – Run!
launch_training(cfg)



