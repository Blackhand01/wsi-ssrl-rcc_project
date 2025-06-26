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
# Cell 2 – SLURM Submission via SSH per locale VSCode (upload dataset once + rsync + sshpass)
import os, subprocess, traceback, shutil
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv, find_dotenv
import yaml

# --- 1) Detect environment ---------------------------------------------------
IN_COLAB  = Path("/content").exists()
IN_VSCODE = not IN_COLAB and bool(os.environ.get("VSCODE_PID"))
print(f"🚀 Detected Colab={IN_COLAB}, VSCode={IN_VSCODE}")

if IN_VSCODE:
    # --- 2) Load .env ----------------------------------------------------------
    dotenv_path = find_dotenv()
    if not dotenv_path:
        raise FileNotFoundError("❌ .env not found! Place it in the project root.")
    print(f"🔍 Loading .env from {dotenv_path}")
    load_dotenv(dotenv_path, override=True)

    # --- 3) Read training.yaml to get dataset_id --------------------------------
    cfg_path   = PROJECT_ROOT / "config" / "training.yaml"
    cfg        = yaml.safe_load(cfg_path.read_text())
    DATASET_ID = cfg["data"]["dataset_id"]

    # --- 4) Read env vars ------------------------------------------------------
    REMOTE_USER      = os.getenv("CLUSTER_USER")
    REMOTE_HOST      = os.getenv("CLUSTER_HOST")
    REMOTE_BASE_PATH = os.getenv("REMOTE_BASE_PATH")
    SBATCH_MODULE    = os.getenv("SBATCH_MODULE", "python/3.9")
    SBATCH_PARTITION = os.getenv("SBATCH_PARTITION", "global")
    MAIL_USER        = os.getenv("RESPONSABILE_EMAIL", os.getenv("MEMBER_EMAIL"))
    CLUSTER_PW       = os.getenv("CLUSTER_PASSWORD", "")

    missing = [v for v in ("CLUSTER_USER","CLUSTER_HOST","REMOTE_BASE_PATH") if not os.getenv(v)]
    if missing:
        raise KeyError(f"🌱 Missing env vars: {missing}. Check your .env file.")

    # --- 5) Check for sshpass --------------------------------------------------
    USE_SSHPASS = bool(CLUSTER_PW and shutil.which("sshpass"))
    if USE_SSHPASS:
        print("🔐 Using sshpass for non-interactive SSH/rsync.")
    else:
        print("🔐 sshpass not found or no CLUSTER_PASSWORD: falling back to interactive SSH.")

    def ssh_cmd(remote_expr: str):
        """Return the command list to run an SSH command (with or without sshpass)."""
        base = []
        if USE_SSHPASS:
            base += ["sshpass", "-p", CLUSTER_PW]
        base += ["ssh", f"{REMOTE_USER}@{REMOTE_HOST}", remote_expr]
        return base

    def rsync_cmd_args(src: str, dest: str, exclude_processed: bool = True):
        """Return args for rsync, using sshpass if available."""
        args = ["rsync", "-avz"]
        if exclude_processed:
            args += ["--exclude", "data/processed"]
        if USE_SSHPASS:
            sshpart = f"sshpass -p {CLUSTER_PW} ssh -o StrictHostKeyChecking=no"
            args += ["-e", sshpart]
        args += [src, dest]
        return args

    # --- 6) Write local sbatch script ------------------------------------------
    LOCAL_SCRIPT = Path.cwd() / "hpc_submit.sh"
    print(f"   • SSH target: {REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE_PATH}")

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
        # --- 7) Ensure remote base directory exists -----------------------------
        subprocess.run(ssh_cmd(f"mkdir -p {REMOTE_BASE_PATH}"), check=True)

        # --- 8) Sync project code only (excl. processed) -----------------------
        prj_src  = f"{PROJECT_ROOT}/"
        prj_dest = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE_PATH}/"
        subprocess.run(rsync_cmd_args(prj_src, prj_dest, exclude_processed=True), check=True)
        print("🔄 Project code synchronized via rsync")

        # --- 9) Upload dataset once if missing ---------------------------------
        remote_ds = f"{REMOTE_BASE_PATH}/data/processed/{DATASET_ID}"
        # test -d returns 0 if exists
        test = subprocess.run(ssh_cmd(f"test -d {remote_ds}"), capture_output=True)
        if test.returncode != 0:
            print("📦 Dataset not on cluster, uploading now (this may take a while)…")
            local_ds  = f"{PROJECT_ROOT}/data/processed/{DATASET_ID}/"
            remote_ds_dest = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_ds}/"
            subprocess.run(rsync_cmd_args(local_ds, remote_ds_dest, exclude_processed=False), check=True)
            print("✅ Dataset uploaded to cluster")
        else:
            print("📦 Dataset already present on cluster, skipping upload")

        # --- 10) Submit SLURM job via SSH ---------------------------------------
        # If using sshpass, we can capture output; otherwise let interactive prompt through
        if USE_SSHPASS:
            result = subprocess.run(
                ssh_cmd(f"cd {REMOTE_BASE_PATH} && sbatch {LOCAL_SCRIPT.name}"),
                capture_output=True, text=True, check=True
            )
            print(f"🔍 sbatch stdout: {result.stdout.strip()}")
            print(f"📬 Job submitted: {result.stdout.strip().split()[-1]}")
        else:
            # this will prompt for password in the terminal
            subprocess.run(ssh_cmd(f"cd {REMOTE_BASE_PATH} && sbatch {LOCAL_SCRIPT.name}"), check=True)

    except subprocess.CalledProcessError as e:
        print("❌ SLURM submission failed:")
        print(e.stdout or "", e.stderr or "")
    except Exception:
        print("❌ Unexpected error during SLURM submission:")
        traceback.print_exc()

else:
    print("⚠️ SLURM integration skipped: not running in local VSCode.")


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
import yaml, datetime, os
from pathlib import Path

# ------------------------------------------------------------------ #
# 0) EXP_CODE: riprendi da YAML → env → genera nuovo                 #
# ------------------------------------------------------------------ #

yaml_exp = cfg.get("exp_code", "")           # <─ nuovo parametro
env_exp  = os.environ.get("EXP_CODE", "")

if yaml_exp:                                 # 1) priorità al file YAML
    EXP_CODE = yaml_exp
elif env_exp:                                # 2) poi variabile d’ambiente
    EXP_CODE = env_exp
else:                                        # 3) altrimenti nuovo timestamp
    EXP_CODE = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# salva nel processo per eventuali figli
os.environ["EXP_CODE"] = EXP_CODE

print(f"[DEBUG] EXP_CODE → {EXP_CODE}")


# ------------------------------------------------------------------ #
# 1) Carica configurazione generale                                  #
# ------------------------------------------------------------------ #
cfg_path  = PROJECT_ROOT / "config" / "training.yaml"
cfg       = yaml.safe_load(cfg_path.read_text())
DATASET_ID = cfg["data"]["dataset_id"]
cfg["experiment_code"] = EXP_CODE     # lo inseriamo nel dict per eventuali usi downstream

# ------------------------------------------------------------------ #
# 2) Percorsi assoluti (train / val / test)                          #
# ------------------------------------------------------------------ #
for split in ("train", "val", "test"):
    rel = cfg["data"][split].format(dataset_id=DATASET_ID)
    abs_ = (PROJECT_ROOT / rel).resolve()
    cfg["data"][split] = str(abs_)
    print(f"[DEBUG] {split.upper()} → {abs_}")

# ------------------------------------------------------------------ #
# 3) Struttura directory esperimento                                 #
# ------------------------------------------------------------------ #
EXP_ROOT = PROJECT_ROOT / "data" / "processed" / str(DATASET_ID)
EXP_BASE = EXP_ROOT / "experiments" / EXP_CODE                   # unica per tutta la run
EXP_BASE.mkdir(parents=True, exist_ok=True)
(EXP_ROOT / "experiments.md").touch(exist_ok=True)               # indice globale

print(f"[DEBUG] EXP_BASE → {EXP_BASE}")

# ------------------------------------------------------------------ #
# 4) Salva la **copia YAML** solo se non esiste già                  #
# ------------------------------------------------------------------ #
exp_yaml = EXP_BASE / f"training_{EXP_CODE}.yaml"
if not exp_yaml.exists():                          # evita duplicati
    exp_yaml.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"[DEBUG] Scritto   {exp_yaml}")
else:
    print(f"[DEBUG] Config già presente → {exp_yaml}")

# %%
# Cell 5 – Import all trainer modules
import importlib, sys
from utils.training_utils import TRAINER_REGISTRY

trainer_mods = [
    "trainers.simclr",
    "trainers.moco_v2",
    "trainers.rotation",
    "trainers.jigsaw",
    "trainers.supervised",   # ✅ CORRETTO
    "trainers.transfer",
]

for m in trainer_mods:
    importlib.reload(sys.modules[m]) if m in sys.modules else importlib.import_module(m)

print(f"[DEBUG] Registered trainers: {list(TRAINER_REGISTRY.keys())}")


# %%
# %% -------------------------------------------------------------------- #
# Cell 6 – Helper utilities (Tee, paths, selezione, …)                    #
# ----------------------------------------------------------------------- #
import contextlib, sys, time, inspect
from pathlib import Path
import torch
from utils.training_utils import get_latest_checkpoint, load_checkpoint
from trainers.train_classifier import train_classifier

# ──────────────────────────────────────────────────────────────────────── #
# I/O helpers                                                             #
# ──────────────────────────────────────────────────────────────────────── #
class _Tee:
    """Duplica stdout / stderr su console *e* file."""
    def __init__(self, *targets): self.targets = targets
    def write(self, data):  [t.write(data) and t.flush() for t in self.targets]
    def flush(self):        [t.flush()      for t in self.targets]

def _global_experiments_append(line: str):
    """Aggiunge una riga all’indice globale `experiments.md`."""
    exp_md = EXP_ROOT / "experiments.md"
    with exp_md.open("a") as f:
        f.write(line.rstrip() + "\n")

# ──────────────────────────────────────────────────────────────────────── #
# Path builders & artefact checks                                         #
# ──────────────────────────────────────────────────────────────────────── #
def _paths(model_name: str) -> dict[str, Path]:
    """Restituisce tutti i path (dir/log/ckpt/features/clf) per un modello."""
    mdir = EXP_BASE / model_name
    mdir.mkdir(parents=True, exist_ok=True)
    return {
        "dir"      : mdir,
        "log"      : mdir / f"log_{EXP_CODE}.md",
        "features" : mdir / f"{model_name}_features.pt",
        "clf"      : mdir / f"{model_name}_classifier.joblib",
        "ckpt_pref": f"{model_name}_epoch",        # usato da save_checkpoint
    }

def _completed(paths: dict, is_ssl: bool) -> bool:
    """True se TUTTI gli artefatti richiesti sono già presenti."""
    ckpt_ok = get_latest_checkpoint(paths["dir"], prefix=paths["ckpt_pref"]) is not None
    if not ckpt_ok:
        return False
    if is_ssl:
        return paths["features"].exists() and paths["clf"].exists()
    return True

# ──────────────────────────────────────────────────────────────────────── #
# Selezione modelli & trainer helpers                                     #
# ──────────────────────────────────────────────────────────────────────── #
def _select_models(cfg: dict) -> dict[str, dict]:
    sel = cfg.get("run_model", "all").lower()
    return {
        n: c for n, c in cfg["models"].items()
        if sel in ("all", n) or
           (sel == "sl"  and c.get("type") == "sl") or
           (sel == "ssl" and c.get("type") == "ssl")
    }

def _init_trainer(name: str, m_cfg: dict, data_cfg: dict, ckpt_dir: Path):
    Trainer = TRAINER_REGISTRY[name]
    tr = Trainer(m_cfg, data_cfg)
    tr.ckpt_dir = ckpt_dir         # salva i .pt qui
    return tr

# ──────────────────────────────────────────────────────────────────────── #
# Training / resume / artefacts                                           #
# ──────────────────────────────────────────────────────────────────────── #
def _run_full_training(trainer, epochs: int):
    """Loop di training (identico al codice esistente, solo incapsulato)."""
    has_val = hasattr(trainer, "validate_epoch")
    total_batches = getattr(trainer, "batches_train", None)
    if total_batches is None:
        try: total_batches = len(trainer.train_loader)
        except TypeError: total_batches = None

    for epoch in range(1, epochs + 1):
        t0 = time.time(); run_loss = run_corr = seen = 0
        print(f"--- Epoch {epoch}/{epochs} ---")
        for i, batch in enumerate(trainer.train_loader, 1):
            sig = inspect.signature(trainer.train_step)
            res = trainer.train_step(batch) if len(sig.parameters) == 1 \
                  else trainer.train_step(*batch)
            if len(res) == 4: _, loss, corr, bs = res
            else:             loss, bs = res; corr = 0
            run_loss += loss * bs; run_corr += corr; seen += bs

            # progress bar
            if total_batches:
                pct = (i/total_batches)*100
                eta = ((time.time()-t0)/i)*(total_batches-i)
                msg = (f"  Batch {i}/{total_batches} ({pct:.1f}%) | "
                       f"Loss: {run_loss/seen:.4f}")
                if has_val: msg += f" | Acc: {run_corr/seen:.3f}"
                msg += f" | Elapsed: {time.time()-t0:.1f}s | ETA: {eta:.1f}s"
            else:
                msg = f"  Batch {i} | Loss: {run_loss/seen:.4f}"
                if has_val: msg += f" | Acc: {run_corr/seen:.3f}"
                msg += f" | Elapsed: {time.time()-t0:.1f}s"
            print(msg)

        if has_val:
            v_loss, v_acc = trainer.validate_epoch()
            trainer.post_epoch(epoch, v_acc)
            print(f"Val -> Loss: {v_loss:.4f} | Acc: {v_acc:.3f}")
        else:
            trainer.post_epoch(epoch, run_loss/seen)

        print(f"Epoch {epoch} completed in {time.time()-t0:.1f}s\n")

def _resume_or_train(trainer, paths: dict, epochs: int):
    ckpt = get_latest_checkpoint(paths["dir"], prefix=paths["ckpt_pref"])
    if ckpt:
        print(f"⏩ Resume da {ckpt.name}")
        load_checkpoint(
            ckpt,
            model=torch.nn.Sequential(
                trainer.encoder,
                getattr(trainer, "projector", torch.nn.Identity())
            )
        )
    else:
        _run_full_training(trainer, epochs)

def _ensure_ssl_artifacts(trainer, paths: dict):
    if not paths["features"].exists():
        trainer.extract_features_to(str(paths["features"]))
    if not paths["clf"].exists():
        train_classifier(str(paths["features"]), str(paths["clf"]))

# %%
# %% -------------------------------------------------------------------- #
# Cell 7 – Modular Launch & Auto-Recover                                  #
# ----------------------------------------------------------------------- #
def launch_training(cfg: dict) -> None:
    """Lancia (o recupera) tutti i modelli selezionati."""
    for name, m_cfg in _select_models(cfg).items():
        paths   = _paths(name)
        is_ssl  = m_cfg.get("type") == "ssl"
        epochs  = int(m_cfg["training"]["epochs"])

        # ---------- logging (append) ----------------------------------- #
        with open(paths["log"], "a") as lf, \
             contextlib.redirect_stdout(_Tee(sys.stdout, lf)), \
             contextlib.redirect_stderr(_Tee(sys.stderr, lf)):

            if _completed(paths, is_ssl):
                print(f"✅ Artefatti completi per '{name}' – skip\n")
                continue

            trainer = _init_trainer(name, m_cfg, cfg["data"], paths["dir"])
            print(f"Device: {trainer.device} 🚀  Starting training for '{name}'")
            print(f"→ Model config: {m_cfg}\n")

            _resume_or_train(trainer, paths, epochs)

            if is_ssl:
                _ensure_ssl_artifacts(trainer, paths)

            # Aggiorna indice globale
            last_ckpt = get_latest_checkpoint(paths["dir"], prefix=paths["ckpt_pref"])
            rel = last_ckpt.relative_to(EXP_ROOT) if last_ckpt else "-"
            _global_experiments_append(f"| {EXP_CODE} | {name} | {epochs} | {rel} |")

# 🚀 Avvio immediato
launch_training(cfg)


