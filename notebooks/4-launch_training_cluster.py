#!/usr/bin/env python
# coding: utf-8

import os
import sys
import subprocess
import traceback
import shutil
import yaml
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv, find_dotenv
import importlib.util
import datetime
import contextlib
import time
import inspect
import torch
from utils.training_utils import (
    TRAINER_REGISTRY,
    get_latest_checkpoint,
    load_checkpoint,
)
from trainers.train_classifier import train_classifier

# Environment Setup & Dependencies
print("📦 [DEBUG] Avvio configurazione ambiente...")

IN_COLAB = Path("/content").exists()
if IN_COLAB:
    print("📍 [DEBUG] Ambiente Google Colab rilevato.")
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
else:
    print("💻 [DEBUG] Ambiente locale rilevato (VSCode o simile).")

ENV_PATHS = {
    "colab": "/content/drive/MyDrive/ColabNotebooks/wsi-ssrl-rcc_project",
    "local": "/Users/stefanoroybisignano/Desktop/MLA/project/wsi-ssrl-rcc_project",
}
PROJECT_ROOT = Path(ENV_PATHS["colab" if IN_COLAB else "local"]).resolve()
sys.path.append(str(PROJECT_ROOT / "src"))
print(f"📁 [DEBUG] PROJECT_ROOT → {PROJECT_ROOT}")

def _pip_install(pkgs):
    import importlib.util
    missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
    if missing:
        print(f"🔧 [DEBUG] Installazione pacchetti mancanti: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
    else:
        print("✅ [DEBUG] Tutti i pacchetti richiesti sono già installati.")

_pip_install(["torch", "torchvision", "webdataset", "tqdm", "pillow", "pyyaml", "joblib"])

DATA_TARBALL = PROJECT_ROOT / "data" / "processed"

# SLURM Submission (modulare)
def _detect_env() -> tuple[bool, bool]:
    in_colab = Path("/content").exists()
    in_vscode = not in_colab and bool(os.environ.get("VSCODE_PID"))
    print(f"🚀 In Colab={in_colab}, VSCode={in_vscode}")
    return in_colab, in_vscode

def _load_env_vars() -> dict:
    dotenv_path = find_dotenv()
    if not dotenv_path:
        raise FileNotFoundError("❌ .env non trovato nella root del progetto.")
    load_dotenv(dotenv_path, override=True)

    env = {
        "USER": os.getenv("CLUSTER_USER"),
        "HOST": os.getenv("CLUSTER_HOST"),
        "BASE": os.getenv("REMOTE_BASE_PATH"),
        "PW": os.getenv("CLUSTER_PASSWORD", ""),
        "MAIL": os.getenv("RESPONSABILE_EMAIL", os.getenv("MEMBER_EMAIL")),
        "PART": os.getenv("SBATCH_PARTITION", "global"),
        "MOD": os.getenv("SBATCH_MODULE", "intel/python/3/2019.4.088"),
    }
    missing = [k for k, v in env.items() if k in ("USER", "HOST", "BASE") and not v]
    if missing:
        raise KeyError(f"🌱 Mancano variabili nel .env: {missing}")
    return env

def _ssh_cmd(env: dict, cmd: str) -> list[str]:
    base = []
    if env["PW"] and shutil.which("sshpass"):
        base += ["sshpass", "-p", env["PW"]]
    base += ["ssh", f"{env['USER']}@{env['HOST']}", cmd]
    return base

def _rsync_cmd(env: dict, src: str, dst: str, extra: list[str] | None = None) -> list[str]:
    cmd = ["rsync", "-avz"]
    if extra:
        cmd += extra
    if env["PW"] and shutil.which("sshpass"):
        cmd += ["-e", f"sshpass -p {env['PW']} ssh -o StrictHostKeyChecking=no"]
    cmd += [src, dst]
    return cmd

def build_sbatch_script(env: dict, script_path: Path):
    body = dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=rcc_ssrl_launch
        #SBATCH --ntasks=1
        #SBATCH --cpus-per-task=4
        #SBATCH --mem-per-cpu=4G
        #SBATCH --time=2:00:00
        #SBATCH --gres=gpu:1
        #SBATCH --partition={env['PART']}
        #SBATCH --output=%x_%j.out
        #SBATCH --error=%x_%j.err
        #SBATCH --mail-type=END,FAIL
        #SBATCH --mail-user={env['MAIL']}
        #SBATCH --workdir={env['BASE']}

        module purge
        module load {env['MOD']}

        cd {env['BASE']}
        python notebooks/4-launch_training_cluster.py --config config/training.yaml
    """)
    script_path.write_text(body)
    script_path.chmod(0o755)
    print(f"📝 sbatch script aggiornato: {script_path}")

def sync_project_code(env: dict):
    print("🔄 Sync codice progetto → cluster")
    subprocess.run(
        _rsync_cmd(env,
                   f"{PROJECT_ROOT}/",
                   f"{env['USER']}@{env['HOST']}:{env['BASE']}/",
                   ["--delete", "--exclude", ".git", "--exclude", "data/processed"]),
        check=True
    )

def upload_dataset_tar(env: dict, dataset_id: str):
    local_tar = PROJECT_ROOT / "data" / "processed" / f"{dataset_id}.tar.gz"
    remote_dir = f"{env['BASE']}/data/processed"
    remote_tar = f"{remote_dir}/{dataset_id}.tar.gz"
    remote_ds = f"{remote_dir}/{dataset_id}"

    if subprocess.run(_ssh_cmd(env, f"test -d {remote_ds}")).returncode == 0:
        print("📦 Dataset già presente sul cluster, skip upload")
        return

    if not local_tar.exists():
        print("🗜️  Creo tarball locale del dataset …")
        subprocess.run(
            ["tar", "-czf", str(local_tar), "-C",
             str(PROJECT_ROOT / "data" / "processed"), dataset_id],
            check=True
        )

    subprocess.run(_ssh_cmd(env, f"mkdir -p {remote_dir}"), check=True)

    print("🚚  Upload tarball …")
    subprocess.run(
        _rsync_cmd(env, str(local_tar),
                   f"{env['USER']}@{env['HOST']}:{remote_tar}"),
        check=True
    )

    print("📂  Estrazione sul cluster …")
    subprocess.run(_ssh_cmd(env,
        f"tar -xzf {remote_tar} -C {remote_dir} && rm -f {remote_tar}"),
        check=True
    )

def submit_job(env: dict, script_name: str):
    cmd = f"cd {env['BASE']} && sbatch {script_name}"
    if env["PW"] and shutil.which("sshpass"):
        output = subprocess.check_output(_ssh_cmd(env, cmd))
        print("📬 Job submit:", output.decode().strip())
    else:
        subprocess.run(_ssh_cmd(env, cmd), check=True)

IN_COLAB, IN_VSCODE = _detect_env()
if IN_VSCODE:
    try:
        env = _load_env_vars()

        cfg = yaml.safe_load((PROJECT_ROOT / "config" / "training.yaml").read_text())
        DATASET_ID = cfg["data"]["dataset_id"]

        subprocess.run(_ssh_cmd(env, f"mkdir -p {env['BASE']}"), check=True)

        SBATCH_LOCAL = PROJECT_ROOT / "hpc_submit.sh"
        build_sbatch_script(env, SBATCH_LOCAL)
        sync_project_code(env)
        upload_dataset_tar(env, DATASET_ID)
        submit_job(env, SBATCH_LOCAL.name)

    except Exception:
        print("❌ Submission fallita:")
        traceback.print_exc()
else:
    print("⚠️  Skip esecuzione locale.")

# Configuration & Directory Setup
yaml_exp = cfg.get("exp_code", "")
env_exp = os.environ.get("EXP_CODE", "")

if yaml_exp:
    EXP_CODE = yaml_exp
elif env_exp:
    EXP_CODE = env_exp
else:
    EXP_CODE = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

os.environ["EXP_CODE"] = EXP_CODE

print(f"[DEBUG] EXP_CODE → {EXP_CODE}")

cfg_path = PROJECT_ROOT / "config" / "training.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
DATASET_ID = cfg["data"]["dataset_id"]
cfg["experiment_code"] = EXP_CODE

for split in ("train", "val", "test"):
    rel = cfg["data"][split].format(dataset_id=DATASET_ID)
    abs_ = (PROJECT_ROOT / rel).resolve()
    cfg["data"][split] = str(abs_)
    print(f"[DEBUG] {split.upper()} → {abs_}")

EXP_ROOT = PROJECT_ROOT / "data" / "processed" / str(DATASET_ID)
EXP_BASE = EXP_ROOT / "experiments" / EXP_CODE
EXP_BASE.mkdir(parents=True, exist_ok=True)
(EXP_ROOT / "experiments.md").touch(exist_ok=True)

print(f"[DEBUG] EXP_BASE → {EXP_BASE}")

exp_yaml = EXP_BASE / f"training_{EXP_CODE}.yaml"
if not exp_yaml.exists():
    exp_yaml.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"[DEBUG] Scritto   {exp_yaml}")
else:
    print(f"[DEBUG] Config già presente → {exp_yaml}")

# Modular Launch & Auto-Recover
def launch_training(cfg: dict) -> None:
    for name, m_cfg in _select_models(cfg).items():
        paths = _paths(name)
        is_ssl = m_cfg.get("type") == "ssl"
        epochs = int(m_cfg["training"]["epochs"])

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

            last_ckpt = get_latest_checkpoint(paths["dir"], prefix=paths["ckpt_pref"])
            rel = last_ckpt.relative_to(EXP_ROOT) if last_ckpt else "-"
            _global_experiments_append(f"| {EXP_CODE} | {name} | {epochs} | {rel} |")

if IN_COLAB:
    launch_training(cfg)
else:
    print("⏩ Training delegato a SLURM: skip esecuzione locale.")

# Post-SLURM auto-download
if IN_VSCODE:
    load_dotenv(find_dotenv(), override=True)
    CLUSTER_USER = os.getenv("CLUSTER_USER")
    CLUSTER_HOST = os.getenv("CLUSTER_HOST")
    REMOTE_BASE = os.getenv("REMOTE_BASE_PATH")
    CLUSTER_PW = os.getenv("CLUSTER_PASSWORD", "")
    USE_SSHPASS = bool(CLUSTER_PW and shutil.which("sshpass"))

    def _rsync(src: str, dest: str):
        args = ["rsync", "-avz"]
        if USE_SSHPASS:
            sshpart = f"sshpass -p {CLUSTER_PW} ssh -o StrictHostKeyChecking=no"
            args += ["-e", sshpart]
        args += [src, dest]
        subprocess.run(args, check=True)

    dataset_id = cfg["data"]["dataset_id"]
    remote_exp = f"{REMOTE_BASE}/data/processed/{dataset_id}/experiments/{EXP_CODE}/"
    local_exp = EXP_ROOT / "experiments" / EXP_CODE
    local_exp.mkdir(parents=True, exist_ok=True)

    print(f"⬇️  Downloading artifacts of EXP_CODE={EXP_CODE} …")
    try:
        _rsync(f"{CLUSTER_USER}@{CLUSTER_HOST}:{remote_exp}", str(local_exp))
        print("✅  Artifacts downloaded in:", local_exp)
    except subprocess.CalledProcessError as e:
        print("⚠️  Download failed (job forse non ancora terminato):")
        print(e.stderr or "")
else:
    print("↩️  Download skipped: non siamo in VSCode/LEGION.")
