# %%
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

!pip install --quiet torch torchvision webdataset tqdm pillow scikit-learn joblib


# %%
from tqdm import tqdm
import csv, logging, joblib, sys, importlib, yaml, torch, numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

# 📁 Configurazione percorso progetto
config_path = Path('/content/drive/MyDrive/ColabNotebooks/wsi-ssrl-rcc_project/config/training.yaml')
with config_path.open('r') as f:
    cfg = yaml.safe_load(f)

colab_root = Path(cfg['env_paths']['colab'])
local_root = Path(cfg['env_paths']['local'])
PROJECT_ROOT = colab_root if colab_root.exists() else local_root

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


trainer_modules = [
    "trainers.simclr",
    "trainers.moco_v2",
    "trainers.rotation",
    #"trainers.jigsaw",
    "trainers.jepa",
    "trainers.supervised",
    "trainers.transfer",
]
for m in trainer_modules:
    if m in sys.modules:
        importlib.reload(sys.modules[m])
    else:
        importlib.import_module(m)
from utils.training_utils import TRAINER_REGISTRY, load_checkpoint
# Normalizza i path dei dati
for split in ['train','val','test']:
    rel = cfg['data'].get(split)
    if rel:
        cfg['data'][split] = str(PROJECT_ROOT / rel)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EVAL")
yaml_exp = cfg.get("exp_code", "")
if yaml_exp:
    EXP_CODE = yaml_exp
else:
  EXP_CODE = "error"
# 📁 Setup esperimento specifico
experiment_dir = PROJECT_ROOT / "data/processed/dataset_9f30917e/experiments" / EXP_CODE
print(f"📁 Esperimento in {experiment_dir}")


# %%
# ─── Helpers ───────────────────────────────────────────────────────────────
def extract_patient_id(key: str) -> str:
    for p in key.split("_"):
        if p.startswith("HP") or p.startswith("H"):
            return p
    return "UNKNOWN"

def extract_label_from_key(key: str) -> str:
    return "not_tumor" if key.startswith("not_tumor") else key.split("_")[0]

def compute_metrics(keys, y_pred, le):
    # ground-truth per paziente
    all_labels = defaultdict(list)
    for k in keys:
        all_labels[extract_patient_id(k)].append(extract_label_from_key(k))
    true_labels = {}
    for pid, labs in all_labels.items():
        tumor = [l for l in labs if l!="not_tumor"]
        if len(set(tumor))==1:
            true_labels[pid]=tumor[0]
    # votazioni
    preds = defaultdict(list)
    for k,p in zip(keys, y_pred):
        pid=extract_patient_id(k)
        if le.classes_[p]!="not_tumor":
            preds[pid].append(p)
    y_true, y_maj, valid = [], [], []
    for pid,votes in preds.items():
        if pid in true_labels and votes:
            gt = true_labels[pid]
            maj = Counter(votes).most_common(1)[0][0]
            y_true.append(le.transform([gt])[0])
            y_maj.append(maj)
            valid.append(pid)
    if not y_true:
        raise RuntimeError("Nessun paziente valutabile")
    report = classification_report(y_true, y_maj, target_names=[c for c in le.classes_ if c!="not_tumor"])
    cm     = confusion_matrix(y_true, y_maj)
    acc    = np.mean(np.array(y_true)==np.array(y_maj))
    f1     = f1_score(y_true, y_maj, average="macro")
    prec   = precision_score(y_true, y_maj, average="macro")
    rec    = recall_score(y_true, y_maj, average="macro")
    return dict(
        y_true=y_true, y_maj=y_maj, valid=valid,
        report=report, cm=cm,
        metrics=(acc,f1,prec,rec),
    )

def write_md_log(save_dir: Path, model_name: str, cm, report: str, valid: list[str], metrics: tuple[float,float,float,float], y_true: list[int], y_maj: list[int], le):
    acc,f1,prec,rec = metrics
    md = save_dir/"evals_log.md"
    with open(md, "w") as f:
        f.write(f"# 🧠 Modello: {model_name}\n\n")
        f.write("## 📊 Risultati Majority Voting (paziente-level)\n```text\n")
        f.write(report)
        f.write("\n```\n\n")
        f.write("## 📉 Confusion Matrix\n```text\n")
        f.write(str(cm))
        f.write("\n```\n\n")
        f.write(f"✅ Totale pazienti classificati: {len(valid)}\n\n")
        # sezione per-paziente
        f.write("## 🧾 Predizione per paziente\n```text\n")
        for pid, t_enc, p_enc in zip(valid, y_true, y_maj):
            true_lbl = le.inverse_transform([t_enc])[0]
            pred_lbl = le.inverse_transform([p_enc])[0]
            f.write(f"Paziente {pid}: predetto = {pred_lbl} | reale = {true_lbl}\n")
        f.write("```\n\n")
        f.write("## 📈 Metriche sintetiche\n")
        f.write(f"- Accuracy        : {acc:.4f}\n")
        f.write(f"- Macro F1        : {f1:.4f}\n")
        f.write(f"- Macro Precision : {prec:.4f}\n")
        f.write(f"- Macro Recall    : {rec:.4f}\n")

def append_summary_csv(common_csv:Path, model_name:str, metrics):
    if not common_csv.exists():
        with open(common_csv, "w", newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["Model","Accuracy","Macro F1","Macro Precision","Macro Recall","N_Patients"])
    acc,f1,prec,rec = metrics
    # N_Patients è già nel report: la lunghezza di valid
    writer=csv.writer(open(common_csv,"a",newline=""))
    writer.writerow([model_name,acc,f1,prec,rec])


# %%
# ─── Evaluation ─────────────────────────────────────────────────────────────
from utils.training_utils import TRAINER_REGISTRY, load_checkpoint
from trainers.extract_features import extract_features
import webdataset as wds
import torchvision.transforms as T
from PIL import Image
import joblib
import torch
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from pathlib import Path
import numpy as np

def make_loader(test_path: str, batch_size: int = 64):
    ds = (
        wds.WebDataset(test_path,
                       shardshuffle=False,
                       handler=wds.warn_and_continue,
                       empty_check=False)
         .decode("pil")
         .map(lambda s: {
             "img": T.ToTensor()(
                 next((v for v in s.values() if isinstance(v, Image.Image)), None)
                 .convert("RGB")
             ),
             "key": s["__key__"] + "." + next((k for k in s.keys() if k.endswith(".jpg")), "")
         })
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

def evaluate_model(trainer, test_path: str, save_dir: Path, model_name: str, ssl: bool):
    eval_md = save_dir / "evals_log.md"
    if eval_md.exists():
        logger.info(f"📄 evals_log.md già presente in {save_dir.name}, skip evaluation.")
        print(f"\n📄 evals_log.md già presente per {model_name} ===")
        return
    # 1) checkpoint migliore
    ckpt = sorted(save_dir.glob("*Trainer_best_epoch*.pt"))[-1]

    # 2) carica SOLO l'encoder per i SSL
    if ssl:
        sd = torch.load(ckpt, map_location="cpu")["model_state_dict"]
        # estrai solo i parametri che iniziano con "0."
        enc_sd = {k.split(".",1)[1]: v for k, v in sd.items() if k.startswith("0.")}
        # scegli il modulo encoder corretto
        if hasattr(trainer, "encoder_q"):
            trainer.encoder_q.load_state_dict(enc_sd)
            feat_mod = trainer.encoder_q
        elif hasattr(trainer, "encoder"):
            trainer.encoder.load_state_dict(enc_sd)
            feat_mod = trainer.encoder
        else:
            # Rotation ha solo .model
            load_checkpoint(ckpt, model=trainer.model)
            feat_mod = trainer.model
    else:
        # supervised / transfer
        load_checkpoint(ckpt, model=trainer.model)
        feat_mod = trainer.model

    feat_mod = feat_mod.to(trainer.device).eval()

    # 3) classificatore o label-encoder
    if ssl:
        clf_data = joblib.load(save_dir / f"{model_name}_classifier.joblib")
        clf, le, scaler = (
            clf_data["model"],
            clf_data["label_encoder"],
            clf_data.get("scaler", None),
        )
    else:
        le, scaler = trainer.label_encoder, None

    # 4) DataLoader
    loader = make_loader(test_path)

    # 5) estrai predizioni
    if ssl:
        feats = extract_features(feat_mod, loader, trainer.device)
        X, keys = feats["features"].numpy(), feats["keys"]
        if scaler: X = scaler.transform(X)
        y_pred = clf.predict(X)

        # patch-level report
        true_patch = [extract_label_from_key(k) for k in keys]
        pred_patch = [le.classes_[p] for p in y_pred]
        print(f"\n=== Patch-level report per {model_name} ===")
        print(classification_report(
            true_patch, pred_patch,
            labels=[c for c in le.classes_ if c!="not_tumor"],
            target_names=[c for c in le.classes_ if c!="not_tumor"]
        ))

    else:
        # streaming per-paziente
        keys, y_pred = [], []
        true_labels  = {}
        vote_counts  = defaultdict(Counter)
        with torch.no_grad():
            for batch in loader:
                imgs = batch["img"].to(trainer.device)
                ks   = batch["key"]
                logits = feat_mod(imgs)
                preds  = torch.argmax(logits, dim=1).cpu().tolist()
                for k,p in zip(ks, preds):
                    keys.append(k); y_pred.append(p)
                    pid = extract_patient_id(k)
                    lbl = extract_label_from_key(k)
                    if lbl!="not_tumor" and pid not in true_labels:
                        true_labels[pid] = lbl
                    if le.inverse_transform([p])[0]!="not_tumor":
                        vote_counts[pid][p] += 1

    # 6) metriche & log per tutti i modelli
    out = compute_metrics(keys, y_pred, le)
    write_md_log(
        save_dir, model_name,
        out["cm"], out["report"],
        out["valid"], out["metrics"],
        out["y_true"], out["y_maj"], le
    )
    append_summary_csv(
        save_dir.parent / "evaluation_summary_all_models.csv",
        model_name, out["metrics"]
    )
    print(f"✔️ Finished eval for {model_name}")


# %%
# ─── Main loop ─────────────────────────────────────────────────────────────
run_model = cfg.get("run_model","all").lower()
models = cfg["models"].items() if run_model=="all" else [(run_model,cfg["models"][run_model])]
for name, m_cfg in models:
    if name not in TRAINER_REGISTRY:
        logger.warning(f"Trainer '{name}' non trovato, skip.")
        continue
    trainer = TRAINER_REGISTRY[name](m_cfg, cfg["data"])
    test_p = str(cfg["data"]["test"])
    model_dir = experiment_dir/name
    if not model_dir.exists():
        logger.warning(f"Dir {model_dir} mancante, skip.")
        continue
    is_ssl = (name not in ["supervised","transfer"])
    evaluate_model(trainer, test_p, model_dir, name, ssl=is_ssl)


