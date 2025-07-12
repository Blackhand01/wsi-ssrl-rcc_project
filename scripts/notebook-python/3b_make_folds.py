# %% [markdown]
# I N_FOLD (N_FOLD = 4) fold non sono 4 “versioni” alternative del tuo modello da confrontare e poi scegliere la migliore, ma piuttosto  4 partizioni **complementari** dello stesso dataset che usi **tutte** per:
# 
# 1. **Stimare la generalizzazione in modo solido**
# 
#    * In ciascun fold tieni fuori \~ 1/N_FOLD dei pazienti per la validazione e trai indicazioni su come andrà il modello su nuovi pazienti.
#    * Alla fine riporti la **media** (± deviazione) delle metriche sui N_FOLD fold: quella è la stima *complessiva* delle prestazioni, non scegli “il migliore” perché non è un’ipotesi di ottimizzazione ma di valutazione.
# 
# 2. **Tuning iper‐parametri senza over-fit**
# 
#    * Se vuoi ottimizzare learning rate, weight-decay o temperature, misuri l’accuracy (e le altre metriche) su ciascun fold per una configurazione, poi scegli la configurazione che **in media** lavora meglio.
#    * Non scegli “il fold migliore” ma la **combo di iper-parametri** che massimizza la media cross-fold.
# 
# 3. **Identificare instabilità o gruppi “difficili”**
# 
#    * Se vedi che in un fold particular (cioè su quei 6 pazienti di validazione) il modello crolla rispetto agli altri, capisci che c’è qualcosa di particolare in quei soggetti: magari artefatti, stain diversi, classe sbilanciata.
#    * Ti aiuta a diagnosticare problemi di bias su specifici pazienti o sottogruppi.
# 
# ---
# 
# ### Cosa ottieni praticamente
# 
# * **Metrica finale** → ad esempio “accuracy paziente‐level = 0.82 ± 0.05” (media e std)
# * **Curva di training/val per ogni fold** → ti mostra over-fit o under-fit
# * **Heatmap “fold-wise”** → eventuali casi di failure diagnostico
# 
# ---
# 
# ### In sintesi
# 
# | Scopo dei fold      | Come li usi                                                     | Risultato che migliora la tua analisi                                      |
# | ------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
# | **Valutazione**     | addestri N_FOLD volte (train su 4/N_FOLD, val su 1/N_FOLD) e calcoli media/std | Stima di performance **più affidabile** e non “pazzo” per un singolo split |
# | **Tuning**          | cerchi gli iper-param che massimizzano la media cross-fold      | Parametri scelti **in modo robusto**, meno over-fit                        |
# | **Diagnosi errori** | identifichi fold “deboli” con metriche drammatiche              | Capisci se c’è un sottoinsieme di pazienti problematico                    |
# | **Reporting**       | fornisci intervallo di confidenza (es. 95%)                     | Risultati **clinicamente credibili** per l’ospedale                        |
# 
# Non “scegli” un fold migliore: **usali tutti** per capire come si comporta il tuo SSL+probe in scenari diversi e per **tirare fuori** una metrica unica (media±std) che sia solida e ripetibile.
# 

# %%
# Cell 1 – Environment Setup & Dependencies
import os, sys, subprocess
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
print(f"📁 [DEBUG] PROJECT_ROOT → {PROJECT_ROOT}")

# %%
# ------------------------------------------------------------------
# Cell 2 – Generazione e stampa distribuzione dei fold patient-level
# ------------------------------------------------------------------

import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Directory di output
OUT_DIR = PROJECT_ROOT / f"data/processed/dataset_{DATASET_ID}/folds_metadata"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Numero di fold downstream (escludendo il test hold-out)
N_FOLD = 4

# 1) Carica tutto il DataFrame
df = pd.read_parquet(PARQUET_PATH)

# 2) Fold0: ricicla split originario
df_train0 = df[df['split']=='train'].drop(columns='split').reset_index(drop=True)
df_val0   = df[df['split']=='val']  .drop(columns='split').reset_index(drop=True)
df_test   = df[df['split']=='test'] .drop(columns='split').reset_index(drop=True)

df_train0.to_parquet(OUT_DIR/'patch_df_train_fold0.parquet',      index=False)
df_val0  .to_parquet(OUT_DIR/'patch_df_val_fold0.parquet',        index=False)
df_test  .to_parquet(OUT_DIR/'patch_df_test_holdout.parquet',     index=False)

print(f"✔️ Fold0 salvato: train={len(df_train0)} patch, val={len(df_val0)}, test_holdout={len(df_test)}")

# Funzione di verifica distribuzione
ALL_CLASSES = sorted(df['subtype'].unique())
def print_distribution(df_part, part_name, fold_idx):
    pid_counts   = df_part.groupby('subtype')['patient_id'].nunique().to_dict()
    patch_counts = df_part['subtype'].value_counts().to_dict()
    print(f"\n📋 Fold{fold_idx} {part_name}")
    print("  pazienti per classe:", {c: pid_counts.get(c,0) for c in ALL_CLASSES})
    print("  patch per classe:   ", {c: patch_counts.get(c,0) for c in ALL_CLASSES})

# Verifica Fold0
print_distribution(df_train0, "train", 0)
print_distribution(df_val0,   "val",   0)

# 3) Prepara df_folds escludendo il test originario
df_folds = df[df['split'] != 'test'].drop(columns='split').reset_index(drop=True)

# 4) Costruisci DataFrame pazienti unici con etichetta principale
patient_df = (
    df_folds
    .groupby('patient_id')['subtype']
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

# Per controllo: tiene traccia dei pazienti visti
fold_pid_sets = {}

# 5) StratifiedKFold su pazienti
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=42)
for fold_idx, (train_pid_idx, val_pid_idx) in enumerate(skf.split(patient_df, patient_df['subtype']), start=1):
    train_pids = patient_df.loc[train_pid_idx, 'patient_id']
    val_pids   = patient_df.loc[val_pid_idx,   'patient_id']

    df_tr = df_folds[df_folds['patient_id'].isin(train_pids)].reset_index(drop=True)
    df_va = df_folds[df_folds['patient_id'].isin(val_pids)]  .reset_index(drop=True)

    df_tr.to_parquet(OUT_DIR/f"patch_df_train_fold{fold_idx}.parquet", index=False)
    df_va.to_parquet(OUT_DIR/f"patch_df_val_fold{fold_idx}.parquet",   index=False)

    print(f"✔️ Fold{fold_idx} salvato: train={len(df_tr)} patch, val={len(df_va)} patch")
    print_distribution(df_tr, "train", fold_idx)
    print_distribution(df_va,   "val",   fold_idx)

    # Salva set di pazienti per controllo
    fold_pid_sets[f"fold{fold_idx}_train"] = set(train_pids)
    fold_pid_sets[f"fold{fold_idx}_val"] = set(val_pids)

# 6) Verifica incrociata: i pazienti cambiano nei fold?
print("\n🔎 Verifica che i pazienti nei fold siano differenti:")
for i in range(1, N_FOLD):
    for j in range(i+1, N_FOLD+1):
        inters_train = fold_pid_sets[f"fold{i}_train"] & fold_pid_sets[f"fold{j}_train"]
        inters_val   = fold_pid_sets[f"fold{i}_val"]   & fold_pid_sets[f"fold{j}_val"]
        print(f"  • Fold{i} vs Fold{j} → comuni (train): {len(inters_train)} | comuni (val): {len(inters_val)}")

print("\n🎉 Generazione & verifica distribuzione completata per tutti i fold!")


# %%
# ------------------------------------------------------------------
# Cell 3 – Salvataggio riepilogo distribuzione e pazienti per fold
# ------------------------------------------------------------------

folds_md_path = OUT_DIR / "folds_summary.md"
folds_csv_path = OUT_DIR / "folds_patients.csv"

lines_md = []
lines_csv = ["fold,split,n_patients,patient_ids"]

for fold_idx in range(N_FOLD + 1):
    fold_tag = f"fold{fold_idx}"

    # Carica i file del fold
    df_tr = pd.read_parquet(OUT_DIR / f"patch_df_train_{fold_tag}.parquet")
    df_va = pd.read_parquet(OUT_DIR / f"patch_df_val_{fold_tag}.parquet")

    lines_md.append(f"✔️ Fold{fold_idx} salvato: train={len(df_tr)} patch, val={len(df_va)} patch\n")

    for split_name, df_part in [("train", df_tr), ("val", df_va)]:
        pid_counts   = df_part.groupby('subtype')['patient_id'].nunique().to_dict()
        patch_counts = df_part['subtype'].value_counts().to_dict()

        lines_md.append(f"📋 Fold{fold_idx} {split_name}")
        lines_md.append(f"  pazienti per classe: {pid_counts}")
        lines_md.append(f"  patch per classe:    {patch_counts}\n")

        # Riga CSV per questo split
        patient_ids = sorted(df_part['patient_id'].unique())
        lines_csv.append(f"{fold_tag},{split_name},{len(patient_ids)},{';'.join(patient_ids)}")

# Verifica incrociata dei pazienti tra i fold
lines_md.append("🔎 Verifica che i pazienti nei fold siano differenti:")
for i in range(1, N_FOLD):
    for j in range(i+1, N_FOLD+1):
        pid_i_tr = set(pd.read_parquet(OUT_DIR / f"patch_df_train_fold{i}.parquet")['patient_id'].unique())
        pid_j_tr = set(pd.read_parquet(OUT_DIR / f"patch_df_train_fold{j}.parquet")['patient_id'].unique())
        pid_i_val = set(pd.read_parquet(OUT_DIR / f"patch_df_val_fold{i}.parquet")['patient_id'].unique())
        pid_j_val = set(pd.read_parquet(OUT_DIR / f"patch_df_val_fold{j}.parquet")['patient_id'].unique())

        train_inter = pid_i_tr & pid_j_tr
        val_inter   = pid_i_val & pid_j_val
        lines_md.append(f"  • Fold{i} vs Fold{j} → comuni (train): {len(train_inter)} | comuni (val): {len(val_inter)}")

# Scrivi file Markdown
with open(folds_md_path, "w") as f:
    f.write("\n".join(lines_md))

# Scrivi file CSV
with open(folds_csv_path, "w") as f:
    f.write("\n".join(lines_csv))

print(f"📝 File riepilogo scritto in: {folds_md_path}")
print(f"📄 CSV pazienti scritto in:   {folds_csv_path}")


# %%
# ============================================================
#  create webdatasets folds
# ============================================================
!pip install --quiet webdataset tqdm

import tarfile, shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import webdataset as wds

# --- PATHS ---------------------------------------------------
DATASET_ID  = "9f30917e"
PR          = Path("/content/drive/MyDrive/ColabNotebooks/wsi-ssrl-rcc_project")
DS_DIR      = PR / f"data/processed/dataset_{DATASET_ID}"
FOLDS_DIR   = DS_DIR / "folds_metadata"
WEBDIR      = DS_DIR / "webdataset"
WEBDIR.mkdir(parents=True, exist_ok=True)

CANON_TRAIN = WEBDIR / "train/patches-0000.tar"
CANON_VAL   = WEBDIR / "val/patches-0000.tar"
CANON_TEST  = WEBDIR / "test/patches-0000.tar"

# ------------------------------------------------------------------
# 0) Fold0 = copia grezza (+ holdout)
# ------------------------------------------------------------------
for split, src in (("train", CANON_TRAIN), ("val", CANON_VAL)):
    dst = WEBDIR / f"fold0/{split}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst / "patches-0000.tar")
shutil.copy2(CANON_TEST, WEBDIR / "test_holdout.tar")
print("✔️  Copiato fold0 + test_holdout")

# ------------------------------------------------------------------
# 1) Indicizza tutti e tre i tar canonici
# ------------------------------------------------------------------
index = {}      # key -> (tar_path, member_name)
tar_cache = {}

def build_index(tar_path):
    t = tarfile.open(tar_path, "r")
    tar_cache[tar_path] = t
    for m in t.getmembers():
        if m.isfile() and m.name.endswith(".jpg"):
            index[Path(m.name).stem] = (tar_path, m.name)

for tp in (CANON_TRAIN, CANON_VAL, CANON_TEST):
    build_index(tp)
print(f"🔍 Indicizzati {len(index)} patch totali")

def get_bytes(key):
    tar_path, member = index[key]
    return tar_cache[tar_path].extractfile(member).read()

# ------------------------------------------------------------------
# 2) Funzione helper con filtraggio delle chiavi mancanti
# ------------------------------------------------------------------
def parquet_to_tar(parq_path: Path, out_tar: Path, desc: str):
    df = pd.read_parquet(parq_path)
    # calcola le chiavi clean
    df["key"] = df.apply(lambda r: f"{r.subtype}_{r.patient_id.replace('.','')}_{int(r.x)}_{int(r.y)}", axis=1)
    # filtra solo quelle presenti
    mask = df["key"].isin(index)
    missing = (~mask).sum()
    if missing:
        print(f"⚠️  {desc}: {missing} patch scartate perché non trovate nei tar originali")
    df = df[mask].reset_index(drop=True)

    # scrivi il tar
    with wds.TarWriter(str(out_tar)) as sink:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
            img_bytes = get_bytes(row["key"])
            sink.write({
                "__key__": row["key"],
                "jpg": img_bytes,
                "cls": row.subtype.encode("utf-8"),
            })

# ------------------------------------------------------------------
# 3) Genera fold1‥fold3
# ------------------------------------------------------------------
for fold in range(1, 4):
    for split in ("train", "val"):
        parq = FOLDS_DIR / f"patch_df_{split}_fold{fold}.parquet"
        out_dir = WEBDIR / f"fold{fold}/{split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_tar = out_dir / "patches-0000.tar"
        parquet_to_tar(parq, out_tar, f"fold{fold}-{split}")

print("🎉  WebDataset per fold1-3 costruiti con successo")



