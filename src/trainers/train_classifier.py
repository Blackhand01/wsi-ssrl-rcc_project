# src/train_classifier.py

import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

def train_classifier(features_path: str, parquet_path: str, output_model: str = "classifier.joblib"):
    """
    Allena un classificatore sulle feature estratte dai modelli SSL.

    Args:
        features_path (str): Path al file .pt contenente features e keys.
        parquet_path (str): Path al Parquet contenente mapping patch → label.
        output_model (str): Path per salvare il modello allenato.
    """

    # Carica features
    data = torch.load(features_path)
    features = data["features"].numpy()
    keys = data["keys"]

    print(f"✅ Caricate {len(keys)} keys e {features.shape} features")

    # Carica parquet
    df = pd.read_parquet(parquet_path)

    # Costruisci mapping: patient_id -> label
    patient_to_label = dict(zip(df["patient_id"], df["subtype"]))

    # Associa label alle keys (es. ccRCC_H19.754_000104.jpg → ccRCC)
    labels = []
    for k in keys:
        # Estrai patient_id dal nome della patch (es. ccRCC_HP19.754_000104.jpg)
        parts = Path(k).stem.split("_")
        patient_id = parts[1] if len(parts) >= 2 else None
        label = patient_to_label.get(patient_id, None)
        labels.append(label)

    labels = np.array(labels)
    valid_mask = labels != None
    features = features[valid_mask]
    labels = labels[valid_mask]

    print(f"✅ Dataset filtrato: {features.shape[0]} samples")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, stratify=y, random_state=42
    )

    # Classifier
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Eval
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Salva modello
    joblib.dump({"model": clf, "label_encoder": le}, output_model)
    print(f"💾 Classifier salvato in {output_model}")



######Esenpio di utilizzo######
# Dopo aver estratto le feature
# train_classifier(
#     features_path="simclr_val_features.pt",
#     parquet_path="/Users/mimmo/Desktop/mimmo/MLA/project_FP03/wsi-ssrl-rcc_project/data/processed/patch_df_2500.parquet",
#     output_model="simclr_classifier.joblib"
# )
# ##########################################