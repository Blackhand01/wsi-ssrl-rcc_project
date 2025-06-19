# src/trainers/train_classifier.py

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
from collections import Counter

def train_classifier(features_path: str, output_model: str = "classifier.joblib"):
    """
    Trains a classifier using features extracted from SSL models,
    by inferring labels directly from patch filenames.
    """

    # Load features and keys
    data = torch.load(features_path)
    features = data["features"].numpy()
    keys = data["keys"]
    print(f"✅ Loaded {len(keys)} keys and {features.shape} features")

    # Extract label from prefix of the key
    def extract_label(key: str) -> str | None:
        """
        Parses the label from the key, treating 'not_tumor' as a distinct class.
        Example: 'not_tumor_HP12.390_000001.jpg' → 'not_tumor'
        """
        parts = key.split("_")
        if len(parts) >= 2 and parts[0] == "not" and parts[1] == "tumor":
            return "not_tumor"
        else:
            return parts[0] if len(parts) >= 1 else None


    labels = [extract_label(k) for k in keys]
    valid_mask = [lbl is not None for lbl in labels]

    features = features[valid_mask]
    labels = np.array([lbl for lbl in labels if lbl is not None])

    print("📊 Class distribution:")
    print(Counter(labels))
    print(f"✅ Filtered dataset: {features.shape[0]} samples")

    if features.shape[0] < 10:
        raise ValueError("❌ Too few valid samples.")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=0.2, stratify=y, random_state=42
    )

    # Classifier
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(X_train, y_train)

    # Eval
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    Path(output_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": le}, output_model)
    print(f"💾 Classifier saved to {output_model}")
