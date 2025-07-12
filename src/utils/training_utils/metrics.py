# metrics.py
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np
from typing import Any
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin

class TemperatureScaler(BaseEstimator, TransformerMixin):
    """
    Calibrates logits by scaling temperature to minimize Negative Log-Likelihood.
    """

    def __init__(self, init_temp: float = 1.0, bounds=(1e-2, 10.0)):
        self.init_temp = init_temp
        self.bounds = bounds
        self.temperature_ = init_temp

    def _nll(self, t: float, logits: np.ndarray, labels: np.ndarray) -> float:
        scaled = np.exp(logits / t)
        probs  = scaled / scaled.sum(axis=1, keepdims=True)
        idx    = np.arange(len(labels))
        return -np.mean(np.log(probs[idx, labels] + 1e-12))

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        res = minimize(
            fun=lambda x: self._nll(x[0], logits, labels),
            x0=[self.init_temp],
            bounds=[self.bounds],
            method="L-BFGS-B"
        )
        self.temperature_ = float(res.x[0]) if res.success else self.init_temp
        return self

    def transform_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.exp(logits / self.temperature_)
        return scaled / scaled.sum(axis=1, keepdims=True)

    def fit_transform(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return self.fit(logits, labels).transform_proba(logits)
    

def apply_temperature_scaling(
    logits: np.ndarray,
    labels: np.ndarray
) -> TemperatureScaler:
    scaler = TemperatureScaler().fit(logits, labels)
    return scaler



def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax over last dimension of logits.
    """
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

def compute_classification_metrics(
    keys: list[str],
    y_pred: np.ndarray,
    le,
    *,
    mc_stats: dict[str, float]
) -> dict:
    """
    Raccoglie tutte le metriche principali per classificazione:
      - accuracy, macro_f1, macro_roc_auc
      - confusion matrix (lista) e classification report (dict)
      - statistiche MC-dropout passate in mc_stats
    """
    y_true = [le.transform([k])[0] for k in keys]
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_pred, average="macro", multi_class="ovo")
    cm = confusion_matrix(y_true, y_pred).tolist()
    cr = classification_report(y_true, y_pred, output_dict=True)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
        "class_report": cr,
        **mc_stats
    }

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i+1])
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * (mask.mean())
    return float(ece)

def mc_dropout_statistics(mc_logits: np.ndarray) -> dict[str, float]:
    """
    Calcola incertezza epistemica (varianza) e entropia media
    su T passaggi MC-Dropout (shape: [T, N, C]).
    """
    mean_logits = mc_logits.mean(axis=0)
    probs       = softmax(mean_logits)
    var_per_sample = mc_logits.var(axis=0).sum(axis=1)
    epistemic = float(np.mean(var_per_sample))
    ent = -np.sum(probs * np.log(probs + 1e-12), axis=1).mean()
    return {
        "uncertainty_epistemic": epistemic,
        "uncertainty_entropy": float(ent)
    }

def aggregate_fold_metrics(
    per_fold: list[dict[str, float]]
) -> dict[str, dict[str, float]]:
    agg: dict[str, dict[str, float]] = {}
    keys = per_fold[0].keys()
    for k in keys:
        vals = np.array([fold[k] for fold in per_fold], dtype=float)
        agg[k] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals, ddof=1))
        }
    return agg


