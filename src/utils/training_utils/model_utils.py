# src/utils/training_utils/model_utils.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Tuple

def create_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    Crea un backbone torchvision con fc finale dimensione num_classes.
    Supporta 'resnet18' e 'resnet50'.
    """
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Backbone '{name}' non supportato. Usa 'resnet18' o 'resnet50'.")
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def mc_dropout_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    T: int = 20
) -> np.ndarray:
    """
    Esegue T forward-pass in modalità train (con dropout attivo).
    Restituisce un array shape [T, N, C] con i logits raccolti.

    Args:
        model: modello PyTorch con dropout attivo.
        loader: DataLoader con batch (x, y) o (x, _).
        device: dispositivo CUDA o CPU.
        T: numero di forward-pass MC-Dropout.
    """
    model.train()
    all_logits = []

    with torch.no_grad():
        for _ in range(T):
            logits_t = []
            for xb, *_ in loader:  # ignora label
                xb = xb.to(device)
                logits = model(xb).cpu().numpy()
                logits_t.append(logits)
            all_logits.append(np.concatenate(logits_t, axis=0))

    return np.stack(all_logits, axis=0)




class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss (SimCLR)."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature
        pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)], dim=0)
        mask = ~torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        neg = sim.masked_select(mask).view(2 * N, 2 * N - 1)
        return -(pos - torch.logsumexp(neg, dim=1)).mean()
