# extract_features.py

import torch
import torchvision.transforms as T
from tqdm import tqdm

def extract_features(backbone, dataloader, device):
    backbone = backbone.to(device).eval()

    all_features = []
    all_keys = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Support both tuple/list and dict batch formats
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs_tensor, keys = batch
            elif isinstance(batch, dict):
                imgs_tensor, keys = batch["img"], batch["key"]
            else:
                raise RuntimeError(f"Unknown batch format: {type(batch)}")

            imgs_tensor = imgs_tensor.to(device)
            feats = backbone(imgs_tensor)
            all_features.append(feats.cpu())
            all_keys.extend(keys)

    features = torch.cat(all_features)
    return {"features": features, "keys": all_keys}
