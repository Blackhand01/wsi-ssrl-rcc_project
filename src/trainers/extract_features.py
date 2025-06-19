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
          imgs_tensor = batch["img"].to(device)
          keys = batch["key"]
          feats = backbone(imgs_tensor)                
          all_features.append(feats.cpu())
          all_keys.extend(keys)

    features = torch.cat(all_features)
    return {"features": features, "keys": all_keys}
