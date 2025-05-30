# extract_features.py

import torch
import torchvision.transforms as T
from tqdm import tqdm

def extract_features(backbone, dataloader, device):
    backbone = backbone.to(device).eval()

    all_features = []
    all_keys = []

    with torch.no_grad():
        for imgs, keys in tqdm(dataloader, desc="Extracting features"):
            imgs = [i.convert("RGB") for i in imgs]
            imgs = torch.stack([T.ToTensor()(i) for i in imgs]).to(device)
            feats = backbone(imgs)
            all_features.append(feats.cpu())
            all_keys.extend(keys)

    features = torch.cat(all_features)
    return {"features": features, "keys": all_keys}

######## How to implement in the SimCLR trainer ########
# from feature_extractor import extract_features

# def extract_features_from_simclr(self, output_path: str):
#     """
#     Estrae le feature dal backbone SimCLR e le salva in output_path.
#     """
#     dataloader = self._make_inference_loader(self.train_pattern)
#     features = extract_features(self.backbone, dataloader, self.device)
#     torch.save(features, output_path)
#     LOGGER.info(f"✅ Features salvate in {output_path}")

# def _make_inference_loader(self, shards_pattern: str):
#     ds = (
#         wds.WebDataset(
#             shards_pattern,
#             handler=wds.warn_and_continue,
#             shardshuffle=0,
#             empty_check=False,
#         )
#         .decode("pil")
#         .map(lambda sample: (
#             next((v for k, v in sample.items() if isinstance(v, Image.Image)), None),
#             sample["__key__"]
#         ))
#     )

#     is_cuda = (self.device.type == "cuda")
#     return torch.utils.data.DataLoader(
#         ds,
#         batch_size=self.batch_size,
#         shuffle=False,
#         num_workers=4 if is_cuda else 0,
#         pin_memory=is_cuda,
#     )
# Dopo aver allenato SimCLR
# simclr_trainer = SimCLRTrainer(model_cfg, data_cfg)
# simclr_trainer.extract_features_from_simclr("simclr_features.pt")
#########################################################