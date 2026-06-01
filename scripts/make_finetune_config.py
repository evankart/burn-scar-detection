"""
Derive configs/finetune_config.yaml from configs/train_config.yaml so the region
lists never drift, then overlay the encoder-fine-tune settings:
  - gradual unfreeze (decoder-only warmup, then unfreeze the encoder)
  - layer-wise LR decay (LLRD) on the encoder
  - stronger augmentation (random resized crop)
  - fire-based validation split (whole held-out fires)
"""
import sys
from pathlib import Path
import yaml

src = yaml.safe_load(open("configs/train_config.yaml"))

src["data"]["val_fires"] = ["carr_fire_2018", "holy_2018"]  # one NorCal + one SoCal, held out of training
# Prithvi 2.0 uses B05/B06/B07 (broadband NIR + SWIR) instead of B8A/B11/B12
src["data"]["bands"] = ["B02", "B03", "B04", "B05", "B06", "B07"]

src["model"]["prithvi_version"] = "2.0"
src["model"]["freeze_backbone"] = True
src["model"]["unfreeze_after_epoch"] = 2  # decoder-only for 2 epochs, then unfreeze encoder

src["training"].update({
    "epochs": 16,
    "learning_rate": 3.0e-4,
    "backbone_lr_multiplier": 0.05,   # encoder base LR = 5% of decoder LR
    "llrd_decay": 0.75,               # shallow Prithvi layers get progressively lower LR
    "weight_decay": 0.05,
    "warmup_epochs": 1,
    "early_stopping_patience": 5,
    "class_weights": [0.5, 0.5],      # neutral; fine-tuning has capacity to learn precision
    "tversky_alpha": 0.5,
    "tversky_beta": 0.5,
})
src["training"]["augmentations"]["random_resized_crop"] = True

src["logging"]["project_name"] = "burn-scar-detection-finetune"

with open("configs/finetune_config.yaml", "w") as f:
    yaml.safe_dump(src, f, sort_keys=False, default_flow_style=False)

n_train = len(src["data"]["train_regions"])
print(f"Wrote configs/finetune_config.yaml: {n_train} train fires, "
      f"val_fires={src['data']['val_fires']}, unfreeze@{src['model']['unfreeze_after_epoch']}, "
      f"epochs={src['training']['epochs']}, llrd_decay={src['training']['llrd_decay']}")
