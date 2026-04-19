"""
Main training script for burn scar detection.

Train on 10 large California fires, evaluate on the held-out Woolsey Fire.

Usage:
    python run_training.py
    python run_training.py --config configs/train_config.yaml
"""

import argparse
import logging
import os
import warnings

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")

import numpy as np
import yaml
import torch

from src.data import SentinelDownloader, process_region, create_dataloaders
from src.model import BurnScarModel
from src.train import Trainer
from src.visualize import plot_training_curves

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def download_regions(regions: list[dict], config_path: str) -> dict:
    """Download all regions, return {name: {pre: Path, post: Path}}."""
    downloader = SentinelDownloader(config_path)
    results = {}
    for region in regions:
        try:
            paths = downloader.download_region(region)
            results[region["name"]] = paths
        except Exception as e:
            logger.error(f"Failed to download {region['name']}: {e}")
    return results


def collect_patches(regions: list[dict], downloaded: dict, config: dict) -> list[dict]:
    """Process downloaded regions into patches."""
    all_patches = []
    for region in regions:
        name = region["name"]
        if name not in downloaded:
            logger.warning(f"Skipping {name} — not downloaded")
            continue
        try:
            patches = process_region(
                downloaded[name]["pre"],
                downloaded[name]["post"],
                bands=config["data"]["bands"],
                patch_size=config["data"]["patch_size"],
                region_name=name,
            )
            logger.info(f"  {name}: {len(patches)} patches")
            all_patches.extend(patches)
        except Exception as e:
            logger.warning(f"  {name}: SKIPPED — {e}")
    return all_patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["training"]["seed"])

    train_regions = config["data"]["train_regions"]
    test_regions = config["data"]["test_regions"]
    all_regions = train_regions + test_regions

    # --- 1. Download all regions ---
    logger.info("=== Downloading Sentinel-2 imagery ===")
    logger.info(f"Train fires: {[r['name'] for r in train_regions]}")
    logger.info(f"Test fires:  {[r['name'] for r in test_regions]}")
    downloaded = download_regions(all_regions, args.config)

    # --- 2. Build train patches (fire-based split, Woolsey excluded) ---
    logger.info("=== Preprocessing train fires ===")
    train_patches = collect_patches(train_regions, downloaded, config)

    if len(train_patches) < 10:
        logger.error("Too few training patches. Check downloads.")
        return

    # Split train patches 90/10 into train/val
    rng = np.random.default_rng(config["training"]["seed"])
    indices = rng.permutation(len(train_patches))
    split = int(len(train_patches) * config["data"]["train_split"])
    train_idx, val_idx = indices[:split], indices[split:]
    split_patches = {
        "train": [train_patches[i] for i in train_idx],
        "val":   [train_patches[i] for i in val_idx],
        "test":  [],
    }
    logger.info(
        f"Patch split — train: {len(split_patches['train'])}, "
        f"val: {len(split_patches['val'])}"
    )

    # --- 3. Create dataloaders ---
    dataloaders = create_dataloaders(split_patches, config)

    # --- 4. Initialize model (downloads Prithvi weights on first run) ---
    logger.info("=== Initializing model ===")
    model = BurnScarModel(
        num_classes=config["model"]["num_classes"],
        in_channels=config["model"]["in_channels"],
        freeze_backbone=config["model"]["freeze_backbone"],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # --- 5. Train ---
    logger.info("=== Training ===")
    trainer = Trainer(model, config, dataloaders)
    history = trainer.train()

    plot_training_curves(history, save_path="training_curves.png")
    logger.info("Training complete. Best model saved to checkpoints/best_model.pt")
    logger.info("Run inference on Woolsey Fire: python run_inference.py --region woolsey_fire_2018")


if __name__ == "__main__":
    main()
