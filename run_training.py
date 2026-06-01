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
import torch

from src.data import HLSDownloader, process_region, create_dataloaders, load_config
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
    downloader = HLSDownloader(config_path)
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
                dnbr_threshold=config["data"].get("dnbr_threshold", 0.10),
                background_keep=config["data"].get("background_keep", 0.3),
                max_patches=config["data"].get("max_patches_per_region"),
                prithvi_version=config["model"].get("prithvi_version", "1.0"),
            )
            logger.info(f"  {name}: {len(patches)} patches")
            all_patches.extend(patches)
        except Exception as e:
            logger.warning(f"  {name}: SKIPPED — {e}")
    return all_patches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--experiment-name", default="default")
    # Optional overrides for the precision/recall sweep. Tversky alpha penalizes
    # false positives, beta penalizes false negatives (alpha=beta=0.5 == Dice).
    parser.add_argument("--tversky-alpha", type=float, default=None)
    parser.add_argument("--tversky-beta", type=float, default=None)
    parser.add_argument("--class-weights", type=float, nargs=2, default=None,
                        help="Override CE class weights, e.g. --class-weights 0.5 0.5")
    parser.add_argument("--prithvi-version", default=None, choices=["1.0", "2.0"],
                        help="Override Prithvi encoder version (default from config)")
    parser.add_argument("--download-only", action="store_true",
                        help="Download/cache all regions then exit (no training). "
                             "Lets a brightness diagnostic run before GPU hours.")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.tversky_alpha is not None:
        config["training"]["tversky_alpha"] = args.tversky_alpha
    if args.tversky_beta is not None:
        config["training"]["tversky_beta"] = args.tversky_beta
    if args.class_weights is not None:
        config["training"]["class_weights"] = list(args.class_weights)
    if args.prithvi_version is not None:
        config["model"]["prithvi_version"] = args.prithvi_version
    logger.info(
        f"Loss config — class_weights={config['training']['class_weights']}, "
        f"tversky_alpha={config['training'].get('tversky_alpha', 0.5)}, "
        f"tversky_beta={config['training'].get('tversky_beta', 0.5)}"
    )

    torch.manual_seed(config["training"]["seed"])

    train_regions = config["data"]["train_regions"]
    test_regions = config["data"]["test_regions"]
    all_regions = train_regions + test_regions

    # --- 1. Download all regions ---
    logger.info("=== Downloading HLS imagery ===")
    logger.info(f"Train fires: {[r['name'] for r in train_regions]}")
    logger.info(f"Test fires:  {[r['name'] for r in test_regions]}")
    downloaded = download_regions(all_regions, args.config)

    if args.download_only:
        logger.info(f"--download-only: cached {len(downloaded)}/{len(all_regions)} "
                    f"regions. Exiting before training.")
        return

    # --- 2. Build train patches (fire-based split, test fires excluded) ---
    logger.info("=== Preprocessing train fires ===")
    train_patches = collect_patches(train_regions, downloaded, config)

    # Hard-negative regions: unburned dry SoCal terrain (≈all-background masks).
    # They teach the post-only model that dark/dry land is not burned, curbing
    # over-prediction. Downloaded + patched through the same pipeline.
    negative_regions = config["data"].get("negative_regions", [])
    if negative_regions:
        logger.info(f"=== Preprocessing hard-negative regions: {[r['name'] for r in negative_regions]} ===")
        downloaded_neg = download_regions(negative_regions, args.config)
        neg_patches = collect_patches(negative_regions, downloaded_neg, config)
        logger.info(f"Added {len(neg_patches)} hard-negative patches from {len(negative_regions)} regions")
        train_patches.extend(neg_patches)

    if len(train_patches) < 10:
        logger.error("Too few training patches. Check downloads.")
        return

    val_fires = config["data"].get("val_fires", [])
    if val_fires:
        # Fire-based split: hold out whole fires for validation so train and val
        # patches never come from the same scene. A random patch split lets
        # patches from one fire land in both train and val (spatially adjacent,
        # near-duplicate), giving an optimistic val IoU that hides overfitting —
        # which matters most when fine-tuning the full ViT.
        split_patches = {
            "train": [p for p in train_patches if p.get("region_name") not in val_fires],
            "val":   [p for p in train_patches if p.get("region_name") in val_fires],
            "test":  [],
        }
        logger.info(f"Fire-based val split — holding out {val_fires} for validation")
    else:
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
        f"Patch split — train: {len(split_patches['train'])}, val: {len(split_patches['val'])}"
    )

    # --- 3. Create dataloaders ---
    dataloaders = create_dataloaders(split_patches, config)

    # --- 4. Initialize model (downloads Prithvi weights on first run) ---
    logger.info("=== Initializing model ===")
    prithvi_ver = config["model"].get("prithvi_version", "1.0")
    model = BurnScarModel(
        num_classes=config["model"]["num_classes"],
        in_channels=config["model"]["in_channels"],
        freeze_backbone=config["model"]["freeze_backbone"],
        prithvi_version=prithvi_ver,
    )
    logger.info(f"Prithvi version: {prithvi_ver}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,}")

    # --- 5. Train ---
    checkpoint_dir = f"checkpoints/{args.experiment_name}"
    logger.info(f"=== Training (experiment: {args.experiment_name}) ===")
    trainer = Trainer(model, config, dataloaders, checkpoint_dir=checkpoint_dir)
    history = trainer.train()

    curves_path = f"{checkpoint_dir}/training_curves.png"
    plot_training_curves(history, save_path=curves_path)
    logger.info(f"Training complete. Best model saved to {checkpoint_dir}/best_model.pt")
    logger.info("Run inference on Woolsey Fire: python run_inference.py --region woolsey_fire_2018")


if __name__ == "__main__":
    main()
