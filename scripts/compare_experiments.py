"""
Compare frozen vs. fine-tuned encoder experiments.

Usage:
    PYTHONPATH=. python scripts/compare_experiments.py
    PYTHONPATH=. python scripts/compare_experiments.py --frozen checkpoints/frozen --finetuned checkpoints/finetuned
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import _restore_crs, normalize_bands, generate_burn_mask
from src.model import BurnScarModel
from src.train import compute_metrics
from run_inference import run_inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_history(checkpoint_dir: Path) -> dict:
    history_path = checkpoint_dir / "history.pt"
    if not history_path.exists():
        raise FileNotFoundError(f"No history.pt in {checkpoint_dir}")
    return torch.load(history_path, map_location="cpu", weights_only=False)


def plot_comparison(frozen_history: dict, finetuned_history: dict, unfreeze_epoch: int, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hist, label, color in [
        (frozen_history, "Frozen encoder", "#2196F3"),
        (finetuned_history, "Fine-tuned encoder", "#F44336"),
    ]:
        epochs = range(1, len(hist["train"]) + 1)
        train_loss = [m["loss"] for m in hist["train"]]
        val_loss = [m["loss"] for m in hist["val"]]
        train_iou = [m["mean_iou"] for m in hist["train"]]
        val_iou = [m["mean_iou"] for m in hist["val"]]

        axes[0].plot(epochs, train_loss, "-", color=color, alpha=0.4, linewidth=1)
        axes[0].plot(epochs, val_loss, "-", color=color, label=label, linewidth=2)
        axes[1].plot(epochs, train_iou, "-", color=color, alpha=0.4, linewidth=1)
        axes[1].plot(epochs, val_iou, "-", color=color, label=label, linewidth=2)

    axes[1].axvline(x=unfreeze_epoch + 0.5, color="#F44336", linestyle="--", alpha=0.5, label="Encoder unfrozen")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean IoU")
    axes[1].set_title("Validation IoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Frozen vs. Fine-tuned Prithvi Encoder", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {save_path}")
    plt.close()


def evaluate_on_woolsey(checkpoint_path: Path, config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = BurnScarModel(
        num_classes=config["model"]["num_classes"],
        in_channels=config["model"]["in_channels"],
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    cache_dir = Path(config["data"]["cache_dir"])
    pre_ds = _restore_crs(xr.open_dataset(cache_dir / "woolsey_fire_2018_pre.nc", engine="h5netcdf"))
    post_ds = _restore_crs(xr.open_dataset(cache_dir / "woolsey_fire_2018_post.nc", engine="h5netcdf"))
    post_ds = post_ds.rio.reproject_match(pre_ds)

    pred_mask, true_mask, _ = run_inference(
        model, post_ds, pre_ds,
        bands=config["data"]["bands"],
        patch_size=config["data"]["patch_size"],
        device=device,
        dnbr_threshold=config["data"].get("dnbr_threshold", 0.10),
    )

    tp = int(((pred_mask == 1) & (true_mask == 1)).sum())
    fp = int(((pred_mask == 1) & (true_mask == 0)).sum())
    fn = int(((pred_mask == 0) & (true_mask == 1)).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {"recall": recall, "precision": precision, "iou": iou}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default="checkpoints/frozen")
    parser.add_argument("--finetuned", default="checkpoints/finetuned")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--finetune-config", default="configs/finetune_config.yaml")
    args = parser.parse_args()

    frozen_dir = Path(args.frozen)
    finetuned_dir = Path(args.finetuned)

    frozen_history = load_history(frozen_dir)
    finetuned_history = load_history(finetuned_dir)

    with open(args.finetune_config) as f:
        ft_config = yaml.safe_load(f)
    unfreeze_epoch = ft_config["model"]["unfreeze_after_epoch"]

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    plot_comparison(frozen_history, finetuned_history, unfreeze_epoch, "results/frozen_vs_finetuned.png")

    logger.info("Evaluating frozen model on Woolsey Fire...")
    frozen_metrics = evaluate_on_woolsey(frozen_dir / "best_model.pt", args.config)

    logger.info("Evaluating fine-tuned model on Woolsey Fire...")
    finetuned_metrics = evaluate_on_woolsey(finetuned_dir / "best_model.pt", args.finetune_config)

    print("\n" + "=" * 60)
    print("  Woolsey Fire Results — Frozen vs. Fine-tuned Encoder")
    print("=" * 60)
    print(f"  {'Metric':<12} {'Frozen':>10} {'Fine-tuned':>12} {'Delta':>10}")
    print("-" * 60)
    for metric in ["recall", "precision", "iou"]:
        f_val = frozen_metrics[metric]
        ft_val = finetuned_metrics[metric]
        delta = ft_val - f_val
        sign = "+" if delta >= 0 else ""
        print(f"  {metric.capitalize():<12} {f_val*100:>9.1f}% {ft_val*100:>11.1f}% {sign}{delta*100:>9.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
