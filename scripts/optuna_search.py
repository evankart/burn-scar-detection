"""
Optuna hyperparameter search for the Prithvi-EO-2.0 burn-scar fine-tune.

Tunes four knobs over ~10 trials, maximizing validation burn-class IoU on the
held-out *training* fires (carr_fire_2018 + holy_2018, i.e. finetune_config's
`val_fires`). Burn-class IoU (not mean-IoU) is the objective because background
dominates these scenes, so mean-IoU is half-saturated and dilutes the
precision/recall tradeoff the tuned knobs control. The test fires (palisades_2025, eaton_2025, woolsey_2018,
thomas_2017) are NEVER seen: only train-role fires are turned into patches, and
the val split holds out whole fires from that train set — so there is no path
by which a test fire enters the search. The script asserts this before running.

Search space (TODO.md item 1):
    learning_rate          1e-4 .. 1e-3   (log)
    backbone_lr_multiplier 0.01 .. 0.1    (log)
    tversky_alpha          0.3  .. 0.7    (beta = 1 - alpha)
    class_weights[1]       0.4  .. 0.7    (burn weight; weights = [1 - w1, w1])

Designed for AWS g5.xlarge. Data download + patch extraction happen ONCE; each
trial only rebuilds the model + Trainer. Upload results to S3 via
cloud/run_optuna.sh.

Usage:
    python scripts/optuna_search.py --config configs/finetune_config.yaml \
        --n-trials 10 --epochs 8 --experiment-name optuna
"""

import argparse
import copy
import logging
import os
import shutil
import sys
import warnings
from pathlib import Path

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore", message=".*unauthenticated.*HF Hub.*")

import numpy as np
import torch
import yaml

from src.data import create_dataloaders, load_config
from src.model import BurnScarModel
from src.train import Trainer
# Reuse the exact download + patch-extraction path the real training run uses.
from run_training import collect_patches, download_regions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optuna_search")


def build_split_patches(config: dict, config_path: str) -> dict:
    """Download train-role fires once and split into train/val by `val_fires`.

    Test-role fires are excluded by construction (only train_regions are
    processed). Returns {"train": [...], "val": [...], "test": []}.
    """
    train_regions = config["data"]["train_regions"]
    test_regions = config["data"]["test_regions"]
    val_fires = config["data"].get("val_fires", [])

    if not val_fires:
        raise SystemExit(
            "finetune_config must set data.val_fires (the fires to validate on). "
            "Search needs a fixed fire-based val split."
        )

    test_names = {r["name"] for r in test_regions}
    train_names = {r["name"] for r in train_regions}

    # Guardrails: val fires must be train-role; no test fire may leak in.
    leaked = set(val_fires) & test_names
    if leaked:
        raise SystemExit(f"val_fires overlaps test fires {leaked} — refusing to run.")
    missing = set(val_fires) - train_names
    if missing:
        raise SystemExit(f"val_fires {missing} are not train-role fires in the config.")

    logger.info(f"Train fires ({len(train_names)}): excludes test fires {sorted(test_names)}")
    logger.info(f"Validation fires (held out for the objective): {val_fires}")

    downloaded = download_regions(train_regions, config_path)
    train_patches = collect_patches(train_regions, downloaded, config)
    if len(train_patches) < 10:
        raise SystemExit("Too few training patches — check downloads.")

    split = {
        "train": [p for p in train_patches if p.get("region_name") not in val_fires],
        "val":   [p for p in train_patches if p.get("region_name") in val_fires],
        "test":  [],
    }
    if not split["val"]:
        raise SystemExit(f"No val patches produced for {val_fires} — check the cache.")

    # Final safety check: no test-fire patch anywhere.
    contaminated = {p.get("region_name") for p in train_patches} & test_names
    if contaminated:
        raise SystemExit(f"Test fire(s) {contaminated} found in patch set — aborting.")

    logger.info(f"Patches — train: {len(split['train'])}, val: {len(split['val'])}")
    return split


def make_objective(base_config: dict, dataloaders: dict, epochs: int, work_dir: Path):
    """Build the Optuna objective: train one fine-tune and return best val mean-IoU."""

    def objective(trial) -> float:
        cfg = copy.deepcopy(base_config)
        tc = cfg["training"]
        # Score the trial — and select each trial's best_model.pt — on burn-class
        # IoU, not mean-IoU. Background dominates these scenes (~8–30% burned), so
        # mean-IoU is half-saturated and dilutes exactly the precision/recall
        # tradeoff the tuned knobs control. Trainer reads this for both best_metric
        # and checkpoint selection, keeping the two consistent.
        cfg["logging"]["metric_monitor"] = "val_iou_burn_scar"

        lr      = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        bb_mult = trial.suggest_float("backbone_lr_multiplier", 0.01, 0.1, log=True)
        alpha   = trial.suggest_float("tversky_alpha", 0.3, 0.7)
        burn_w  = trial.suggest_float("class_weight_burn", 0.4, 0.7)

        tc["learning_rate"] = lr
        tc["backbone_lr_multiplier"] = bb_mult
        tc["tversky_alpha"] = alpha
        tc["tversky_beta"] = 1.0 - alpha          # keep alpha + beta = 1
        tc["class_weights"] = [1.0 - burn_w, burn_w]  # [background, burn]
        tc["epochs"] = epochs

        torch.manual_seed(cfg["training"]["seed"])
        model = BurnScarModel(
            num_classes=cfg["model"]["num_classes"],
            in_channels=cfg["model"]["in_channels"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
        )

        trial_dir = work_dir / f"trial_{trial.number}"
        trainer = Trainer(model, cfg, dataloaders, checkpoint_dir=str(trial_dir))
        logger.info(
            f"Trial {trial.number}: lr={lr:.2e} bb_mult={bb_mult:.3f} "
            f"alpha={alpha:.2f} class_weights=[{1 - burn_w:.2f}, {burn_w:.2f}]"
        )
        trainer.train()
        best_val_iou = float(trainer.best_metric)  # best monitored val burn-class IoU

        # Keep only best_model.pt per trial; drop per-epoch checkpoints to save disk.
        for ckpt in trial_dir.glob("epoch_*.pt"):
            ckpt.unlink(missing_ok=True)

        # Free GPU memory between trials.
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return best_val_iou

    return objective


def save_results(study, work_dir: Path, base_config: dict, config_path: str):
    """Persist best params, study object, plots, and a ready-to-train config."""
    import optuna

    work_dir.mkdir(parents=True, exist_ok=True)
    best = study.best_trial
    alpha = best.params["tversky_alpha"]
    burn_w = best.params["class_weight_burn"]

    summary = {
        "best_value_val_iou_burn_scar": float(best.value),
        "best_trial_number": best.number,
        "n_trials": len(study.trials),
        "best_params": {
            "learning_rate": best.params["learning_rate"],
            "backbone_lr_multiplier": best.params["backbone_lr_multiplier"],
            "tversky_alpha": alpha,
            "tversky_beta": 1.0 - alpha,
            "class_weights": [1.0 - burn_w, burn_w],
        },
    }
    (work_dir / "best_params.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    logger.info(f"Best trial #{best.number}: val_iou_burn_scar={best.value:.4f}")
    logger.info(f"Best params:\n{yaml.safe_dump(summary['best_params'], sort_keys=False)}")

    # Pickle the full study for later analysis / the notebook.
    try:
        import joblib
        joblib.dump(study, work_dir / "study.pkl")
    except Exception as e:
        logger.warning(f"Could not pickle study: {e}")

    # A finetune config with the best HPs baked in (extends the same base config).
    tuned_cfg = {
        "extends": Path(config_path).name,
        "training": {
            "learning_rate": float(best.params["learning_rate"]),
            "backbone_lr_multiplier": float(best.params["backbone_lr_multiplier"]),
            "tversky_alpha": float(alpha),
            "tversky_beta": float(1.0 - alpha),
            "class_weights": [float(1.0 - burn_w), float(burn_w)],
        },
        # Keep the final retrain consistent with the search: select best_model on
        # burn-class IoU too (not mean-IoU).
        "logging": {"metric_monitor": "val_iou_burn_scar"},
    }
    tuned_path = Path("configs") / "finetune_optuna_config.yaml"
    tuned_path.write_text(
        "# Auto-generated by scripts/optuna_search.py — best Optuna trial.\n"
        "# Use for the final retrain: EXP=finetune_v3 CONFIG=configs/finetune_optuna_config.yaml\n"
        + yaml.safe_dump(tuned_cfg, sort_keys=False)
    )
    logger.info(f"Wrote tuned config: {tuned_path}")

    # Optuna visualizations (matplotlib backend; saved as PNG for the notebook).
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization.matplotlib import (
            plot_optimization_history, plot_param_importances,
        )
        plot_optimization_history(study)
        plt.tight_layout(); plt.savefig(work_dir / "optuna_history.png", dpi=120); plt.close()
        if len(study.trials) > 1:
            plot_param_importances(study)
            plt.tight_layout(); plt.savefig(work_dir / "optuna_param_importances.png", dpi=120); plt.close()
        logger.info(f"Saved Optuna plots to {work_dir}/")
    except Exception as e:
        logger.warning(f"Could not render Optuna plots: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/finetune_config.yaml")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8,
                        help="Epochs per trial (shorter than the final retrain to save GPU hours).")
    parser.add_argument("--experiment-name", default="optuna")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import optuna

    config = load_config(args.config)
    work_dir = Path("checkpoints") / args.experiment_name

    # Build data ONCE — reused across every trial.
    logger.info("=== Preparing data (download + patches, once) ===")
    split_patches = build_split_patches(config, args.config)
    dataloaders = create_dataloaders(split_patches, config)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name="burn_scar_finetune",
    )
    objective = make_objective(config, dataloaders, args.epochs, work_dir)

    logger.info(f"=== Optuna search: {args.n_trials} trials, {args.epochs} epochs each ===")
    study.optimize(objective, n_trials=args.n_trials)

    save_results(study, work_dir, config, args.config)
    logger.info("=== Search complete ===")


if __name__ == "__main__":
    main()
