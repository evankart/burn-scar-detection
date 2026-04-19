"""
Training loop for burn scar segmentation model.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> dict:
    pixel_accuracy = float((preds == labels).sum() / labels.size)

    ious = []
    for cls in range(num_classes):
        intersection = ((preds == cls) & (labels == cls)).sum()
        union = ((preds == cls) | (labels == cls)).sum()
        ious.append(intersection / union if union > 0 else float("nan"))
    mean_iou = float(np.nanmean(ious))

    tp = ((preds == 1) & (labels == 1)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"pixel_accuracy": pixel_accuracy, "mean_iou": mean_iou, "f1_burn_scar": float(f1)}


class Trainer:
    def __init__(self, model: nn.Module, config: dict, dataloaders: dict, device=None):
        self.config = config
        self.tc = config["training"]
        self.dataloaders = dataloaders
        self.device = device or get_device()
        self.model = model.to(self.device)

        encoder_params, decoder_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_params if "encoder" in name else decoder_params).append(param)

        lr = self.tc["learning_rate"]
        self.optimizer = AdamW([
            {"params": encoder_params, "lr": lr * self.tc["backbone_lr_multiplier"]},
            {"params": decoder_params, "lr": lr},
        ], weight_decay=self.tc["weight_decay"])

        warmup = LinearLR(self.optimizer, start_factor=0.1, total_iters=self.tc["warmup_epochs"])
        cosine = CosineAnnealingLR(self.optimizer, T_max=self.tc["epochs"] - self.tc["warmup_epochs"])
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup, cosine],
            milestones=[self.tc["warmup_epochs"]],
        )

        weights = torch.tensor(self.tc["class_weights"], dtype=torch.float32).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.best_metric = 0.0
        self.patience_counter = 0
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.use_wandb = False
        try:
            import wandb
            if wandb.api.api_key:
                wandb.init(project=config["logging"]["project_name"], config=config)
                self.use_wandb = True
            else:
                logger.info("W&B not logged in, logging to console only")
        except Exception:
            logger.info("W&B not available, logging to console only")

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, batch in enumerate(self.dataloaders["train"]):
            images = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

            if batch_idx % self.config["logging"]["log_every_n_steps"] == 0:
                logger.info(f"Epoch {epoch} [{batch_idx}/{len(self.dataloaders['train'])}] loss={loss.item():.4f}")

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels, num_classes=self.model.num_classes)
        metrics["loss"] = total_loss / len(self.dataloaders["train"])
        return metrics

    @torch.no_grad()
    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in self.dataloaders["val"]:
            images = batch["pixel_values"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels, num_classes=self.model.num_classes)
        metrics["loss"] = total_loss / len(self.dataloaders["val"])
        return metrics

    def train(self) -> dict:
        logger.info(f"Training on {self.device} for {self.tc['epochs']} epochs")
        history = {"train": [], "val": []}

        for epoch in range(1, self.tc["epochs"] + 1):
            t0 = time.time()

            if (
                epoch == self.config["model"].get("unfreeze_after_epoch", 0) + 1
                and hasattr(self.model, "unfreeze_backbone")
            ):
                self.model.unfreeze_backbone()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch}/{self.tc['epochs']} ({elapsed:.1f}s) — "
                f"train_loss={train_metrics['loss']:.4f} train_iou={train_metrics['mean_iou']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_iou={val_metrics['mean_iou']:.4f}"
            )

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            if self.use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

            monitored = val_metrics[self.config["logging"]["metric_monitor"].replace("val_", "")]
            if monitored > self.best_metric:
                self.best_metric = monitored
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.tc["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if self.use_wandb:
            import wandb
            wandb.finish()

        return history

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        path = self.checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(state, path)
        if is_best:
            torch.save(state, self.checkpoint_dir / "best_model.pt")
            logger.info(f"Saved best model (val_iou={metrics['mean_iou']:.4f})")
