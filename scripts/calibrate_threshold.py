"""
Calibrate the decision threshold on the TRAINING fires only (never the held-out
test fires), to avoid leakage. Picks the threshold that maximizes mean per-fire
IoU on the train set; that value is then deployed unchanged to the test fires.

Mirrors the production pipeline exactly: same sliding-window inference
(run_inference, return_prob=True) and the same NDWI water exclusion.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import xarray as xr

from src.data import _restore_crs, load_config
from src.model import BurnScarModel
from src.utils import get_device, water_mask
from run_inference import run_inference


def iou_pr(prob, true, valid, thr):
    p = (prob > thr)[valid]
    t = true[valid].astype(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return iou, prec, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/balanced_chaparral/best_model.pt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bands = cfg["data"]["bands"]; ps = cfg["data"]["patch_size"]
    dnbr_t = cfg["data"].get("dnbr_threshold", 0.10); cache = cfg["data"]["cache_dir"]
    device = get_device()

    model = BurnScarModel(num_classes=cfg["model"]["num_classes"], in_channels=cfg["model"]["in_channels"])
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model_state_dict"])
    model = model.to(device)

    train = cfg["data"]["train_regions"]; test = cfg["data"]["test_regions"]

    def collect(regions):
        out = []
        for r in regions:
            name = r["name"]
            pp, qp = f"{cache}/{name}_pre.nc", f"{cache}/{name}_post.nc"
            if not (Path(pp).exists() and Path(qp).exists()):
                print(f"  skip {name} (no cache)"); continue
            pre = _restore_crs(xr.open_dataset(pp, engine="h5netcdf"))
            post = _restore_crs(xr.open_dataset(qp, engine="h5netcdf")).rio.reproject_match(pre)
            _, true, image, prob = run_inference(model, post, pre, bands=bands, patch_size=ps,
                                                  device=device, dnbr_threshold=dnbr_t, return_prob=True)
            w = water_mask(post)
            if w.shape == prob.shape:
                prob = prob.copy(); true = true.copy()
                prob[w] = 0.0; true[w] = 0
            valid = ~(np.isnan(image).any(axis=0) | (np.nan_to_num(image).max(axis=0) == 0))
            out.append((name, prob, true, valid))
            print(f"  {name}: done")
        return out

    print("Collecting TRAIN-fire predictions...")
    train_data = collect(train)

    thresholds = np.round(np.arange(0.30, 0.81, 0.02), 2)
    print("\nthr   mean_train_IoU   (per-fire IoU)")
    best_thr, best_iou = 0.5, -1
    for thr in thresholds:
        ious = [iou_pr(p, t, v, thr)[0] for _, p, t, v in train_data]
        m = float(np.mean(ious))
        if m > best_iou:
            best_iou, best_thr = m, float(thr)
    for thr in thresholds:
        ious = [iou_pr(p, t, v, thr)[0] for _, p, t, v in train_data]
        mark = "  <-- best" if abs(thr - best_thr) < 1e-6 else ""
        print(f"{thr:.2f}      {np.mean(ious):.4f}{mark}")

    print(f"\n==> Honest optimal threshold (train fires): {best_thr:.2f}  (mean train IoU {best_iou:.4f})")
    cur = cfg["data"].get("pred_threshold", 0.5)
    print(f"    Current deployed threshold: {cur}")

    # Information only: apply the train-calibrated threshold to the test fires.
    print("\nApplying calibrated threshold to HELD-OUT test fires (report only):")
    test_data = collect(test)
    print(f"\n{'fire':24s} {'@'+str(cur)+' IoU':>12s} {'@'+str(best_thr)+' P/R/IoU':>26s}")
    for name, p, t, v in test_data:
        i0, _, _ = iou_pr(p, t, v, cur)
        i1, pr1, rc1 = iou_pr(p, t, v, best_thr)
        print(f"{name:24s} {i0:>12.3f}   {pr1:>6.3f}/{rc1:.3f}/{i1:.3f}")


if __name__ == "__main__":
    main()
