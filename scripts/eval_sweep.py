"""
Evaluate checkpoints on the held-out test fires at a FIXED decision threshold.
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

TEST_FIRES = ["woolsey_fire_2018", "east_troublesome_2020", "thomas_fire_2017"]


def metrics(pred, true, valid):
    p = pred[valid].astype(bool)
    t = true[valid].astype(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return prec, rec, iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--config", default="configs/train_config.yaml")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    bands = cfg["data"]["bands"]
    ps = cfg["data"]["patch_size"]
    dnbr_t = cfg["data"].get("dnbr_threshold", 0.10)
    cache = cfg["data"]["cache_dir"]
    device = get_device()

    # Pre-load test scenes once
    scenes = {}
    for name in TEST_FIRES:
        pre = _restore_crs(xr.open_dataset(f"{cache}/{name}_pre.nc", engine="h5netcdf"))
        post = _restore_crs(xr.open_dataset(f"{cache}/{name}_post.nc", engine="h5netcdf"))
        post = post.rio.reproject_match(pre)
        scenes[name] = (pre, post)

    results = {}
    for ckpt in args.checkpoints:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        # Build the encoder version this checkpoint was trained with (stored in
        # its config) so a 2.0 checkpoint loads correctly. Bands are the same
        # physical HLS bands for both versions, so the cached scenes work as-is.
        ck_ver = state.get("config", {}).get("model", {}).get("prithvi_version", "1.0")
        model = BurnScarModel(num_classes=cfg["model"]["num_classes"],
                              in_channels=cfg["model"]["in_channels"],
                              prithvi_version=ck_ver)
        model.load_state_dict(state["model_state_dict"])
        model = model.to(device)
        label = ckpt.split("/")[-2]
        per_fire = {}
        for name in TEST_FIRES:
            pre, post = scenes[name]
            pred, true, image = run_inference(
                model, post, pre, bands=bands, patch_size=ps, device=device,
                dnbr_threshold=dnbr_t, pred_threshold=args.threshold,
                prithvi_version=ck_ver,
            )
            # Same NDWI water exclusion as the deployed pipeline.
            water = water_mask(post)
            if water.shape == pred.shape:
                pred = pred.copy(); true = true.copy()
                pred[water] = 0; true[water] = 0
            valid = ~(np.isnan(image).any(axis=0) | (np.nan_to_num(image).max(axis=0) == 0))
            per_fire[name] = metrics(pred, true, valid)
        results[label] = per_fire

    # Report
    print(f"\n=== Test-fire evaluation @ fixed threshold {args.threshold} ===")
    hdr = f"{'config':<22}" + "".join(f"{f.split('_')[0]:>26}" for f in TEST_FIRES) + f"{'MACRO':>26}"
    print(hdr)
    print(f"{'':22}" + "".join(f"{'P / R / IoU':>26}" for _ in TEST_FIRES) + f"{'P / R / IoU':>26}")
    for label, pf in results.items():
        row = f"{label:<22}"
        macro = np.zeros(3)
        for f in TEST_FIRES:
            p, r, i = pf[f]
            macro += [p, r, i]
            row += f"{p:>7.3f} /{r:>6.3f} /{i:>6.3f}"
        macro /= len(TEST_FIRES)
        row += f"{macro[0]:>7.3f} /{macro[1]:>6.3f} /{macro[2]:>6.3f}"
        print(row)


if __name__ == "__main__":
    main()
