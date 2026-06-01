"""
Brightness diagnostic for a Prithvi 2.0 fine-tune (run after download, before
training). For each band in the chosen config, pools the per-band median surface
reflectance across the TRAINING fires only (never test/val fires) and compares it
to the version's pretraining mean. The ratio mean/median is the implied per-band
brightness gain — if it sits near 1.0, HLS already matches the 2.0 pretraining
distribution and no gain is needed; if it is consistently >1, the same darkness
correction that helped Prithvi 1.0 likely applies to 2.0 as well.

This does NOT modify normalization or train — it only prints the numbers so we
can decide whether to set a 2.0 GAIN before committing GPU hours.

Usage:
    python scripts/band_stats_v2.py --config configs/finetune_config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import xarray as xr
import yaml

from src.data import _restore_crs
from src.model import PRITHVI_VERSIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/finetune_config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    bands = cfg["data"]["bands"]
    version = cfg["model"].get("prithvi_version", "1.0")
    cache_dir = Path(cfg["data"]["cache_dir"])

    vcfg = PRITHVI_VERSIONS[version]
    mean = np.array(vcfg["mean"], dtype=np.float32)
    std = np.array(vcfg["std"], dtype=np.float32)
    val_fires = set(cfg["data"].get("val_fires", []))
    test_fires = {r["name"] for r in cfg["data"].get("test_regions", [])}
    excluded = val_fires | test_fires

    train_fires = [
        r["name"] for r in cfg["data"]["train_regions"] if r["name"] not in excluded
    ]

    print(f"=== Prithvi {version} band brightness/scale diagnostic ===")
    print(f"bands={bands}")
    print(f"pretraining mean={[round(float(m), 4) for m in mean]}")
    print(f"pretraining std ={[round(float(s), 4) for s in std]}")
    print(f"pooling stats over {len(train_fires)} TRAIN fires "
          f"(excluding {sorted(excluded)})\n")

    meds = {b: [] for b in bands}
    stds = {b: [] for b in bands}
    n_used = 0
    for nm in train_fires:
        path = cache_dir / f"{nm}_post.nc"
        if not path.exists():
            print(f"  skip {nm}: no cached post.nc")
            continue
        try:
            d = _restore_crs(xr.open_dataset(path, engine="h5netcdf"))
        except Exception as e:
            print(f"  skip {nm}: {e}")
            continue
        missing = [b for b in bands if b not in d.variables]
        if missing:
            print(f"  skip {nm}: missing bands {missing}")
            continue
        for b in bands:
            a = np.clip(d[b].values.astype(np.float32), 0, 1)
            a = a[np.isfinite(a) & (a > 0)]
            if a.size:
                meds[b].append(float(np.median(a)))
                stds[b].append(float(np.std(a)))
        n_used += 1

    if n_used == 0:
        print("\nNo usable training caches found — cannot compute gains.")
        return

    print(f"\nUsed {n_used} fires.\n")
    # --- Center: median vs pretraining mean -> implied brightness gain ---
    print("CENTER (brightness):")
    print(f"{'band':<6}{'pool_median':>13}{'pretrain_mean':>15}{'implied_gain':>14}")
    gains = []
    for i, b in enumerate(bands):
        pool_med = float(np.mean(meds[b])) if meds[b] else float("nan")
        gain = float(mean[i] / (pool_med + 1e-6))
        gains.append(round(gain, 4))
        print(f"{b:<6}{pool_med:>13.4f}{float(mean[i]):>15.4f}{gain:>14.3f}")

    print(f"\nImplied GAIN_{version.replace('.', '')} = {gains}")
    spread = max(gains) - min(gains)
    if all(abs(g - 1.0) < 0.15 for g in gains):
        print("→ Gains ~1.0: HLS already matches 2.0 pretraining; no gain needed.")
    else:
        print(f"→ Gains depart from 1.0 (spread {spread:.2f}): consider setting "
              f"GAIN_2 in src/data.normalize_bands and re-running this diagnostic.")

    # --- Scale: pooled std vs registry std -> unit/scale sanity-check ---
    # IMPORTANT: the registry std is the *pretraining-distribution* std (what we
    # z-score against so the frozen encoder sees its expected distribution). HLS
    # is darker and less varied than pretraining, so pooled_std < registry_std is
    # EXPECTED and not a bug — it is the same domain gap the brightness gain
    # addresses. For reference, the verified 1.0 stats give std ratios ~0.3-0.6.
    # This check therefore only flags a GROSS scale error: if my hand-derived 2.0
    # stats were off by a unit factor (e.g. raw DN vs 0-1 reflectance), the ratio
    # would be ~10x or ~0.01x off. Anything in ~0.1-3x is plausible.
    print("\nSCALE (unit sanity-check; ratio<1 expected, only gross errors flagged):")
    print(f"{'band':<6}{'pool_std':>10}{'registry_std':>14}{'ratio':>9}")
    ratios = []
    for i, b in enumerate(bands):
        pool_std = float(np.mean(stds[b])) if stds[b] else float("nan")
        ratio = float(pool_std / (std[i] + 1e-6))
        ratios.append(round(ratio, 3))
        print(f"{b:<6}{pool_std:>10.4f}{float(std[i]):>14.4f}{ratio:>9.2f}")

    if all(0.1 < r < 3.0 for r in ratios):
        print("→ All std ratios in 0.1–3x: registry std is on the right unit scale "
              "(sub-1 ratios just reflect the HLS-vs-pretraining domain gap).")
    else:
        bad = [b for b, r in zip(bands, ratios) if not (0.1 < r < 3.0)]
        print(f"→ GROSS scale error for {bad} (ratios {ratios}): the registry std for "
              f"{version} is likely on the wrong unit scale (e.g. raw DN vs 0-1) and "
              f"must be corrected before training.")


if __name__ == "__main__":
    main()
