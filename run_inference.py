"""
Run inference on a region and cache results for the Streamlit app.

Usage:
    python run_inference.py --region woolsey_fire_2018
    python run_inference.py --region all
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from src.data import normalize_bands, generate_burn_mask, compute_dnbr, _restore_crs, load_config, FMASK_BAD_BITS
from src.model import BurnScarModel
from src.utils import get_device, water_mask, cloud_over_water_mask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_inference(
    model: BurnScarModel,
    post_ds: xr.Dataset,
    pre_ds: xr.Dataset,
    bands: list[str],
    patch_size: int = 224,
    device: torch.device = torch.device("cpu"),
    dnbr_threshold: float = 0.10,
    pred_threshold: float = 0.4,
    return_prob: bool = False,
    water_ndwi_threshold: float = 0.0,
) -> tuple[np.ndarray, ...]:
    """Sliding-window inference on a full scene. Returns (pred_mask, true_mask,
    image); with return_prob=True also appends the pre-threshold probability map.
    dnbr_threshold must match the training-label value. See README."""
    image = normalize_bands(post_ds, bands)
    true_mask = generate_burn_mask(pre_ds, post_ds, dnbr_threshold=dnbr_threshold)

    _, h, w = image.shape
    pred_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)

    # Per-pixel validity: nodata (NaN from Fmask-during-download or sensor gaps) excluded.
    valid_px = ~(np.isnan(image).any(axis=0) | (np.nan_to_num(image).max(axis=0) == 0))

    # Pre-inference cloud/water exclusion — applied before the model sees any pixels.
    # Fmask cloud flag (bit 2) is intentionally excluded here: it triggers on smoke/haze
    # over burned land (observed: Thomas fire recall 0.73 → 0.26 when included).
    # water_mask + cloud_over_water_mask are sufficient to prevent ocean/fog false positives.
    pre_cloud = water_mask(post_ds, threshold=water_ndwi_threshold)
    valid_px &= ~pre_cloud

    stride = patch_size // 2
    model.eval()

    # Edge-anchored grid: final row/col flush to the edge so border strips are covered.
    ys = list(range(0, h - patch_size + 1, stride))
    xs = list(range(0, w - patch_size + 1, stride))
    if ys and ys[-1] != h - patch_size:
        ys.append(h - patch_size)
    if xs and xs[-1] != w - patch_size:
        xs.append(w - patch_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                pv = valid_px[y : y + patch_size, x : x + patch_size]
                if not pv.any():
                    continue
                # Impute nodata to 0; only valid pixels are accumulated.
                patch = np.nan_to_num(image[:, y : y + patch_size, x : x + patch_size], nan=0.0)
                tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                probs = torch.softmax(model(tensor), dim=1)[0, 1].cpu().numpy()

                region = pred_mask[y : y + patch_size, x : x + patch_size]
                creg = count_mask[y : y + patch_size, x : x + patch_size]
                region[pv] += probs[pv]
                creg[pv] += 1

    covered = count_mask > 0
    pred_mask[covered] /= count_mask[covered]
    pred_binary = (pred_mask > pred_threshold).astype(np.uint8)

    # Trim a thin border of the valid region (predictions abutting nodata are unreliable).
    from scipy.ndimage import binary_erosion
    trim = ~binary_erosion(valid_px, iterations=2)
    pred_binary[trim] = 0

    # Zero out cloud/water pixels in pred and true masks.
    pred_binary[pre_cloud] = 0
    true_mask[pre_cloud] = 0

    if return_prob:
        prob = pred_mask.copy()
        prob[trim] = 0.0
        prob[~covered] = 0.0
        prob[pre_cloud] = 0.0
        return pred_binary, true_mask, image, prob

    return pred_binary, true_mask, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="Region name or 'all'")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/finetune_v3/best_model.pt")
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()

    model = BurnScarModel(
        num_classes=config["model"]["num_classes"],
        in_channels=config["model"]["in_channels"],
    )

    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint: {args.checkpoint}")
    else:
        logger.warning(f"No checkpoint found at {args.checkpoint}, using random weights")

    model = model.to(device)

    all_regions = config["data"].get("test_regions", []) + config["data"].get("train_regions", [])
    if args.region == "all":
        regions = all_regions
    else:
        regions = [r for r in all_regions if r["name"] == args.region]
        if not regions:
            logger.error(f"Region '{args.region}' not found in config")
            return

    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        name = region["name"]
        pre_path = Path(config["data"]["cache_dir"]) / f"{name}_pre.nc"
        post_path = Path(config["data"]["cache_dir"]) / f"{name}_post.nc"

        for path in (pre_path, post_path):
            if not path.exists():
                import subprocess
                s3_key = f"s3://burn-scar-detection/hls-cache/{path.name}"
                logger.info(f"{path.name} not cached — pulling from {s3_key}")
                r = subprocess.run(
                    ["aws", "s3", "cp", s3_key, str(path), "--region", "us-west-2"],
                    capture_output=True,
                )
                if r.returncode != 0:
                    logger.warning(f"Data not found for {name} locally or on S3. Run download first.")
                    break
        else:
            pass  # both paths exist, continue below
        if not pre_path.exists() or not post_path.exists():
            continue

        logger.info(f"Running inference on {name}...")
        pre_ds = _restore_crs(xr.open_dataset(pre_path, engine="h5netcdf"))
        post_ds = _restore_crs(xr.open_dataset(post_path, engine="h5netcdf"))
        post_ds = post_ds.rio.reproject_match(pre_ds)

        ndwi_thresh = config["data"].get("water_ndwi_threshold", 0.0)
        pred_mask, true_mask, image = run_inference(
            model, post_ds, pre_ds,
            bands=config["data"]["bands"],
            patch_size=config["data"]["patch_size"],
            device=device,
            dnbr_threshold=config["data"].get("dnbr_threshold", 0.10),
            pred_threshold=config["data"].get("pred_threshold", 0.4),
            water_ndwi_threshold=ndwi_thresh,
        )

        # Continuous dNBR for the severity overlay — also mask water/clouds.
        dnbr = compute_dnbr(pre_ds, post_ds)
        water = water_mask(post_ds, threshold=ndwi_thresh)
        clouds = cloud_over_water_mask(post_ds)
        if "Fmask" in post_ds:
            clouds |= (post_ds["Fmask"].values.astype(np.uint8) & FMASK_BAD_BITS) != 0
        combined = water | clouds
        if combined.shape == dnbr.shape:
            dnbr[combined] = np.nan

        try:
            from pyproj import Transformer
            crs = post_ds.rio.crs
            minx, miny, maxx, maxy = post_ds.rio.bounds()
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            # Densify all four UTM edges and take min/max so the lat/lon bbox
            # encloses the rotated footprint.
            n = 50
            xs_edge = np.linspace(minx, maxx, n)
            ys_edge = np.linspace(miny, maxy, n)
            ex = np.concatenate([xs_edge, xs_edge, np.full(n, minx), np.full(n, maxx)])
            ey = np.concatenate([np.full(n, miny), np.full(n, maxy), ys_edge, ys_edge])
            lon, lat = transformer.transform(ex, ey)
            bounds = [[float(np.min(lat)), float(np.min(lon))],
                      [float(np.max(lat)), float(np.max(lon))]]
        except Exception:
            buffer = region["buffer_km"] * 0.009
            bounds = [
                [region["lat"] - buffer, region["lon"] - buffer],
                [region["lat"] + buffer, region["lon"] + buffer],
            ]

        out_path = output_dir / f"{name}.npz"
        np.savez_compressed(
            out_path,
            pred_mask=pred_mask,
            true_mask=true_mask,
            image=image,
            dnbr=dnbr,
            bounds=np.array(bounds),
        )
        logger.info(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
