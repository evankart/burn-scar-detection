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
import yaml

from src.data import normalize_bands, generate_burn_mask, _restore_crs
from src.model import BurnScarModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_inference(
    model: BurnScarModel,
    post_ds: xr.Dataset,
    pre_ds: xr.Dataset,
    bands: list[str],
    patch_size: int = 224,
    device: torch.device = torch.device("cpu"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run sliding-window inference on a full scene.
    Returns (pred_mask, true_mask, image).
    """
    image = normalize_bands(post_ds, bands)
    true_mask = generate_burn_mask(pre_ds, post_ds, dnbr_threshold=0.10)

    _, h, w = image.shape
    pred_mask = np.zeros((h, w), dtype=np.float32)
    count_mask = np.zeros((h, w), dtype=np.float32)

    stride = patch_size // 2
    model.eval()

    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[:, y : y + patch_size, x : x + patch_size]
                if np.isnan(patch).any() or patch.max() == 0:
                    continue

                tensor = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

                pred_mask[y : y + patch_size, x : x + patch_size] += probs
                count_mask[y : y + patch_size, x : x + patch_size] += 1

    valid = count_mask > 0
    pred_mask[valid] /= count_mask[valid]
    pred_binary = (pred_mask > 0.4).astype(np.uint8)

    from scipy.ndimage import binary_erosion
    data_valid = ~(np.isnan(image).any(axis=0) | (image.max(axis=0) == 0))
    data_valid = binary_erosion(data_valid, iterations=10)
    pred_binary[~data_valid] = 0

    # Mask out water using NDWI from raw bands (before z-score normalization).
    # Water has NDWI = (Green - NIR) / (Green + NIR) > 0.1.
    # Bands: B02=0, B03=1(Green), B04=2, B08=3(NIR), B11=4, B12=5
    b03 = post_ds["B03"].values.astype(float)
    b08 = post_ds["B08"].values.astype(float)
    denom = b03 + b08
    denom[denom == 0] = np.nan
    ndwi = (b03 - b08) / denom
    water_mask = ndwi > 0.1
    pred_binary[water_mask] = 0

    return pred_binary, true_mask, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, help="Region name or 'all'")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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

        if not pre_path.exists() or not post_path.exists():
            logger.warning(f"Data not found for {name}. Run download first.")
            continue

        logger.info(f"Running inference on {name}...")
        pre_ds = _restore_crs(xr.open_dataset(pre_path, engine="h5netcdf"))
        post_ds = _restore_crs(xr.open_dataset(post_path, engine="h5netcdf"))
        post_ds = post_ds.rio.reproject_match(pre_ds)

        pred_mask, true_mask, image = run_inference(
            model, post_ds, pre_ds,
            bands=config["data"]["bands"],
            patch_size=config["data"]["patch_size"],
            device=device,
        )

        try:
            from pyproj import Transformer
            crs = post_ds.rio.crs
            native_bounds = post_ds.rio.bounds()
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            min_lon, min_lat = transformer.transform(native_bounds[0], native_bounds[1])
            max_lon, max_lat = transformer.transform(native_bounds[2], native_bounds[3])
            bounds = [[min_lat, min_lon], [max_lat, max_lon]]
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
            bounds=np.array(bounds),
        )
        logger.info(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
