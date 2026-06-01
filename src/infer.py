"""On-demand burn-scar detection for the Streamlit app.

Given a user-drawn AOI and a post-fire date, download the post-fire HLS
scene, run the model, and return the predicted burn mask + a display image +
lat/lon bounds.

"""
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from src.data import HLSDownloader, normalize_bands, load_config
from src.model import BurnScarModel
from src.utils import get_device, water_mask

logger = logging.getLogger(__name__)
_CONFIG = "configs/train_config.yaml"
HF_REPO = "evankart/burn-scar-detection-data"


def load_model(checkpoint: str = "checkpoints/balanced_chaparral/best_model.pt",
               config_path: str = _CONFIG):
    """Load the deployed model once (cache with st.cache_resource in the app)."""
    cfg = load_config(config_path)
    device = get_device()
    model = BurnScarModel(num_classes=cfg["model"]["num_classes"],
                          in_channels=cfg["model"]["in_channels"])
    # Fetch the checkpoint from HF if it isn't present locally (cloud deploy).
    if not Path(checkpoint).exists():
        from huggingface_hub import hf_hub_download
        logger.info(f"Checkpoint not local — downloading {checkpoint} from {HF_REPO}")
        hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=checkpoint, local_dir=".")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()
    logger.info(f"Loaded {checkpoint} on {device}")
    return model, device, cfg


def _granule_date(granule) -> str:
    """Parse the acquisition date from an HLS granule native-id, e.g. HLS.S30.T11SLT.2018327T184709.v2.0 -> '2018-11-23'."""
    try:
        nid = granule.get("meta", {}).get("native-id", "")
        token = nid.split(".")[3]          # '2018327T184709'
        year, doy = int(token[:4]), int(token[4:7])
        return (datetime(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def _bounds_latlon(post_ds) -> list:
    """4-corner densified UTM->WGS84 bbox (encloses the rotated footprint)."""
    from pyproj import Transformer
    crs = post_ds.rio.crs
    minx, miny, maxx, maxy = post_ds.rio.bounds()
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    n = 50
    xe = np.linspace(minx, maxx, n); ye = np.linspace(miny, maxy, n)
    ex = np.concatenate([xe, xe, np.full(n, minx), np.full(n, maxx)])
    ey = np.concatenate([np.full(n, miny), np.full(n, maxy), ye, ye])
    lon, lat = t.transform(ex, ey)
    return [[float(np.min(lat)), float(np.min(lon))],
            [float(np.max(lat)), float(np.max(lon))]]


def detect_burn_scar(bbox: tuple, post_date: str, model, device, cfg,
                     window_days: int = 30, patch_size: int = 224,
                     pred_threshold: float | None = None) -> dict:
    """
    bbox = (min_lon, min_lat, max_lon, max_lat); post_date = 'YYYY-MM-DD'.
    Returns {pred_mask (H,W uint8), image (6,H,W normalized), bounds [[S,W],[N,E]],
             burned_frac, scene_date, n_scenes}.
    Raises ValueError if no clear scene is found.
    """
    bands = cfg["data"]["bands"]
    if pred_threshold is None:
        pred_threshold = cfg["data"].get("pred_threshold", 0.5)

    dl = HLSDownloader(config_path=_CONFIG)
    end = (datetime.strptime(post_date, "%Y-%m-%d") + timedelta(days=window_days)).strftime("%Y-%m-%d")
    granules = dl.search_scenes(tuple(bbox), f"{post_date}/{end}", max_cloud_cover=20)
    if not granules:
        raise ValueError("No clear HLS scene found for that area within ~30 days of "
                         "the date. Try another date or location.")
    # search_scenes sorts least-cloudy first → granules[0] is the primary scene.
    scene_date = _granule_date(granules[0])
    post_ds = dl.load_and_merge_scenes(granules, tuple(bbox))

    image = normalize_bands(post_ds, bands)  # brightness gain applied here
    _, h, w = image.shape

    pad_h, pad_w = max(0, patch_size - h), max(0, patch_size - w)
    if pad_h or pad_w:
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=np.nan)
    _, H, W = image.shape

    valid_px = ~(np.isnan(image).any(axis=0) | (np.nan_to_num(image).max(axis=0) == 0))
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    stride = patch_size // 2
    ys = list(range(0, H - patch_size + 1, stride)) or [0]
    xs = list(range(0, W - patch_size + 1, stride)) or [0]
    if ys[-1] != H - patch_size: ys.append(H - patch_size)
    if xs[-1] != W - patch_size: xs.append(W - patch_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                pv = valid_px[y:y + patch_size, x:x + patch_size]
                if not pv.any():
                    continue
                patch = np.nan_to_num(image[:, y:y + patch_size, x:x + patch_size], nan=0.0)
                t = torch.from_numpy(patch).unsqueeze(0).float().to(device)
                probs = torch.softmax(model(t), dim=1)[0, 1].cpu().numpy()
                acc[y:y + patch_size, x:x + patch_size][pv] += probs[pv]
                cnt[y:y + patch_size, x:x + patch_size][pv] += 1

    covered = cnt > 0
    prob = np.zeros((H, W), np.float32)
    prob[covered] = acc[covered] / cnt[covered]
    pred = (prob > pred_threshold).astype(np.uint8)

    from scipy.ndimage import binary_erosion
    pred[~binary_erosion(valid_px, iterations=10)] = 0

    # NDWI water mask
    water = water_mask(post_ds, threshold=cfg["data"].get("water_ndwi_threshold", 0.0))
    if pad_h or pad_w:
        water = np.pad(water, ((0, pad_h), (0, pad_w)), constant_values=False)
    pred[water] = 0

    # crop back to the true scene size
    pred = pred[:h, :w]
    image = image[:, :h, :w]
    valid = valid_px[:h, :w]

    burned_frac = float(pred[valid].mean()) if valid.any() else 0.0
    return {
        "pred_mask": pred,
        "image": image,
        "bounds": _bounds_latlon(post_ds),
        "burned_frac": burned_frac,
        "n_scenes": len(granules),
        "scene_date": scene_date,
    }
