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
from src.utils import get_device, water_mask, cloud_over_water_mask

logger = logging.getLogger(__name__)
_CONFIG = "configs/train_config.yaml"
HF_REPO = "evankart/burn-scar-detection-data"


def load_model(checkpoint: str = "checkpoints/finetune_v3/best_model.pt",
               config_path: str = _CONFIG):
    """Load the deployed model once (cache with st.cache_resource in the app)."""
    cfg = load_config(config_path)
    device = get_device()
    # Fetch the checkpoint from HF if it isn't present locally (cloud deploy).
    if not Path(checkpoint).exists():
        from huggingface_hub import hf_hub_download
        logger.info(f"Checkpoint not local — downloading {checkpoint} from {HF_REPO}")
        hf_hub_download(repo_id=HF_REPO, repo_type="dataset", filename=checkpoint, local_dir=".")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model = BurnScarModel(num_classes=cfg["model"]["num_classes"],
                          in_channels=cfg["model"]["in_channels"])
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


_RGB_BANDS = ["B04", "B03", "B02"]
_PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
_PC_TILES = "https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{z}/{x}/{y}@1x"


def fetch_preview_tiles(bbox: tuple, post_date: str, window_days: int = 30) -> dict:
    """Find all S2 tiles covering bbox and return a mosaicked, cropped RGB PNG.

    Uses PC STAC to find scenes per MGRS tile (~1s), fetches pre-cropped PNGs
    from titiler.xyz (~2s/tile), composites them into a seamless mosaic.
    Returns {image_b64, scene_date, cloud_cover}.
    """
    import base64
    import io
    import pystac_client
    import planetary_computer
    import requests as req
    from PIL import Image as PILImage

    catalog = pystac_client.Client.open(_PC_STAC, modifier=planetary_computer.sign_inplace)
    dt_start = datetime.strptime(post_date, "%Y-%m-%d")
    # Cap end date to 3 days ago — Sentinel-2 processing + PC ingestion takes 1–3 days.
    max_end = datetime.utcnow() - timedelta(days=3)

    def _search(days: int, cloud_lt: int = 50):
        end = min(dt_start + timedelta(days=days), max_end)
        if end <= dt_start:
            return []
        return list(catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{post_date}/{end.strftime('%Y-%m-%d')}",
            query={"eo:cloud_cover": {"lt": cloud_lt}},
            sortby="+properties.eo:cloud_cover",
            max_items=20,
        ).items())

    items = _search(14) or _search(window_days)
    if not items:
        # Check if passes exist at all (no cloud filter) to give a specific error.
        all_items = _search(window_days, cloud_lt=100)
        if not all_items:
            raise ValueError(
                "No Sentinel-2 passes found over this area in the 30-day window. "
                "Try a different date or a slightly larger area."
            )
        covers = sorted(round(i.properties.get("eo:cloud_cover", 100)) for i in all_items)
        raise ValueError(
            f"Sentinel-2 passes found but all have high cloud cover "
            f"(lowest: {covers[0]}%). Try a date with clearer skies."
        )

    # One item per MGRS tile (least cloudy wins per tile)
    tile_items: dict = {}
    for item in items:
        tile = item.properties.get("s2:mgrs_tile") or item.id.split("_T")[-1][:5]
        if tile not in tile_items:
            tile_items[tile] = item

    best = min(tile_items.values(), key=lambda i: i.properties.get("eo:cloud_cover", 100))
    scene_date = best.datetime.strftime("%Y-%m-%d")
    cloud_cover = round(best.properties.get("eo:cloud_cover", 0), 1)

    min_lon, min_lat, max_lon, max_lat = bbox
    composite: np.ndarray | None = None

    for item in tile_items.values():
        visual_href = item.assets["visual"].href
        crop_url = (
            f"https://titiler.xyz/cog/bbox/{min_lon},{min_lat},{max_lon},{max_lat}.png"
            f"?url={req.utils.quote(visual_href)}&max_size=1024&nodata=0"
        )
        r = req.get(crop_url, timeout=30)
        if r.status_code != 200:
            continue
        img = PILImage.open(io.BytesIO(r.content)).convert("RGBA")
        if composite is None:
            composite = np.array(img)
        else:
            # Resize to match composite if titiler returns off-by-one dimensions
            # due to per-tile reprojection rounding.
            if img.size != (composite.shape[1], composite.shape[0]):
                img = img.resize((composite.shape[1], composite.shape[0]), PILImage.LANCZOS)
            arr = np.array(img)
            empty = (composite[..., :3].sum(axis=-1) == 0) | (composite[..., 3] == 0)
            composite[empty] = arr[empty]

    if composite is None:
        raise ValueError("Failed to fetch any scene tiles for this area.")

    buf = io.BytesIO()
    PILImage.fromarray(composite, mode="RGBA").save(buf, format="PNG")
    return {
        "image_b64": base64.b64encode(buf.getvalue()).decode(),
        "scene_date": scene_date,
        "cloud_cover": cloud_cover,
        # MGRS tiles composited into this mosaic, and each tile's acquisition date.
        # Lets the app warn when an AOI spans tiles (possibly from different dates).
        "tiles": sorted(tile_items.keys()),
        "tile_dates": {t: it.datetime.strftime("%Y-%m-%d") for t, it in tile_items.items()},
    }


def fetch_scene(bbox: tuple, post_date: str, cfg, window_days: int = 30) -> dict:
    """Download RGB-only preview scene — no inference, ~half the download of full detection.

    Returns {image (3,H,W normalized R/G/B), bounds [[S,W],[N,E]], scene_date, n_scenes,
             valid_frac, post_ds}.
    Raises ValueError if no clear scene is found.
    """
    dl = HLSDownloader(config_path=_CONFIG)
    # Temporarily override bands to RGB only for fast preview download.
    dl.bands = _RGB_BANDS
    end = (datetime.strptime(post_date, "%Y-%m-%d") + timedelta(days=window_days)).strftime("%Y-%m-%d")
    granules = dl.search_scenes(tuple(bbox), f"{post_date}/{end}", max_cloud_cover=20)
    if not granules:
        raise ValueError("No clear HLS scene found within ~30 days. Try another date or location.")
    scene_date = _granule_date(granules[0])
    post_ds = dl.load_and_merge_scenes(granules, tuple(bbox))
    image = normalize_bands(post_ds, _RGB_BANDS)
    valid_frac = float((~(np.isnan(image).any(axis=0) | (np.nan_to_num(image).max(axis=0) == 0))).mean())
    return {
        "image": image,
        "bounds": _bounds_latlon(post_ds),
        "scene_date": scene_date,
        "n_scenes": len(granules),
        "valid_frac": valid_frac,
        "post_ds": post_ds,
    }


def detect_burn_scar(bbox: tuple, post_date: str, model, device, cfg,
                     window_days: int = 30, patch_size: int = 224,
                     pred_threshold: float | None = None,
                     prefetched: dict | None = None) -> dict:
    """
    bbox = (min_lon, min_lat, max_lon, max_lat); post_date = 'YYYY-MM-DD'.
    Pass prefetched=fetch_scene(...) to skip the download step.
    Returns {pred_mask (H,W uint8), image (6,H,W normalized), bounds [[S,W],[N,E]],
             burned_frac, scene_date, n_scenes}.
    Raises ValueError if no clear scene is found.
    """
    bands = cfg["data"]["bands"]
    if pred_threshold is None:
        pred_threshold = cfg["data"].get("pred_threshold", 0.5)

    _fmask_raw = None
    if prefetched is not None:
        image = prefetched["image"]
        post_ds = prefetched["post_ds"]
        scene_date = prefetched["scene_date"]
        n_scenes = prefetched["n_scenes"]
        bounds = prefetched["bounds"]
    else:
        dl = HLSDownloader(config_path=_CONFIG)
        end = (datetime.strptime(post_date, "%Y-%m-%d") + timedelta(days=window_days)).strftime("%Y-%m-%d")
        granules = dl.search_scenes(tuple(bbox), f"{post_date}/{end}", max_cloud_cover=50)
        if not granules:
            raise ValueError("No clear HLS scene found for that area within ~30 days of "
                             "the date. Try another date or location.")
        scene_date = _granule_date(granules[0])
        n_scenes = len(granules)
        post_ds = dl.load_and_merge_scenes(granules, tuple(bbox))
        image = normalize_bands(post_ds, bands)
        bounds = _bounds_latlon(post_ds)
        _fmask_raw = dl.load_fmask(granules[0], tuple(bbox))

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

    # NDWI+MNDWI water mask + cloud-over-water mask (fallback when Fmask unavailable)
    water = water_mask(post_ds, threshold=cfg["data"].get("water_ndwi_threshold", 0.0))
    clouds = cloud_over_water_mask(post_ds)
    combined = water | clouds
    if pad_h or pad_w:
        combined = np.pad(combined, ((0, pad_h), (0, pad_w)), constant_values=False)
    pred[combined] = 0

    # HLS Fmask: zero out cloud (bit 1) and cloud shadow (bit 3) pixels.
    if _fmask_raw is not None:
        fmask = _fmask_raw
        if fmask.shape != (h, w):
            from PIL import Image as _PIL
            fmask = np.array(_PIL.fromarray(fmask).resize((w, h), _PIL.NEAREST))
        cloud_mask = ((fmask & 0b00001010) != 0)
        if pad_h or pad_w:
            cloud_mask = np.pad(cloud_mask, ((0, pad_h), (0, pad_w)), constant_values=False)
        pred[cloud_mask] = 0

    # crop back to the true scene size
    pred = pred[:h, :w]
    image = image[:, :h, :w]
    valid = valid_px[:h, :w]

    burned_frac = float(pred[valid].mean()) if valid.any() else 0.0
    return {
        "pred_mask": pred,
        "image": image,
        "bounds": bounds,
        "burned_frac": burned_frac,
        "n_scenes": n_scenes,
        "scene_date": scene_date,
    }
