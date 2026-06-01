"""
HLS data pipeline: download, preprocessing, and PyTorch dataset.

Uses NASA Earthdata's Harmonized Landsat Sentinel-2 (HLS) product (same as data Prithvi-EO was pretrained on).
"""

import logging
import platform
from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import numpy as np
import rioxarray  # noqa: F401 — activates rio accessor on xarray
import torch
import xarray as xr
import yaml
from shapely.geometry import box, mapping
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

HLS_COLLECTION = "HLSS30.v2.0"


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Load a YAML config, expanding the single ``data.fires`` registry into the
    role-keyed region lists the rest of the pipeline consumes.

    The source-of-truth config keeps every fire in ONE list, each tagged with a
    ``role`` (``train`` / ``test`` / ``negative``) so the train/test/negative
    splits can never drift apart:

        data:
          fires:
            - {name: woolsey_fire_2018, role: test,  lat: ..., ...}
            - {name: august_complex_2020, role: train, lat: ..., ...}

    This returns the config with ``train_regions`` / ``test_regions`` /
    ``negative_regions`` derived from those roles (``role`` stripped from each
    region dict). Configs that already use the explicit lists (e.g. the derived
    finetune_config.yaml) are passed through unchanged, so this is backward
    compatible.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data = config.get("data", {})
    fires = data.get("fires")
    if fires is not None:
        buckets: dict[str, list] = {"train": [], "test": [], "negative": []}
        for fire in fires:
            role = fire.get("role", "train")
            region = {k: v for k, v in fire.items() if k != "role"}
            buckets.setdefault(role, []).append(region)
        # fires is the source of truth: derived lists overwrite any stale keys.
        data["train_regions"] = buckets["train"]
        data["test_regions"] = buckets["test"]
        if buckets["negative"]:
            data["negative_regions"] = buckets["negative"]
    return config


# --- Download HLS Data ---
class HLSDownloader:
    """Downloads and caches HLS imagery from NASA Earthdata."""

    def __init__(self, config_path: str = "configs/train_config.yaml"):
        self.config = load_config(config_path)

        self.bands = self.config["data"]["bands"]
        self.cache_dir = Path(self.config["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            earthaccess.login(strategy="environment")
        except Exception:
            earthaccess.login(strategy="netrc")

    def _region_bbox(self, lat: float, lon: float, buffer_km: float) -> tuple:
        buffer_deg = buffer_km * 0.009
        return (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)

    def search_scenes(self, bbox: tuple, date_range: str, max_cloud_cover: int = 15) -> list:
        results = earthaccess.search_data(
            short_name="HLSS30",
            bounding_box=bbox,
            temporal=tuple(date_range.split("/")),
            cloud_cover=(0, max_cloud_cover),
            count=15,
        )
        results.sort(key=lambda g: g.get("umm", {}).get("CloudCover", 100))
        logger.info(f"Found {len(results)} HLS scenes for bbox={bbox}, dates={date_range}")
        return results

    def _get_mgrs_tile(self, granule) -> str:
        """Extract MGRS tile ID from HLS granule native-id.
        Format: HLS.S30.T11SLT.2018327T184709.v2.0 → 11SLT
        """
        native_id = granule.get("meta", {}).get("native-id", "")
        parts = native_id.split(".")
        for part in parts:
            if len(part) == 6 and part[0] == "T" and part[1:3].isdigit():
                return part[1:]
        return ""

    def load_scene(self, granule, bbox: tuple) -> xr.Dataset:
        band_arrays = {}
        files = earthaccess.open([granule])

        for band in self.bands:
            matched = [f for f in files if f.path.endswith(f".{band}.tif")]
            if not matched:
                raise ValueError(f"Band {band} not found in granule assets")

            da = rioxarray.open_rasterio(matched[0], mask_and_scale=True)
            geom = box(*bbox)
            da = da.rio.clip([mapping(geom)], crs="EPSG:4326")
            band_arrays[band] = da.squeeze("band", drop=True)

        ref_band = band_arrays[self.bands[0]]
        for band_name, da in band_arrays.items():
            if da.shape != ref_band.shape:
                band_arrays[band_name] = da.rio.reproject_match(ref_band)

        ds = xr.Dataset(band_arrays)
        ds.load()
        return ds

    def _make_target_grid(self, bbox: tuple, resolution: float = 30.0) -> xr.DataArray:
        """
        Build a reference DataArray covering the full bbox in the local UTM zone.
        All scenes are reprojected to this grid so cross-tile mosaics are always
        the right size.
        """
        from pyproj import Transformer

        lon_c = (bbox[0] + bbox[2]) / 2
        lat_c = (bbox[1] + bbox[3]) / 2
        zone = int((lon_c + 180) / 6) + 1
        utm_epsg = f"EPSG:326{zone:02d}" if lat_c >= 0 else f"EPSG:327{zone:02d}"

        transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
        x_min, y_min = transformer.transform(bbox[0], bbox[1])
        x_max, y_max = transformer.transform(bbox[2], bbox[3])

        nx = int(round((x_max - x_min) / resolution))
        ny = int(round((y_max - y_min) / resolution))
        x_coords = x_min + (np.arange(nx) + 0.5) * resolution
        y_coords = y_max - (np.arange(ny) + 0.5) * resolution

        ref = xr.DataArray(
            np.zeros((ny, nx), dtype=np.float32),
            dims=["y", "x"],
            coords={"y": y_coords, "x": x_coords},
        )
        return ref.rio.write_crs(utm_epsg)

    def load_and_merge_scenes(self, granules: list, bbox: tuple, max_scenes: int = 10) -> xr.Dataset:
        """
        Mosaic up to max_scenes onto a fixed UTM grid spanning the full bbox.

        Uses rioxarray.merge, which places every scene at its true geographic
        position, so scenes from different MGRS tiles tile together correctly.
        Granules arrive sorted least-cloudy-first; merge keeps the first valid
        pixel and fills nodata gaps from later scenes. This replaces a per-scene
        reproject_match + manual gap-fill that stretched a single clipped tile
        across the whole grid and short-circuited on the first scene, leaving
        multi-tile bboxes (e.g. fires at a 4-tile junction) mostly nodata.
        """
        import gc
        from rioxarray.merge import merge_arrays

        ref_da = self._make_target_grid(bbox)
        target_crs = ref_da.rio.crs
        bounds = ref_da.rio.bounds()

        per_band: dict[str, list] = {b: [] for b in self.bands}
        loaded = 0
        last_err = None
        for granule in granules[:max_scenes]:
            granule_id = granule.get("meta", {}).get("native-id", "unknown")
            try:
                ds = self.load_scene(granule, bbox)
            except Exception as e:
                last_err = e
                logger.warning(f"Skipping scene {granule_id}: {type(e).__name__}: {e}")
                continue
            for band in self.bands:
                da = ds[band].rio.write_nodata(np.nan)
                if da.rio.crs is not None and da.rio.crs != target_crs:
                    da = da.rio.reproject(target_crs)
                per_band[band].append(da)
            loaded += 1
            del ds
            gc.collect()

        if loaded == 0:
            msg = "No scenes could be loaded"
            if last_err:
                msg += f" (last error: {type(last_err).__name__}: {last_err})"
            raise ValueError(msg)

        merged = {}
        for band in self.bands:
            mosaic = merge_arrays(per_band[band], bounds=bounds, res=(30.0, 30.0), nodata=np.nan)
            merged[band] = mosaic.squeeze(drop=True)
        out = xr.Dataset(merged).rio.write_crs(target_crs)
        for band in self.bands:
            out[band].encoding.clear()
            for k in ("_FillValue", "scale_factor", "add_offset"):
                out[band].attrs.pop(k, None)
        valid_pct = (~np.isnan(out[self.bands[0]].values)).mean() * 100
        logger.info(f"Merged {loaded} scene(s) across tiles: {valid_pct:.1f}% valid pixels")
        return out

    def _filter_to_tile(self, granules: list, tile_id: str) -> list:
        """Keep only granules from a specific MGRS tile."""
        return [g for g in granules if self._get_mgrs_tile(g) == tile_id]

    def _cache_has_bands(self, path: Path) -> bool:
        """True if the cached NetCDF contains every band in self.bands."""
        try:
            with xr.open_dataset(path, engine="h5netcdf") as ds:
                return all(b in ds.variables for b in self.bands)
        except Exception:
            return False

    def download_region(self, region: dict) -> dict[str, Path]:
        name = region["name"]
        pre_path = self.cache_dir / f"{name}_pre.nc"
        post_path = self.cache_dir / f"{name}_post.nc"

        if pre_path.exists() and post_path.exists():
            # Band-validate the cache: Prithvi 1.0 caches store B8A/B11/B12, while
            # 2.0 needs B05/B06/B07. A stale 1.0 NetCDF lacks the 2.0 bands and
            # would KeyError downstream, so re-download if the requested bands are
            # not all present (and vice versa).
            if self._cache_has_bands(post_path) and self._cache_has_bands(pre_path):
                logger.info(f"Using cached data for {name}")
                return {"pre": pre_path, "post": post_path}
            logger.info(
                f"Cached {name} is missing requested bands {self.bands} — re-downloading"
            )

        bbox = self._region_bbox(region["lat"], region["lon"], region["buffer_km"])

        allow_multitile = region.get("allow_multitile", False)

        post_start = region["post_fire_date"]
        post_window_days = region.get("post_fire_window_days", 90)
        post_end_dt = datetime.strptime(post_start, "%Y-%m-%d") + timedelta(days=post_window_days)
        post_granules = self.search_scenes(
            bbox, f"{post_start}/{post_end_dt.strftime('%Y-%m-%d')}", max_cloud_cover=20
        )
        if not post_granules:
            raise ValueError(f"No post-fire HLS scenes found for {name}")

        if allow_multitile:
            target_tile = None
            logger.info(f"Post-fire: multi-tile mosaic ({len(post_granules)} scenes across tiles)")
        else:
            target_tile = self._get_mgrs_tile(post_granules[0])
            post_granules = self._filter_to_tile(post_granules, target_tile)
            logger.info(f"Post-fire: locked to MGRS tile {target_tile} ({len(post_granules)} scenes)")

        pre_start = region["pre_fire_date"]
        pre_granules = []
        for days, max_cc in [(28, 20), (60, 30), (90, 40)]:
            end_dt = datetime.strptime(pre_start, "%Y-%m-%d") + timedelta(days=days)
            candidates = self.search_scenes(
                bbox, f"{pre_start}/{end_dt.strftime('%Y-%m-%d')}", max_cloud_cover=max_cc
            )
            matched = candidates if allow_multitile else self._filter_to_tile(candidates, target_tile)
            if matched:
                pre_granules = matched
                where = "across tiles" if allow_multitile else f"on tile {target_tile}"
                logger.info(f"Pre-fire: found {len(pre_granules)} scenes {where}")
                break
        if not pre_granules:
            raise ValueError(f"No pre-fire HLS scenes found for {name}")

        pre_ds = self.load_and_merge_scenes(pre_granules, bbox)
        pre_ds.to_netcdf(pre_path, engine="h5netcdf")
        logger.info(f"Saved pre-fire scene for {name}: {pre_path}")

        post_ds = self.load_and_merge_scenes(post_granules, bbox)
        post_ds.to_netcdf(post_path, engine="h5netcdf")
        logger.info(f"Saved post-fire scene for {name}: {post_path}")

        return {"pre": pre_path, "post": post_path}

    def download_all(self) -> dict[str, dict[str, Path]]:
        all_regions = (
            self.config["data"].get("train_regions", [])
            + self.config["data"].get("test_regions", [])
        )
        results = {}
        for region in all_regions:
            try:
                results[region["name"]] = self.download_region(region)
            except Exception as e:
                logger.error(f"Failed to download {region['name']}: {e}")
        return results


# --- Preprocessing ---

def _restore_crs(ds: xr.Dataset) -> xr.Dataset:
    """Restore CRS from spatial_ref variable after loading from NetCDF."""
    if "spatial_ref" in ds:
        crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
        if crs_wkt:
            ds = ds.rio.write_crs(crs_wkt)
    return ds


def _compute_spectral_indices(ds: xr.Dataset) -> xr.Dataset:
    nir = ds["B8A"].astype(np.float32)
    swir2 = ds["B12"].astype(np.float32)
    red = ds["B04"].astype(np.float32)
    green = ds["B03"].astype(np.float32)
    eps = 1e-8
    ds["NBR"] = (nir - swir2) / (nir + swir2 + eps)
    ds["NDVI"] = (nir - red) / (nir + red + eps)
    ds["NDWI"] = (green - nir) / (green + nir + eps)
    return ds


def generate_burn_mask(
    pre_ds: xr.Dataset,
    post_ds: xr.Dataset,
    dnbr_threshold: float = 0.10,
) -> np.ndarray:
    """Generate a binary burn scar mask using dNBR. 1 = burned, 0 = unburned."""
    post_ds = post_ds.rio.reproject_match(pre_ds)
    pre_ds = _compute_spectral_indices(pre_ds)
    post_ds = _compute_spectral_indices(post_ds)
    dnbr = pre_ds["NBR"].values - post_ds["NBR"].values
    mask = (dnbr > dnbr_threshold).astype(np.uint8)

    try:
        from scipy.ndimage import binary_closing, binary_opening
        struct = np.ones((3, 3))
        mask = binary_opening(mask, structure=struct, iterations=1).astype(np.uint8)
        mask = binary_closing(mask, structure=struct, iterations=1).astype(np.uint8)
    except ImportError:
        logger.warning("scipy not available, skipping morphological cleanup")

    burned_pct = mask.sum() / mask.size * 100
    logger.info(f"Burn mask: {burned_pct:.1f}% pixels burned (threshold={dnbr_threshold})")
    return mask


def compute_dnbr(pre_ds: xr.Dataset, post_ds: xr.Dataset) -> np.ndarray:
    """Continuous dNBR (pre-fire NBR − post-fire NBR) on the pre-fire grid.

    Higher values = more severe burn. Unlike generate_burn_mask this keeps the
    raw signal (no threshold, no morphology) so it can be binned into USGS
    severity classes for display.

    Computed directly from B8A/B12 with a guarded denominator: HLS surface
    reflectance can be slightly negative (atmospheric correction), and where
    NIR+SWIR2 is approximately 0 the NBR ratio explodes to millions. Those
    degenerate pixels (water, deep shadow, scene edges - never real burns) and
    any non-physical abs(dNBR) > 2 are returned as NaN so the severity overlay
    leaves them transparent.
    """
    post_ds = post_ds.rio.reproject_match(pre_ds)

    def _nbr(ds: xr.Dataset) -> np.ndarray:
        nir = ds["B8A"].values.astype(np.float32)
        swir2 = ds["B12"].values.astype(np.float32)
        denom = nir + swir2
        out = np.full(nir.shape, np.nan, dtype=np.float32)
        ok = np.abs(denom) > 1e-3
        out[ok] = (nir[ok] - swir2[ok]) / denom[ok]
        return out

    dnbr = _nbr(pre_ds) - _nbr(post_ds)
    dnbr[np.abs(dnbr) > 2.0] = np.nan
    return dnbr.astype(np.float32)


def normalize_bands(ds: xr.Dataset, bands: list[str],
                    prithvi_version: str = "1.0") -> np.ndarray:
    """
    Stack and normalize HLS bands using Prithvi pretraining statistics.
    Returns (C, H, W). prithvi_version selects normalization stats + gain.

    Prithvi 1.0 (B02,B03,B04,B8A,B11,B12 — 0-1 reflectance):
      A brightness gain is applied before z-scoring because HLS LaSRC output
      runs ~1.4-1.9x darker than Prithvi 1.0's pretraining distribution.
      This corrected Woolsey IoU from 0.53 → 0.73 (see over_prediction_analysis.md).

    Prithvi 2.0 (B02,B03,B04,B05,B06,B07 — different bands, different stats):
      The 2.0 pretraining used raw DN / 10000, with per-band stats derived from
      a much larger and more diverse dataset (4.2M scenes). No brightness gain
      is applied for 2.0 — whether one is needed should be verified empirically.
    """
    from src.model import PRITHVI_VERSIONS

    vcfg = PRITHVI_VERSIONS[prithvi_version]
    MEAN = vcfg["mean"]
    STD  = vcfg["std"]

    # Brightness gain for 1.0 only — calibrated so training-fire median
    # reflectance matches the Prithvi 1.0 pretraining mean.
    GAIN_1 = [1.8793, 1.7172, 1.5741, 1.4097, 1.1295, 1.1276]
    GAIN = GAIN_1 if prithvi_version == "1.0" else [1.0] * len(bands)

    arrays = []
    for i, band in enumerate(bands):
        arr = ds[band].values.astype(np.float32)
        arr = np.clip(arr, 0, 1)
        arr = arr * GAIN[i]
        arr = (arr - MEAN[i]) / STD[i]
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def create_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 224,
    stride: int | None = None,
    min_burn_fraction: float = 0.01,
    background_keep: float = 0.3,
    min_valid_fraction: float = 0.5,
    max_patches: int | None = None,
) -> list[dict]:
    """Slice image and mask into patches, keeping all burns + a fraction of
    pure-background patches. Burn scars are rare, so retaining every background
    patch would swamp the positive class and bias the model toward predicting
    "unburned"; background_keep caps that imbalance.

    A patch is kept when at least min_valid_fraction of its pixels are valid
    (not nodata); the remaining nodata is imputed to 0 (≈ per-band mean in
    z-scored space). Requiring *every* pixel to be valid — as an earlier
    version did — silently dropped entire large scenes whose every 224x224
    window clipped a nodata gap, starving training of the biggest fires.
    max_patches caps the per-call count so a single mega-fire can't dominate."""
    if stride is None:
        stride = patch_size // 4

    _, h, w = image.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[:, y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            if img_patch.shape[1:] != (patch_size, patch_size) or mask_patch.shape != (patch_size, patch_size):
                continue

            valid = ~(np.isnan(img_patch).any(axis=0) | (np.nan_to_num(img_patch).max(axis=0) == 0))
            if valid.mean() < min_valid_fraction:
                continue
            img_patch = np.nan_to_num(img_patch, nan=0.0)

            burn_frac = mask_patch.sum() / mask_patch.size
            if burn_frac >= min_burn_fraction or np.random.random() < background_keep:
                patches.append({"image": img_patch, "mask": mask_patch, "burn_fraction": burn_frac})

    if max_patches is not None and len(patches) > max_patches:
        idx = np.random.choice(len(patches), size=max_patches, replace=False)
        patches = [patches[i] for i in idx]

    logger.info(
        f"Created {len(patches)} patches ({sum(1 for p in patches if p['burn_fraction'] > 0)} with burns)"
    )
    return patches


def process_region(
    pre_path: Path,
    post_path: Path,
    bands: list[str],
    patch_size: int = 224,
    region_name: str = "",
    dnbr_threshold: float = 0.10,
    background_keep: float = 0.3,
    max_patches: int | None = None,
    prithvi_version: str = "1.0",
) -> list[dict]:
    """Full preprocessing pipeline for a single region."""
    pre_ds = _restore_crs(xr.open_dataset(pre_path, engine="h5netcdf"))
    post_ds = _restore_crs(xr.open_dataset(post_path, engine="h5netcdf"))
    post_ds = post_ds.rio.reproject_match(pre_ds)

    mask = generate_burn_mask(pre_ds, post_ds, dnbr_threshold=dnbr_threshold)
    image = normalize_bands(post_ds, bands, prithvi_version=prithvi_version)
    patches = create_patches(
        image, mask, patch_size=patch_size,
        background_keep=background_keep, max_patches=max_patches,
    )

    for p in patches:
        p["region_name"] = region_name

    return patches


# --- Dataset ---
class BurnScarDataset(Dataset):
    """PyTorch dataset for burn scar segmentation patches."""

    def __init__(self, patches: list[dict], augment: bool = False, augment_config: dict | None = None):
        self.patches = patches
        self.augment = augment
        self.augment_config = augment_config or {}

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int) -> dict:
        patch = self.patches[idx]
        image = np.ascontiguousarray(patch["image"])
        mask = np.ascontiguousarray(patch["mask"])

        if self.augment:
            image, mask = self._apply_augmentations(image, mask)

        return {
            "pixel_values": torch.from_numpy(image).float(),
            "labels": torch.from_numpy(mask).long(),
        }

    def _apply_augmentations(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.augment_config.get("random_flip", True):
            if np.random.random() > 0.5:
                image = np.flip(image, axis=-1).copy()
                mask = np.flip(mask, axis=-1).copy()
            if np.random.random() > 0.5:
                image = np.flip(image, axis=-2).copy()
                mask = np.flip(mask, axis=-2).copy()

        if self.augment_config.get("random_rotate", True):
            k = np.random.randint(0, 4)
            image = np.rot90(image, k, axes=(-2, -1)).copy()
            mask = np.rot90(mask, k, axes=(-2, -1)).copy()

        if self.augment_config.get("random_resized_crop", False) and np.random.random() > 0.5:
            import torch
            import torch.nn.functional as F
            C, H, W = image.shape
            scale = np.random.uniform(0.7, 1.0)
            ch, cw = max(1, int(H * scale)), max(1, int(W * scale))
            y0 = np.random.randint(0, H - ch + 1)
            x0 = np.random.randint(0, W - cw + 1)
            it = torch.from_numpy(image[:, y0:y0 + ch, x0:x0 + cw]).unsqueeze(0).float()
            it = F.interpolate(it, size=(H, W), mode="bilinear", align_corners=False)
            image = it.squeeze(0).numpy()
            mt = torch.from_numpy(mask[y0:y0 + ch, x0:x0 + cw].astype(np.float32))[None, None]
            mt = F.interpolate(mt, size=(H, W), mode="nearest")
            mask = mt.squeeze(0).squeeze(0).numpy().astype(np.int64)

        return image, mask


def create_dataloaders(patches: dict[str, list[dict]], config: dict) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from pre-split patch dicts.
    patches must have keys "train", "val", and optionally "test".
    """
    train_patches = patches["train"]
    val_patches = patches["val"]
    test_patches = patches.get("test", [])

    logger.info(f"Split: {len(train_patches)} train, {len(val_patches)} val, {len(test_patches)} test")

    aug_config = config["training"].get("augmentations", {})
    batch_size = config["training"]["batch_size"]

    if platform.system() == "Darwin":
        num_workers = 0
        pin_memory = False
    else:
        num_workers = config["data"]["num_workers"]
        pin_memory = torch.cuda.is_available()

    return {
        "train": DataLoader(
            BurnScarDataset(train_patches, augment=True, augment_config=aug_config),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        ),
        "val": DataLoader(
            BurnScarDataset(val_patches, augment=False),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        "test": DataLoader(
            BurnScarDataset(test_patches, augment=False),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
    }
