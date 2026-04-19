"""
Sentinel-2 data pipeline: download, preprocessing, and PyTorch dataset.
"""

import logging
import platform
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rioxarray  # noqa: F401 — activates rio accessor on xarray
import torch
import xarray as xr
import yaml
from pystac_client import Client
from shapely.geometry import box, mapping
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

PLANETARY_COMPUTER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_2_COLLECTION = "sentinel-2-l2a"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

class SentinelDownloader:
    """Downloads and caches Sentinel-2 L2A imagery from Planetary Computer."""

    def __init__(self, config_path: str = "configs/train_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.bands = self.config["data"]["bands"]
        self.cache_dir = Path(self.config["data"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = Client.open(PLANETARY_COMPUTER_URL)

    def _region_bbox(self, lat: float, lon: float, buffer_km: float) -> tuple:
        buffer_deg = buffer_km * 0.009
        return (lon - buffer_deg, lat - buffer_deg, lon + buffer_deg, lat + buffer_deg)

    def search_scenes(self, bbox: tuple, date_range: str, max_cloud_cover: int = 15) -> list:
        search = self.catalog.search(
            collections=[SENTINEL_2_COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
            max_items=15,
        )
        items = list(search.items())
        logger.info(f"Found {len(items)} scenes for bbox={bbox}, dates={date_range}")
        return items

    def load_scene(self, item, bbox: tuple) -> xr.Dataset:
        signed_item = pc.sign(item)
        band_arrays = {}

        for band in self.bands:
            asset = signed_item.assets[band]
            da = rioxarray.open_rasterio(asset.href, chunks={"x": 512, "y": 512})
            geom = box(*bbox)
            da = da.rio.clip([mapping(geom)], crs="EPSG:4326")
            if band in ("B11", "B12"):
                da = da.rio.reproject(da.rio.crs, resolution=10, resampling=1)
            band_arrays[band] = da.squeeze("band", drop=True)

        ref_band = band_arrays[self.bands[0]]
        for band_name, da in band_arrays.items():
            if da.shape != ref_band.shape:
                band_arrays[band_name] = da.rio.reproject_match(ref_band)

        ds = xr.Dataset(band_arrays)
        ds.load()  # force dask computation now so I/O errors surface inside try/except callers
        return ds

    def _make_target_grid(self, bbox: tuple, resolution: float = 10.0) -> xr.DataArray:
        """
        Build a reference DataArray covering the full bbox in the local UTM zone.
        All scenes are reprojected to this grid so cross-tile mosaics are always
        the right size — the output grid is derived from the bbox, not from
        whichever scene happens to load first.
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

    def load_and_merge_scenes(self, items: list, bbox: tuple, max_scenes: int = 10) -> xr.Dataset:
        """
        Load up to max_scenes and mosaic them onto a fixed output grid defined by
        the bbox (not by the first scene's extent).  This handles cross-MGRS-tile
        mosaics correctly: scenes from adjacent tiles (e.g. T10SFJ + T10TFK) are
        each reprojected to the same UTM grid before merging, so the output always
        covers the full requested area.  Scenes are processed one at a time to
        avoid OOM on large AOIs.
        """
        import gc

        ref_da = self._make_target_grid(bbox)

        merged = None
        for item in items[:max_scenes]:
            try:
                ds = self.load_scene(item, bbox)
            except Exception as e:
                logger.warning(f"Skipping scene {item.id}: {e}")
                continue

            # Reproject onto the fixed target grid regardless of source tile
            try:
                ds = ds.rio.reproject_match(ref_da)
                ds.load()
            except Exception as e:
                logger.warning(f"Skipping scene {item.id} during reproject: {e}")
                del ds
                gc.collect()
                continue

            if merged is None:
                merged = ds
                valid_pct = (~((merged[self.bands[0]].values == 0) |
                               np.isnan(merged[self.bands[0]].values))).mean() * 100
                logger.info(f"First scene {item.id}: {valid_pct:.1f}% valid pixels")
                if valid_pct > 90:
                    break
                continue

            # Fill zero/NaN pixels in the merged dataset, then free the fill scene
            for band in self.bands:
                base = merged[band].values
                fill = ds[band].values
                nodata = (base == 0) | np.isnan(base)
                if nodata.any():
                    base[nodata] = fill[nodata]
                    merged[band].values[:] = base

            del ds
            gc.collect()

            valid_pct = (~((merged[self.bands[0]].values == 0) |
                           np.isnan(merged[self.bands[0]].values))).mean() * 100
            logger.info(f"After merging scene {item.id}: {valid_pct:.1f}% valid pixels")
            if valid_pct > 90:
                break

        if merged is None:
            raise ValueError("No scenes could be loaded")
        return merged

    @staticmethod
    def _filter_to_tile(items: list, tile_id: str) -> list:
        """Keep only STAC items from a specific MGRS tile."""
        return [it for it in items if it.properties.get("s2:mgrs_tile", "") == tile_id]

    def download_region(self, region: dict) -> dict[str, Path]:
        name = region["name"]
        pre_path = self.cache_dir / f"{name}_pre.nc"
        post_path = self.cache_dir / f"{name}_post.nc"

        if pre_path.exists() and post_path.exists():
            logger.info(f"Using cached data for {name}")
            return {"pre": pre_path, "post": post_path}

        bbox = self._region_bbox(region["lat"], region["lon"], region["buffer_km"])

        # --- Post-fire: item[0] (lowest cloud cover) determines the target MGRS tile ---
        post_start = region["post_fire_date"]
        post_window_days = region.get("post_fire_window_days", 90)
        post_end_dt = datetime.strptime(post_start, "%Y-%m-%d") + timedelta(days=post_window_days)
        post_items = self.search_scenes(bbox, f"{post_start}/{post_end_dt.strftime('%Y-%m-%d')}", max_cloud_cover=20)
        if not post_items:
            raise ValueError(f"No post-fire scenes found for {name}")

        target_tile = post_items[0].properties.get("s2:mgrs_tile", "")
        post_items = self._filter_to_tile(post_items, target_tile)
        logger.info(f"Post-fire: locked to MGRS tile {target_tile} ({len(post_items)} scenes)")

        # --- Pre-fire: search for scenes on the SAME tile, widening window if needed ---
        pre_start = region["pre_fire_date"]
        pre_items = []
        for days, max_cc in [(28, 20), (60, 30), (90, 40)]:
            end_dt = datetime.strptime(pre_start, "%Y-%m-%d") + timedelta(days=days)
            candidates = self.search_scenes(
                bbox, f"{pre_start}/{end_dt.strftime('%Y-%m-%d')}", max_cloud_cover=max_cc
            )
            matched = self._filter_to_tile(candidates, target_tile)
            if matched:
                pre_items = matched
                logger.info(f"Pre-fire: found {len(pre_items)} scenes on tile {target_tile}")
                break
        if not pre_items:
            raise ValueError(f"No pre-fire scenes found on tile {target_tile} for {name}")

        pre_ds = self.load_and_merge_scenes(pre_items, bbox)
        pre_ds.to_netcdf(pre_path, engine="h5netcdf")
        logger.info(f"Saved pre-fire scene for {name}: {pre_path}")

        post_ds = self.load_and_merge_scenes(post_items, bbox)
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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _restore_crs(ds: xr.Dataset) -> xr.Dataset:
    """Restore CRS from spatial_ref variable after loading from NetCDF."""
    if "spatial_ref" in ds:
        crs_wkt = ds["spatial_ref"].attrs.get("crs_wkt")
        if crs_wkt:
            ds = ds.rio.write_crs(crs_wkt)
    return ds


def _compute_spectral_indices(ds: xr.Dataset) -> xr.Dataset:
    nir = ds["B08"].astype(np.float32)
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
    dnbr_threshold: float = 0.27,
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


def normalize_bands(ds: xr.Dataset, bands: list[str]) -> np.ndarray:
    """
    Stack and normalize Sentinel-2 bands using Prithvi pretraining statistics.

    Divides by 10000 (Sentinel-2 L2A scale factor) then applies per-band
    z-score normalization with the mean/std from Prithvi-EO-1.0-100M's
    HLS pretraining data. Returns (C, H, W).
    """
    # Prithvi pretraining stats (DN ÷ 10000 scale)
    MEAN = [0.077523, 0.108099, 0.122859, 0.249720, 0.220421, 0.161083]
    STD  = [0.128153, 0.127003, 0.139948, 0.136834, 0.129168, 0.115451]

    arrays = []
    for i, band in enumerate(bands):
        arr = ds[band].values.astype(np.float32) / 10000.0
        arr = np.clip(arr, 0, 1)
        arr = (arr - MEAN[i]) / STD[i]
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def create_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 224,
    stride: int | None = None,
    min_burn_fraction: float = 0.01,
) -> list[dict]:
    """Slice image and mask into patches, keeping burns + 30% of background."""
    if stride is None:
        stride = patch_size // 2

    _, h, w = image.shape
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            img_patch = image[:, y : y + patch_size, x : x + patch_size]
            mask_patch = mask[y : y + patch_size, x : x + patch_size]

            if img_patch.shape[1:] != (patch_size, patch_size) or mask_patch.shape != (patch_size, patch_size):
                continue
            if np.isnan(img_patch).any() or img_patch.max() == 0:
                continue

            burn_frac = mask_patch.sum() / mask_patch.size
            if burn_frac >= min_burn_fraction or np.random.random() < 0.3:
                patches.append({"image": img_patch, "mask": mask_patch, "burn_fraction": burn_frac})

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
) -> list[dict]:
    """Full preprocessing pipeline for a single region."""
    pre_ds = _restore_crs(xr.open_dataset(pre_path, engine="h5netcdf"))
    post_ds = _restore_crs(xr.open_dataset(post_path, engine="h5netcdf"))
    post_ds = post_ds.rio.reproject_match(pre_ds)

    mask = generate_burn_mask(pre_ds, post_ds)
    image = normalize_bands(post_ds, bands)
    patches = create_patches(image, mask, patch_size=patch_size)

    for p in patches:
        p["region_name"] = region_name

    return patches


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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
