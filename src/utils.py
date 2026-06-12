"""Small shared helpers used across training, inference, and the eval scripts.

Kept dependency-light (torch + numpy + xarray only) so any module can import it
without pulling in the model or data-download stack.
"""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr


def get_device() -> torch.device:
    """Best available torch device: CUDA (AWS GPU) > MPS (Apple) > CPU.

    Centralized so the AWS path is never accidentally skipped — several scripts
    previously hard-coded an MPS-or-CPU check that silently ignored CUDA.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def water_mask(
    ds: xr.Dataset,
    threshold: float = 0.0,
    green_band: str = "B03",
    nir_band: str = "B8A",
    swir_band: str = "B11",
) -> np.ndarray:
    """Boolean water mask combining NDWI and MNDWI.

    NDWI = (green - NIR) / (green + NIR) catches inland water.
    MNDWI = (green - SWIR1) / (green + SWIR1) catches open ocean and coastal
    water that NDWI misses when haze depresses NIR toward zero.
    A pixel is masked if either index exceeds threshold.
    """
    green = ds[green_band].values.astype(np.float32)
    nir = ds[nir_band].values.astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-8)

    mndwi = np.zeros_like(ndwi)
    if swir_band in ds:
        swir = ds[swir_band].values.astype(np.float32)
        mndwi = (green - swir) / (green + swir + 1e-8)

    return (ndwi > threshold) | (mndwi > threshold)


def spectral_cloud_mask(
    ds: xr.Dataset,
    blue_threshold: float = 0.10,
    blue_band: str = "B02",
) -> np.ndarray:
    """Boolean cloud mask based on blue-band brightness.

    Clouds and fog have high blue (B02) reflectance (~0.15–0.40) while
    most land surfaces stay below 0.10. Used as a fallback when HLS Fmask
    is unavailable (e.g. cached scenes). Only masks pixels above threshold,
    so bright sandy/urban land is left untouched unless it reads cloud-bright.
    """
    blue = ds[blue_band].values.astype(np.float32)
    return blue > blue_threshold
