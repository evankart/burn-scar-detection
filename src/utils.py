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


def cloud_over_water_mask(
    ds: xr.Dataset,
    blue_band: str = "B02",
    swir_band: str = "B11",
    green_band: str = "B03",
    blue_threshold: float = 0.10,
    mndwi_threshold: float = -0.2,
) -> np.ndarray:
    """Boolean mask for cloud/fog pixels over water.

    Combines a blue-brightness test (clouds are bright) with an MNDWI
    proximity test (MNDWI > -0.2 indicates water or coastal zone).
    Burn scars and urban land have strongly negative MNDWI (≈ -0.5 to -0.8)
    so they are not caught even when blue reflectance is elevated.
    Use as a fallback when HLS Fmask is unavailable.
    """
    blue  = ds[blue_band].values.astype(np.float32)
    green = ds[green_band].values.astype(np.float32)
    swir  = ds[swir_band].values.astype(np.float32) if swir_band in ds else np.zeros_like(blue)
    mndwi = (green - swir) / (green + swir + 1e-8)
    return (blue > blue_threshold) & (mndwi > mndwi_threshold)
