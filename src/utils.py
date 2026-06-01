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
) -> np.ndarray:
    """Boolean NDWI water mask, NDWI = (green - NIR) / (green + NIR).

    Open water is not burnable, yet both the post-only model and the dNBR label
    produce spurious "burned" pixels over it (where NIR ≈ SWIR ≈ 0, NBR is pure
    noise). Burn scars sit at NDWI ≤ 0 (NIR ≥ green), so a 0 cutoff removes
    ocean/lakes without eating real burns. Deterministic and never tuned on the
    test fires.
    """
    green = ds[green_band].values.astype(np.float32)
    nir = ds[nir_band].values.astype(np.float32)
    ndwi = (green - nir) / (green + nir + 1e-8)
    return ndwi > threshold
