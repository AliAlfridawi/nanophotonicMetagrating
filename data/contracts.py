from __future__ import annotations

import numpy as np
import torch

MIN_FEATURE_SIZE_NM = 50.0
MAX_FEATURE_SIZE_NM = 300.0
MIN_PERIOD_NM = 800.0
MAX_PERIOD_NM = 1200.0

NUM_PILLARS = 3
NUM_VARIABLES = NUM_PILLARS * 2

NUM_WAVELENGTHS = 11
WAVELENGTHS_NM = np.linspace(1500.0, 1600.0, NUM_WAVELENGTHS, dtype=np.float32)
SPECTRUM_DIM = NUM_WAVELENGTHS * 2

_FEATURE_RANGE_NM = MAX_FEATURE_SIZE_NM - MIN_FEATURE_SIZE_NM


def scale_geometry_nm_to_unit(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return (x - MIN_FEATURE_SIZE_NM) / _FEATURE_RANGE_NM


def unscale_geometry_unit_to_nm(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return x * _FEATURE_RANGE_NM + MIN_FEATURE_SIZE_NM


def is_valid_geometry_nm(geometry_nm: np.ndarray) -> bool:
    if geometry_nm.shape[-1] != NUM_VARIABLES:
        return False
    if np.any(geometry_nm < MIN_FEATURE_SIZE_NM) or np.any(geometry_nm > MAX_FEATURE_SIZE_NM):
        return False
    period_nm = float(np.sum(geometry_nm))
    return MIN_PERIOD_NM <= period_nm <= MAX_PERIOD_NM


def build_bandpass_target_spectrum(target_wavelength_nm: float) -> np.ndarray:
    target = np.zeros(SPECTRUM_DIM, dtype=np.float32)
    closest_idx = int(np.argmin(np.abs(WAVELENGTHS_NM - target_wavelength_nm)))
    target[closest_idx] = 1.0
    return target

