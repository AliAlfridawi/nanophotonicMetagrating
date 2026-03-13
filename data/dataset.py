from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.contracts import NUM_VARIABLES, SPECTRUM_DIM, scale_geometry_nm_to_unit


class MetagratingDataset(Dataset):
    def __init__(
        self,
        x_path: str = "data/raw/X_inputs.npy",
        y_path: str = "data/raw/Y_outputs.npy",
    ) -> None:
        x_file = Path(x_path)
        y_file = Path(y_path)

        if not x_file.exists():
            raise FileNotFoundError(f"Missing input geometry file: {x_file}")
        if not y_file.exists():
            raise FileNotFoundError(f"Missing output spectra file: {y_file}")

        x_data = np.load(x_file).astype(np.float32)
        y_data = np.load(y_file).astype(np.float32)

        if x_data.ndim != 2 or x_data.shape[1] != NUM_VARIABLES:
            raise ValueError(f"Expected X shape (N, {NUM_VARIABLES}), got {x_data.shape}")
        if y_data.ndim != 2 or y_data.shape[1] != SPECTRUM_DIM:
            raise ValueError(f"Expected Y shape (N, {SPECTRUM_DIM}), got {y_data.shape}")
        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"X and Y sample count mismatch: {x_data.shape[0]} vs {y_data.shape[0]}"
            )

        x_scaled = scale_geometry_nm_to_unit(x_data).astype(np.float32)

        self._x = torch.from_numpy(x_scaled)
        self._y = torch.from_numpy(y_data)

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._x[index], self._y[index]

