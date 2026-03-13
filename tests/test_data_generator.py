from __future__ import annotations

import numpy as np
import pytest

from simulations.data_generator import _geometry_params_nm_to_um


def test_geometry_params_nm_to_um_returns_native_floats() -> None:
    geometry_nm = np.array([100, 120, 140, 160, 180, 200], dtype=np.float32)

    geometry_um = _geometry_params_nm_to_um(geometry_nm)

    assert geometry_um == pytest.approx((0.1, 0.12, 0.14, 0.16, 0.18, 0.2))
    assert all(isinstance(value, float) for value in geometry_um)


def test_geometry_params_nm_to_um_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="Expected geometry_params shape"):
        _geometry_params_nm_to_um(np.array([[100, 120, 140, 160, 180, 200]], dtype=np.float32))
