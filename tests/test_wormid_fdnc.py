from __future__ import annotations

import numpy as np

from deformeot.wormid_fdnc import FDNCTransferConfig, normalize_fdnc_cloud


def test_fdnc_normalization_is_xyz_median_centered_and_uses_official_scale():
    zyx = np.asarray([[84.0, 42.0, 0.0], [0.0, 42.0, 84.0], [42.0, 0.0, 42.0]])
    normalized = normalize_fdnc_cloud(zyx)
    assert np.allclose(np.median(normalized, axis=0), 0.0)
    # XYZ conversion makes the first output axis the reverse of input Z.
    assert np.allclose(normalized[:, 0], [-0.5, 0.5, 0.0])


def test_fdnc_transfer_config_rejects_uncontrolled_adapter_values():
    for kwargs in ({"scale_um": 0}, {"batch_size": 0}, {"device": "automatic"}):
        try:
            FDNCTransferConfig(**kwargs)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid configuration was accepted: {kwargs}")
