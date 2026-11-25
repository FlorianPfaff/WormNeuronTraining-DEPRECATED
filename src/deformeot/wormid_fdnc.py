"""Optional pretrained fDNC fixed-reference baseline for WormID trajectories.

This adapter deliberately reproduces the released fDNC preprocessing convention:
coordinates are converted to XYZ order, median-centred independently per cloud,
and divided by 84 micrometres (the official ``0.42 * 200`` convention).  No
WormID identity or activity label enters preprocessing or model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .wormid import IdentityBlindSequence
from .wormid_benchmark import TrackResult


@dataclass(frozen=True)
class FDNCTransferConfig:
    """Frozen cross-dataset adapter settings derived from official fDNC code."""

    scale_um: float = 84.0
    batch_size: int = 32
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.scale_um <= 0 or self.batch_size < 1:
            raise ValueError("scale and batch size must be positive")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")


def normalize_fdnc_cloud(points_zyx_um: np.ndarray, *, scale_um: float = 84.0) -> np.ndarray:
    """Apply the label-free published fDNC point-cloud convention."""

    points = np.asarray(points_zyx_um, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("points_zyx_um must be a finite (n, 3) array")
    if scale_um <= 0:
        raise ValueError("scale_um must be positive")
    xyz = points[:, ::-1]
    return (xyz - np.median(xyz, axis=0)) / scale_um


def _load_released_model(model_path: str | Path, source_directory: str | Path, device: str):
    import torch

    source = str(Path(source_directory).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from model import NIT_Registration  # type: ignore[import-not-found]

    use_cuda = device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    model = NIT_Registration(
        input_dim=3,
        n_hidden=128,
        n_layer=6,
        p_rotate=0,
        feat_trans=0,
        cuda=use_cuda,
    )
    checkpoint = torch.load(
        Path(model_path), map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model


def track_fdnc_reference(
    sequence: IdentityBlindSequence,
    *,
    model_path: str | Path,
    source_directory: str | Path,
    config: FDNCTransferConfig = FDNCTransferConfig(),
) -> TrackResult:
    """Infer fixed-reference assignments using the released pretrained network."""

    import torch

    if not sequence.frames:
        raise ValueError("sequence is empty")
    count = len(sequence.frames[0].positions)
    if any(len(frame.positions) != count for frame in sequence.frames):
        raise ValueError("fDNC exhaustive transfer requires a fixed cohort size")
    model = _load_released_model(model_path, source_directory, config.device)
    normalized = [
        normalize_fdnc_cloud(frame.positions, scale_um=config.scale_um)
        for frame in sequence.frames
    ]
    assignments: list[np.ndarray] = [np.arange(count, dtype=np.int64)]
    reference = normalized[0]
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(1, len(normalized), config.batch_size):
            current = normalized[start : start + config.batch_size]
            _, output = model(
                [reference, *current], match_dict=None, ref_idx=0, mode="eval"
            )
            scores = output["p_m"].detach().cpu().numpy()
            for batch_row, points in enumerate(current, start=1):
                matrix = scores[batch_row, : len(points), :count]
                detection_rows, reference_rows = linear_sum_assignment(-matrix)
                assignment = np.full(count, -1, dtype=np.int64)
                assignment[reference_rows] = detection_rows
                assignments.append(assignment)
    cloud_motion = [0.0]
    for previous, current in zip(sequence.frames, sequence.frames[1:]):
        distances = cdist(previous.positions, current.positions)
        cloud_motion.append(
            float(0.5 * (np.median(np.min(distances, axis=1)) + np.median(np.min(distances, axis=0))))
        )
    return TrackResult(
        method="fdnc_reference",
        assignments=tuple(assignments),
        runtime_seconds=float(time.perf_counter() - started),
        cloud_motion_um=np.asarray(cloud_motion, dtype=float),
    )
