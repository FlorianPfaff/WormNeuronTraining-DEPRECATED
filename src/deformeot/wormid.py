"""Read identity-complete calcium trajectories from WormID-Bench NWB files.

The EY recordings used by WormID-Bench store one weighted voxel (the manually
tracked neuron centre) per ROI and time point under
``processing/CalciumActivity/CalciumSeriesSegmentation``.  This loader reads
those centres without touching the much larger image array.  Identity labels
are retained for evaluation and first-frame initialization, but callers can
request identity-blind, independently permuted detection frames for tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import h5py
import numpy as np

from .tracker import DetectionFrame


_TIMEPOINT_RE = re.compile(r"^Seg_tpoint_(\d+)$")


def _decode_label(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


@dataclass(frozen=True)
class WormIDRecording:
    """A complete point-trajectory recording in physical ``(z, y, x)`` units."""

    path: str
    recording_id: str
    positions_zyx_um: np.ndarray
    labels: tuple[str, ...]
    times_seconds: np.ndarray
    spacing_zyx_um: tuple[float, float, float]
    source_xyz_voxels: np.ndarray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_zyx_um, dtype=float)
        source = np.asarray(self.source_xyz_voxels, dtype=float)
        times = np.asarray(self.times_seconds, dtype=float)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions_zyx_um must have shape (frames, neurons, 3)")
        if source.shape != positions.shape:
            raise ValueError("source_xyz_voxels must match positions shape")
        if len(self.labels) != positions.shape[1]:
            raise ValueError("labels must have one entry per neuron")
        if times.shape != (positions.shape[0],):
            raise ValueError("times_seconds must have one entry per frame")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(times)):
            raise ValueError("trajectory arrays must be finite")
        if np.any(np.diff(times) <= 0):
            raise ValueError("times_seconds must be strictly increasing")
        object.__setattr__(self, "positions_zyx_um", positions.copy())
        object.__setattr__(self, "source_xyz_voxels", source.copy())
        object.__setattr__(self, "times_seconds", times.copy())

    @property
    def frames(self) -> int:
        return int(self.positions_zyx_um.shape[0])

    @property
    def neurons(self) -> int:
        return int(self.positions_zyx_um.shape[1])


def _timepoint_groups(group: h5py.Group) -> list[tuple[int, str]]:
    parsed: list[tuple[int, str]] = []
    for key in group.keys():
        match = _TIMEPOINT_RE.match(key)
        if match:
            parsed.append((int(match.group(1)), key))
    parsed.sort()
    if not parsed:
        raise ValueError(f"{group.name} has no Seg_tpoint_<index> groups")
    expected = list(range(parsed[-1][0] + 1))
    actual = [index for index, _ in parsed]
    if actual != expected:
        raise ValueError("calcium segmentation time points are not contiguous from zero")
    return parsed


def load_wormid_recording(
    path: str | Path,
    *,
    segmentation: str = "CalciumSeriesSegmentation",
) -> WormIDRecording:
    """Load one WormID NWB trajectory without loading its image movie.

    The supported benchmark representation has exactly one ``voxel_mask`` row
    per ROI.  A future dense-mask NWB therefore fails loudly instead of silently
    treating every foreground voxel as a different neuron centre.
    """

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    segmentation_path = f"processing/CalciumActivity/{segmentation}"
    label_path = "processing/CalciumActivity/NeuronIDs/labels"
    spacing_path = "general/optophysiology/CalciumImVol/grid_spacing"
    starting_time_path = "acquisition/CalciumImageSeries/starting_time"

    with h5py.File(source_path, "r") as nwb:
        for required in (segmentation_path, label_path, spacing_path, starting_time_path):
            if required not in nwb:
                raise ValueError(f"{source_path} is missing NWB path {required}")

        labels = tuple(_decode_label(value) for value in nwb[label_path][:])
        if not labels:
            raise ValueError("recording contains no neuron labels")
        spacing_xyz = np.asarray(nwb[spacing_path][:], dtype=float)
        if spacing_xyz.shape != (3,) or np.any(~np.isfinite(spacing_xyz)) or np.any(
            spacing_xyz <= 0
        ):
            raise ValueError("CalciumImVol grid_spacing must contain positive xyz values")

        timepoints = _timepoint_groups(nwb[segmentation_path])
        positions_xyz: list[np.ndarray] = []
        expected_ids = np.arange(len(labels), dtype=np.int64)
        for time_index, key in timepoints:
            plane = nwb[f"{segmentation_path}/{key}"]
            for field in ("id", "voxel_mask", "voxel_mask_index"):
                if field not in plane:
                    raise ValueError(f"{plane.name} is missing {field}")
            ids = np.asarray(plane["id"][:], dtype=np.int64)
            if not np.array_equal(ids, expected_ids):
                raise ValueError(
                    f"{plane.name} ROI ids must be the stable sequence 0..{len(labels) - 1}"
                )
            mask = np.asarray(plane["voxel_mask"][:])
            required_fields = {"x", "y", "z", "weight"}
            if mask.dtype.names is None or not required_fields.issubset(mask.dtype.names):
                raise ValueError(f"{plane.name}/voxel_mask lacks x, y, z, weight fields")
            indices = np.asarray(plane["voxel_mask_index"][:], dtype=np.int64)
            if len(mask) != len(labels) or not np.array_equal(
                indices, np.arange(1, len(labels) + 1)
            ):
                raise ValueError(
                    f"{plane.name} is not the one-centre-per-ROI WormID representation"
                )
            if not np.allclose(mask["weight"], 1.0):
                raise ValueError(f"{plane.name} contains non-unit centre weights")
            xyz = np.column_stack([mask["x"], mask["y"], mask["z"]]).astype(float)
            positions_xyz.append(xyz)
            if time_index != len(positions_xyz) - 1:  # defensive against future refactors
                raise AssertionError("time-point ordering changed unexpectedly")

        source_xyz = np.stack(positions_xyz)
        positions_zyx_um = source_xyz[..., ::-1] * spacing_xyz[::-1]
        starting_time = nwb[starting_time_path]
        rate = float(starting_time.attrs.get("rate", np.nan))
        start = float(starting_time[()])
        if not np.isfinite(rate) or rate <= 0 or not np.isfinite(start):
            raise ValueError("CalciumImageSeries starting_time must define a positive rate")
        times = start + np.arange(len(timepoints), dtype=float) / rate

    return WormIDRecording(
        path=str(source_path.resolve()),
        recording_id=source_path.stem,
        positions_zyx_um=positions_zyx_um,
        labels=labels,
        times_seconds=times,
        spacing_zyx_um=tuple(float(value) for value in spacing_xyz[::-1]),
        source_xyz_voxels=source_xyz,
    )


@dataclass(frozen=True)
class IdentityBlindSequence:
    """Tracker inputs paired with hidden truth solely for evaluation."""

    frames: tuple[DetectionFrame, ...]
    truth_indices: tuple[np.ndarray, ...]
    initial_track_ids: tuple[str, ...]
    source_frame_indices: tuple[int, ...]


def identity_blind_sequence(
    recording: WormIDRecording,
    *,
    frame_indices: Sequence[int] | None = None,
    seed: int = 20260720,
) -> IdentityBlindSequence:
    """Permute detections independently while preserving a legal first-frame ID seed.

    Numerical object IDs are fresh anonymous row numbers.  The tracker never sees
    ``truth_indices``; those arrays are returned separately for evaluation.
    """

    if frame_indices is None:
        selected = tuple(range(recording.frames))
    else:
        selected = tuple(int(value) for value in frame_indices)
    if not selected:
        raise ValueError("frame_indices cannot be empty")
    if any(index < 0 or index >= recording.frames for index in selected):
        raise IndexError("frame index outside recording")
    if any(right <= left for left, right in zip(selected, selected[1:])):
        raise ValueError("frame_indices must be strictly increasing")

    rng = np.random.default_rng(seed)
    frames: list[DetectionFrame] = []
    truth: list[np.ndarray] = []
    first_order: np.ndarray | None = None
    for frame_index in selected:
        order = rng.permutation(recording.neurons)
        if first_order is None:
            first_order = order.copy()
        positions = recording.positions_zyx_um[frame_index, order]
        frames.append(
            DetectionFrame(
                positions=positions,
                object_ids=tuple(f"d{row}" for row in range(recording.neurons)),
                time=float(recording.times_seconds[frame_index]),
            )
        )
        truth.append(order.copy())

    assert first_order is not None
    initial_ids = tuple(recording.labels[index] for index in first_order)
    return IdentityBlindSequence(
        frames=tuple(frames),
        truth_indices=tuple(truth),
        initial_track_ids=initial_ids,
        source_frame_indices=selected,
    )


def identity_accuracy(
    sequence: IdentityBlindSequence,
    assignments: Sequence[np.ndarray],
) -> dict[str, float | int]:
    """Score exhaustive MAP assignments against the sequence's hidden identities."""

    if len(assignments) != len(sequence.frames):
        raise ValueError("assignments must have one array per sequence frame")
    # ``initial_track_ids`` are in first-detection order, so track row i has
    # canonical neuron ``truth_indices[0][i]``.
    canonical_truth = sequence.truth_indices[0]
    correct = 0
    misses = 0
    total = 0
    for frame, hidden_order, assignment in zip(
        sequence.frames, sequence.truth_indices, assignments
    ):
        del frame  # the score intentionally uses no coordinate information
        assignment = np.asarray(assignment, dtype=int)
        if assignment.shape != canonical_truth.shape:
            raise ValueError("assignment shape does not match initialized cohort")
        for track_row, detection_row in enumerate(assignment):
            total += 1
            if detection_row < 0:
                misses += 1
            elif detection_row >= len(hidden_order):
                raise IndexError("assignment refers to a nonexistent detection")
            elif int(hidden_order[detection_row]) == int(canonical_truth[track_row]):
                correct += 1
    return {
        "correct": int(correct),
        "total": int(total),
        "misses": int(misses),
        "accuracy": float(correct / total) if total else 1.0,
    }
