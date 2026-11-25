"""Lazy, identity-blind stronger comparators for the bounded paper pilot.

CPD is a registration baseline, not a newly trained model. The fDNC checks verify
adapter fidelity and batching, not the biological validity of its predictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys
import time
from types import FunctionType, SimpleNamespace
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .wormid import IdentityBlindSequence
from .wormid_benchmark import ReferenceTrackerConfig, TrackResult, _affine, _validate_positions


@dataclass(frozen=True)
class CPDConfig:
    """Nonrigid pycpd settings in target nearest-neighbor distance units."""

    alpha: float = 2.0
    beta: float = 2.0
    w: float = 0.0
    max_iterations: int = 100
    tolerance: float = 1e-5

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("CPD alpha must be finite and positive")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("CPD beta must be finite and positive")
        if self.w != 0.0:
            raise ValueError("the exhaustive-cohort pilot freezes CPD w=0")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("CPD max_iterations must be a positive integer")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("CPD tolerance must be finite and positive")


def normalize_cpd_clouds(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Center both clouds on target median and scale by its positive NN median.

Duplicate target positions do not create a zero scale: each row contributes its
nearest strictly positive distance. A completely coincident target is rejected.
"""

    source = _validate_positions(source)
    target = _validate_positions(target)
    if len(source) != len(target):
        raise ValueError("CPD pilot requires equal source and target cardinality")
    distances = cdist(target, target)
    distances[distances <= 0] = np.inf
    nearest_positive = np.min(distances, axis=1)
    nearest_positive = nearest_positive[np.isfinite(nearest_positive)]
    if not len(nearest_positive):
        raise ValueError("CPD cannot normalize a completely coincident target cloud")
    scale = float(np.median(nearest_positive))
    center = np.median(target, axis=0)
    return (
        (source - center) / scale,
        (target - center) / scale,
        {"center_um": center.tolist(), "scale_um": scale},
    )


def _sequence_positions(sequence: IdentityBlindSequence) -> list[np.ndarray]:
    if not sequence.frames:
        raise ValueError("sequence is empty")
    positions = [_validate_positions(frame.positions) for frame in sequence.frames]
    if any(len(points) != len(positions[0]) for points in positions):
        raise ValueError("the pilot baselines require a fixed cohort size")
    return positions


def _cloud_motion(positions: list[np.ndarray]) -> np.ndarray:
    motion = [0.0]
    for previous, current in zip(positions, positions[1:]):
        distance = cdist(previous, current)
        motion.append(float(0.5 * (np.median(distance.min(0)) + np.median(distance.min(1)))))
    return np.asarray(motion, dtype=float)


def track_cpd(
    sequence: IdentityBlindSequence,
    *,
    anchor: str = "reference",
    config: CPDConfig = CPDConfig(),
    geometry: ReferenceTrackerConfig = ReferenceTrackerConfig(),
) -> TrackResult:
    """Affine prealignment, nonrigid CPD, then exhaustive Hungarian association."""

    if anchor not in {"reference", "temporal"}:
        raise ValueError("CPD anchor must be 'reference' or 'temporal'")
    positions = _sequence_positions(sequence)
    from pycpd import DeformableRegistration

    started = time.perf_counter()
    reference = positions[0]
    previous = reference.copy()
    count = len(reference)
    assignments = [np.arange(count, dtype=np.int64)]
    for current in positions[1:]:
        source = reference if anchor == "reference" else previous
        aligned = _affine(source, current, geometry).apply(source)
        source_norm, target_norm, _ = normalize_cpd_clouds(aligned, current)
        registration = DeformableRegistration(X=target_norm, Y=source_norm, **asdict(config))
        transformed, _ = registration.register()
        transformed = np.asarray(transformed, dtype=float)
        if transformed.shape != source_norm.shape or not np.all(np.isfinite(transformed)):
            raise ValueError("CPD returned nonfinite coordinates or changed source cardinality")
        track_rows, detection_rows = linear_sum_assignment(cdist(transformed, target_norm))
        assignment = np.full(count, -1, dtype=np.int64)
        assignment[track_rows] = detection_rows
        assignments.append(assignment)
        previous = current[assignment]
    motion = _cloud_motion(positions)
    return TrackResult(
        method=f"cpd_{anchor}",
        assignments=tuple(assignments),
        runtime_seconds=float(time.perf_counter() - started),
        cloud_motion_um=motion,
    )


def fdnc_assignment_from_scores(scores: np.ndarray) -> np.ndarray:
    """Invert released detection-row/reference-column scores into TrackResult rows."""

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or len(scores) < 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("fDNC exhaustive score matrix must be square with at least two rows")
    if not np.all(np.isfinite(scores)):
        raise ValueError("fDNC score matrix must be finite")
    detection_rows, reference_rows = linear_sum_assignment(-scores)
    assignment = np.full(len(scores), -1, dtype=np.int64)
    assignment[reference_rows] = detection_rows
    return assignment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fdnc_scores(model: Any, clouds: list[np.ndarray]) -> list[np.ndarray]:
    import torch

    with torch.no_grad():
        _, output = model(clouds, match_dict=None, ref_idx=0, mode="eval")
    scores = np.asarray(output["p_m"].detach().cpu().numpy(), dtype=float)
    count = len(clouds[0])
    if scores.ndim != 3 or scores.shape[0] != len(clouds):
        raise ValueError("released fDNC returned an unexpected batch score shape")
    if scores.shape[1] < count or scores.shape[2] != count + 1:
        raise ValueError("released fDNC must expose reference columns plus one outlier column")
    matrices = [scores[index, :len(cloud), :count] for index, cloud in enumerate(clouds)]
    if not all(np.all(np.isfinite(matrix)) for matrix in matrices):
        raise ValueError("released fDNC returned nonfinite in-cohort scores")
    return matrices


def _official_fdnc_assignments(
    model_path: Path, source_directory: Path, normalized: list[np.ndarray], device: str
) -> list[np.ndarray]:
    """Execute released predict_label without editing its hard-coded model path.

The function's private globals replace only torch.load with a checkpoint-path
redirect. Architecture construction, inference, and Hungarian assignment remain
the released implementation. No temporary files or process cwd changes occur.
"""

    import torch

    script = source_directory / "DNC_predict.py"
    specification = importlib.util.spec_from_file_location("_paper_pilot_official_fdnc", script)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot import released fDNC inference at {script}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    class CheckpointTorchProxy:
        def __getattr__(self, name: str):
            return getattr(torch, name)

        def load(self, _released_hardcoded_path, *args, **kwargs):
            kwargs["weights_only"] = False
            return torch.load(model_path, *args, **kwargs)

    published = module.predict_label
    direct = FunctionType(
        published.__code__, {**published.__globals__, "torch": CheckpointTorchProxy()},
        name=published.__name__, argdefs=published.__defaults__, closure=published.__closure__,
    )
    count = len(normalized[0])
    assignments = []
    for current in normalized[1:]:
        labels, _ = direct(normalized[0], np.arange(count), current, cuda=device == "cuda")
        predicted_reference = np.asarray([label for label, _ in labels], dtype=np.int64)
        if not np.array_equal(np.sort(predicted_reference), np.arange(count)):
            raise ValueError("official fDNC prediction did not produce a complete permutation")
        inverse = np.empty(count, dtype=np.int64)
        inverse[predicted_reference] = np.arange(count)
        assignments.append(inverse)
    return assignments


def validate_fdnc(
    model_path: str | Path,
    source_directory: str | Path,
    sequence: IdentityBlindSequence,
    device: str = "cpu",
) -> dict[str, Any]:
    """Validate at most four nonreference clouds against the actual released code.

No hidden identities or activity are read. Numerical score agreement permits
float32 roundoff, but assignments must agree exactly. The resulting status is
an adapter-integrity check, not a claim of tracking accuracy or training purity.
"""

    from . import wormid_fdnc

    if device not in {"cpu", "cuda"}:
        raise ValueError("fDNC validation device must be cpu or cuda")
    positions = _sequence_positions(sequence)[:5]
    if len(positions) < 3:
        raise ValueError("fDNC batching validation requires at least three frames")
    model_path = Path(model_path).resolve()
    source_directory = Path(source_directory).resolve()
    hashes = {"checkpoint": _sha256(model_path)}
    for name in ("model.py", "DNC_predict.py", "Registration_transformer.py"):
        hashes[name] = _sha256(source_directory / name)
    model = wormid_fdnc._load_released_model(model_path, source_directory, device)
    loaded_module = sys.modules.get("model")
    if loaded_module is not None and Path(loaded_module.__file__).resolve() != source_directory / "model.py":
        raise RuntimeError("a different generic 'model' module was imported; use a clean subprocess")

    # Reproduce official pre_matt arithmetic, independently of the adapter.
    normalized = []
    for points in positions:
        xyz = points[:, ::-1] / 0.42
        xyz = (xyz - np.median(xyz, axis=0)) / 200.0
        normalized.append(xyz)
    adapter_normalized = [wormid_fdnc.normalize_fdnc_cloud(points) for points in positions]
    preprocessing_ok = all(np.allclose(left, right, rtol=1e-13, atol=1e-13)
                           for left, right in zip(normalized, adapter_normalized))
    batched = _fdnc_scores(model, normalized)[1:]
    singles = [_fdnc_scores(model, [normalized[0], current])[1] for current in normalized[1:]]
    batch_assignments = [fdnc_assignment_from_scores(matrix) for matrix in batched]
    single_assignments = [fdnc_assignment_from_scores(matrix) for matrix in singles]
    official_assignments = _official_fdnc_assignments(model_path, source_directory, normalized, device)
    # Only anonymous frame coordinates are passed through to the adapter.
    limited_sequence = SimpleNamespace(frames=sequence.frames[:len(positions)])
    adapter = wormid_fdnc.track_fdnc_reference(
        limited_sequence, model_path=model_path, source_directory=source_directory,
        config=wormid_fdnc.FDNCTransferConfig(device=device, batch_size=len(positions) - 1),
    )
    adapter_single = wormid_fdnc.track_fdnc_reference(
        limited_sequence, model_path=model_path, source_directory=source_directory,
        config=wormid_fdnc.FDNCTransferConfig(device=device, batch_size=1),
    )
    checks = {
        "preprocessing_matches_official_arithmetic": bool(preprocessing_ok),
        "single_vs_batch_scores_close": all(np.allclose(left, right, rtol=1e-5, atol=1e-5)
                                             for left, right in zip(singles, batched)),
        "single_vs_batch_assignments_identical": all(np.array_equal(left, right)
                                                     for left, right in zip(single_assignments, batch_assignments)),
        "published_predict_label_matches_score_inversion": all(np.array_equal(left, right)
                                                                 for left, right in zip(official_assignments, single_assignments)),
        "adapter_matches_published_assignments": all(np.array_equal(left, right)
                                                      for left, right in zip(adapter.assignments[1:], official_assignments)),
        "adapter_single_and_batch_identical": all(np.array_equal(left, right)
                                                   for left, right in zip(adapter.assignments, adapter_single.assignments)),
        "adapter_assignments_are_permutations": all(np.array_equal(np.sort(row), np.arange(len(positions[0])))
                                                     for row in adapter.assignments),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "device": device,
        "frames_checked": len(positions),
        "neurons": len(positions[0]),
        "hashes": hashes,
        "maximum_single_batch_score_difference": max(float(np.max(np.abs(left - right)))
                                                       for left, right in zip(singles, batched)),
        "score_tolerance": {"rtol": 1e-5, "atol": 1e-5},
        "assignment_tolerance": "exact permutation equality",
        "official_execution": "Released DNC_predict.predict_label with only hardcoded checkpoint load redirected",
        "limitations": [
            "Fixed 84-micrometer scale; no per-animal tuning or retraining.",
            "No available anatomical side flag is applied; no fluorescence/color input is used.",
            "Outlier scores are discarded for the prespecified exhaustive cohort, as in official Hungarian inference.",
            "This validates inference fidelity, not accuracy or absence of training-data overlap.",
        ],
    }
