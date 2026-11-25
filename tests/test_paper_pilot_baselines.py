from contextlib import nullcontext
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from deformeot.paper_pilot_baselines import (
    CPDConfig, fdnc_assignment_from_scores, normalize_cpd_clouds, track_cpd, validate_fdnc,
)
from deformeot.tracker import DetectionFrame


class BlindOnly:
    def __init__(self, clouds):
        self.frames = tuple(DetectionFrame(np.asarray(points, dtype=float), time=i)
                            for i, points in enumerate(clouds))

    @property
    def truth_indices(self):
        raise AssertionError("a baseline attempted to read hidden identities")

    @property
    def initial_track_ids(self):
        raise AssertionError("a baseline attempted to read semantic identities")


def test_cpd_normalizes_both_clouds_using_target_coordinates():
    target = np.asarray([[1, 2, 3], [3, 2, 3], [7, 2, 3]], dtype=float)
    source = target + [2, 4, 6]
    left, right, metadata = normalize_cpd_clouds(source, target)
    assert metadata == {"center_um": [3.0, 2.0, 3.0], "scale_um": 2.0}
    assert np.allclose(np.median(right, axis=0), 0)
    assert np.allclose(left - right, [1, 2, 3])


def test_cpd_duplicate_points_use_nearest_strictly_positive_distance():
    target = np.asarray([[0, 0, 0], [0, 0, 0], [2, 0, 0]])
    _, _, metadata = normalize_cpd_clouds(target, target)
    assert metadata["scale_um"] == 2.0
    with pytest.raises(ValueError, match="coincident"):
        normalize_cpd_clouds(np.zeros((3, 3)), np.zeros((3, 3)))


@pytest.mark.parametrize("kwargs", [
    {"alpha": 0}, {"alpha": np.nan}, {"beta": -1}, {"beta": np.inf},
    {"w": .1}, {"max_iterations": 0}, {"max_iterations": 1.5}, {"tolerance": np.nan},
])
def test_cpd_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        CPDConfig(**kwargs)


@pytest.mark.parametrize("anchor", ["reference", "temporal"])
@pytest.mark.parametrize("nonrigid", [False, True])
def test_cpd_native_recovers_permutation_translation_and_mild_deformation(anchor, nonrigid):
    pytest.importorskip("pycpd")
    rng = np.random.default_rng(211)
    points = rng.normal(size=(18, 3)) * [6, 4, 3]
    orders = [rng.permutation(len(points)) for _ in range(3)]
    clouds = []
    for index, order in enumerate(orders):
        deformed = points + index * np.asarray([.2, -.15, .1])
        if nonrigid:
            deformed = deformed + .02 * index * np.sin(points)
        clouds.append(deformed[order])
    result = track_cpd(BlindOnly(clouds), anchor=anchor)
    assert result.method == f"cpd_{anchor}"
    for order, assignment in zip(orders, result.assignments):
        assert np.array_equal(order[assignment], orders[0])


def test_cpd_wires_target_source_and_configuration(monkeypatch):
    captured = []

    class Registration:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.points = kwargs["Y"]

        def register(self):
            return self.points, None

    monkeypatch.setitem(sys.modules, "pycpd", SimpleNamespace(DeformableRegistration=Registration))
    points = np.asarray([[0, 0, 0], [3, 1, 1], [1, 4, 0], [2, 0, 6]], dtype=float)
    sequence = BlindOnly([points, points[[2, 0, 3, 1]] + [1, 2, 3]])
    result = track_cpd(sequence, config=CPDConfig(alpha=5, beta=4))
    assert np.array_equal(result.assignments[1], [1, 3, 0, 2])
    assert captured[0]["alpha"] == 5 and captured[0]["beta"] == 4
    assert captured[0]["w"] == 0 and captured[0]["max_iterations"] == 100
    assert captured[0]["tolerance"] == 1e-5


@pytest.mark.parametrize("clouds,error", [
    ([], "empty"), ([np.ones((3, 3)), np.ones((2, 3))], "fixed cohort"),
    ([np.full((3, 3), np.nan)], "finite"),
])
def test_cpd_rejects_invalid_sequences(clouds, error):
    # DetectionFrame already rejects nonfinite values; both boundaries are valid.
    with pytest.raises(ValueError, match=error):
        track_cpd(BlindOnly(clouds))


def test_cpd_rejects_nonfinite_registration(monkeypatch):
    class Registration:
        def __init__(self, **kwargs):
            self.points = kwargs["Y"]

        def register(self):
            return np.full_like(self.points, np.nan), None

    monkeypatch.setitem(sys.modules, "pycpd", SimpleNamespace(DeformableRegistration=Registration))
    points = np.asarray([[0, 0, 0], [3, 1, 1], [1, 4, 0], [2, 0, 6]], dtype=float)
    with pytest.raises(ValueError, match="nonfinite"):
        track_cpd(BlindOnly([points, points]))


def test_cpd_rejects_unknown_anchor():
    with pytest.raises(ValueError, match="anchor"):
        track_cpd(BlindOnly([]), anchor="future")


def test_fdnc_score_direction_is_reference_to_detection_not_the_reverse():
    scores = np.asarray([[0, 0, 10], [10, 0, 0], [0, 10, 0]], dtype=float)
    assert np.array_equal(fdnc_assignment_from_scores(scores), [1, 2, 0])


@pytest.mark.parametrize("scores", [np.ones((2, 3)), np.ones((1, 1)), np.full((2, 2), np.nan)])
def test_fdnc_score_validation(scores):
    with pytest.raises(ValueError):
        fdnc_assignment_from_scores(scores)


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeFDNC:
    def __call__(self, clouds, *, match_dict, ref_idx, mode):
        assert match_dict is None and ref_idx == 0 and mode == "eval"
        count = len(clouds[0])
        scores = np.full((len(clouds), count, count + 1), -100., dtype=float)
        for index, cloud in enumerate(clouds):
            scores[index, :, :count] = -np.sum((cloud[:, None] - clouds[0][None, :]) ** 2, axis=2)
        return None, {"p_m": FakeTensor(scores)}


def test_fdnc_validation_uses_only_coordinates_and_checks_batching(monkeypatch, tmp_path):
    import deformeot.paper_pilot_baselines as pilot
    import deformeot.wormid_fdnc as adapter

    source = tmp_path / "source"
    source.mkdir()
    for name in ("model.py", "DNC_predict.py", "Registration_transformer.py"):
        (source / name).write_text("# test source\n")
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.delitem(sys.modules, "model", raising=False)
    monkeypatch.setattr(adapter, "_load_released_model", lambda *args: FakeFDNC())
    monkeypatch.setattr(pilot, "_official_fdnc_assignments", lambda path, directory, clouds, device: [
        fdnc_assignment_from_scores(-np.sum((cloud[:, None] - clouds[0][None, :]) ** 2, axis=2))
        for cloud in clouds[1:]
    ])
    points = np.asarray([[0, 0, 0], [10, 20, 0], [20, 0, 10]], dtype=float)
    sequence = BlindOnly([points, points[[2, 0, 1]], points[[1, 2, 0]]])
    result = validate_fdnc(checkpoint, source, sequence)
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert len(result["hashes"]["checkpoint"]) == 64
    assert result["maximum_single_batch_score_difference"] == 0


def test_fdnc_validation_flags_batch_dependent_assignments(monkeypatch, tmp_path):
    import deformeot.paper_pilot_baselines as pilot
    import deformeot.wormid_fdnc as adapter

    source = tmp_path / "source"
    source.mkdir()
    for name in ("model.py", "DNC_predict.py", "Registration_transformer.py"):
        (source / name).write_text("# test source\n")
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"test checkpoint")

    class BatchDependent(FakeFDNC):
        def __call__(self, clouds, **kwargs):
            _, output = super().__call__(clouds, **kwargs)
            if len(clouds) > 2:
                output["p_m"].value[:, :, :len(clouds[0])] = np.roll(
                    output["p_m"].value[:, :, :len(clouds[0])], 1, axis=2)
            return None, output

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.delitem(sys.modules, "model", raising=False)
    monkeypatch.setattr(adapter, "_load_released_model", lambda *args: BatchDependent())
    monkeypatch.setattr(pilot, "_official_fdnc_assignments", lambda path, directory, clouds, device: [
        fdnc_assignment_from_scores(-np.sum((cloud[:, None] - clouds[0][None, :]) ** 2, axis=2))
        for cloud in clouds[1:]
    ])
    points = np.asarray([[0, 0, 0], [10, 20, 0], [20, 0, 10]], dtype=float)
    result = validate_fdnc(checkpoint, source, BlindOnly([points, points, points]))
    assert result["status"] == "failed"
    assert not result["checks"]["single_vs_batch_assignments_identical"]
    assert not result["checks"]["adapter_single_and_batch_identical"]
