"""Offline fixtures only; run these on the calculation server, never fetch data."""

import io
import json
from pathlib import Path
import re
import zipfile

import h5py
import numpy as np
import pytest
import yaml

from deformeot.paper_phase2_evidence import (
    HARD_LIMITS,
    EvidenceLimitError,
    PublicReader,
    RangeFile,
    ReadBudget,
    _csv_summary,
    _validate_config,
    _zip_member,
    classify_asset,
    inspect_nwb,
    summarize_zephir,
)


class Response(io.BytesIO):
    def __init__(self, payload, *, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload))}
        self.read_bytes = 0

    def geturl(self):
        return "https://public.example/fixture"

    def read(self, length=-1):
        data = super().read(length)
        self.read_bytes += len(data)
        return data


def range_opener(payload, calls):
    def open_request(request, timeout):
        start, stop = map(int, re.fullmatch(r"bytes=(\d+)-(\d+)",
                                           request.get_header("Range")).groups())
        calls.append((start, stop))
        part = payload[start:stop + 1]
        return Response(part, status=206, headers={"Content-Length": str(len(part)),
            "Content-Range": f"bytes {start}-{stop}/{len(payload)}"})
    return open_request


def test_config_has_exact_order_and_bounded_primary_literature():
    root = Path(__file__).resolve().parents[1]
    config = _validate_config(yaml.safe_load((root / "configs/paper_phase2_evidence.yaml").read_text()))
    assert len(config["related_work"]["papers"]) == 10
    assert [x["family"] for x in config["candidates"]] == ["SK1"] * 4 + ["SK2"] * 2 + ["KK"] * 9
    assert config["reference_identity"]["independent_framewise_manual_ground_truth_confirmed"] is False


@pytest.mark.parametrize("key", HARD_LIMITS)
def test_cannot_expand_hard_limits(key):
    with pytest.raises(ValueError):
        ReadBudget({key: HARD_LIMITS[key] + 1})


def test_budget_bytes_and_deadline():
    now = [0.0]
    budget = ReadBudget({"max_total_bytes": 30, "max_asset_bytes": 20,
                         "max_seconds": 2}, clock=lambda: now[0])
    budget.check("a", 20)
    budget.charge("a", 20)
    with pytest.raises(EvidenceLimitError):
        budget.check("a", 1)
    with pytest.raises(EvidenceLimitError):
        budget.check("b", 11)
    now[0] = 2.0
    with pytest.raises(EvidenceLimitError):
        budget.check("b")


def test_range_200_fallback_is_rejected_without_reading_body():
    response = Response(b"raw movie must not be read")
    budget = ReadBudget({})
    reader = PublicReader(budget, lambda request, timeout: response)
    with pytest.raises(ValueError, match="full-body fallback"):
        reader.range("https://public.example/movie.nwb", "test", 0, 10)
    assert response.read_bytes == 0
    assert budget.total_bytes == 0


def test_wrong_content_range_is_rejected_without_reading_body():
    response = Response(b"123", status=206, headers={"Content-Range": "bytes 4-6/100"})
    reader = PublicReader(ReadBudget({}), lambda request, timeout: response)
    with pytest.raises(ValueError, match="unexpected Content-Range"):
        reader.range("https://public.example/test", "test", 1, 3)
    assert response.read_bytes == 0


def test_repeated_ranges_charge_actual_bytes_and_cache_reuses():
    payload = bytes(range(100))
    calls = []
    budget = ReadBudget({"range_block_bytes": 16})
    reader = PublicReader(budget, range_opener(payload, calls))
    with RangeFile(reader, "https://public.example/file", len(payload), "test") as stream:
        stream.seek(10)
        assert stream.read(10) == payload[10:20]
        stream.seek(10)
        assert stream.read(10) == payload[10:20]
        assert len(calls) == 2
    assert budget.total_bytes == 32
    reader.range("https://public.example/file", "test", 0, 16)
    assert budget.total_bytes == 48


def test_full_movie_read_to_eof_is_refused_before_network():
    calls = []
    reader = PublicReader(ReadBudget({}), range_opener(b"x", calls))
    with RangeFile(reader, "https://public.example/movie", 100 * 1024 * 1024, "test") as stream:
        with pytest.raises(EvidenceLimitError):
            stream.read()
    assert calls == []


def test_zip_tail_and_compact_member_without_image_download():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("metadata.json", b'{"shape_t": 2}')
        archive.writestr("image.dat", b"x" * 200000)
    payload = output.getvalue()
    calls = []
    reader = PublicReader(ReadBudget({"range_block_bytes": 128}), range_opener(payload, calls))
    with RangeFile(reader, "https://public.example/data.zip", len(payload), "test") as stream:
        with zipfile.ZipFile(stream) as archive:
            assert len(archive.infolist()) == 2
            assert json.loads(_zip_member(archive, "metadata.json", 100)) == {"shape_t": 2}
            with pytest.raises(EvidenceLimitError):
                _zip_member(archive, "image.dat", 100)
    assert reader.budget.total_bytes < len(payload)


def nwb_fixture(*, dense=False, missing_frame=False, duplicate_id=False):
    output = io.BytesIO()
    with h5py.File(output, "w") as handle:
        handle["general/subject/subject_id"] = "fixture-worm"
        series = handle.create_group("acquisition/CalciumImageSeries")
        # Image dataset exists only to expose its shape; inspector must never read it.
        series.create_dataset("data", shape=(3, 4, 5, 6), dtype="uint8")
        starting = series.create_dataset("starting_time", data=0.0)
        starting.attrs["rate"] = 2.0
        starting.attrs["unit"] = "seconds"
        volume = handle.create_group("general/optophysiology/CalciumImVol")
        spacing = volume.create_dataset("grid_spacing", data=[1.0, 1.0, 2.0])
        spacing.attrs["unit"] = "micrometers"
        series["imaging_volume"] = volume
        seg = handle.create_group("processing/CalciumActivity/CalciumSeriesSegmentation")
        if dense:
            seg.create_dataset("data", shape=(3, 4, 5, 6), dtype="uint16")
        else:
            for t in range(2 if missing_frame else 3):
                frame = seg.create_group(f"Seg_tpoint_{t}")
                frame["id"] = [11, 11] if duplicate_id else [11, 42]
                frame["voxel_mask"] = np.array([[1, 2, 3, 1], [2, 3, 4, 1]])
                frame["voxel_mask_index"] = [1, 2]
                frame["ID_labels"] = np.array([b"AVAL", b"AVAR"])
    return output.getvalue()


def test_nwb_actual_roi_ids_not_semantic_labels_or_manual_truth():
    with h5py.File(io.BytesIO(nwb_fixture()), "r") as handle:
        observed = inspect_nwb(handle)
    roi = observed["temporal_roi"]
    assert roi["first_frame_ids"] == [11, 42]
    assert roi["complete"] is True
    assert roi["all_ordered_ids_stable"] is True
    assert roi["independent_manual_truth_verified"] is False
    assert roi["identity_semantics_documented"] is False
    assert observed["calibration"]["verified"] is True
    assert observed["time"]["verified"] is True
    status, reasons = classify_asset({"observation": observed, "license_verified": True})
    assert status == "conditional"
    assert "asset_or_subset_expert_verification_and_release_mapping_not_verified" in reasons


@pytest.mark.parametrize("option", ["missing_frame", "duplicate_id"])
def test_incomplete_or_duplicate_roi_references_not_complete(option):
    with h5py.File(io.BytesIO(nwb_fixture(**{option: True})), "r") as handle:
        observed = inspect_nwb(handle)
    assert observed["temporal_roi"]["complete"] is False


def test_dense_segmentation_is_not_read_or_interpolated():
    with h5py.File(io.BytesIO(nwb_fixture(dense=True)), "r") as handle:
        observed = inspect_nwb(handle)
    assert observed["temporal_roi"]["schema"] == "dense_segmentation_or_unknown_data"
    assert observed["temporal_roi"]["complete"] is False


def hdf_bytes(columns):
    output = io.BytesIO()
    with h5py.File(output, "w") as handle:
        for key, value in columns.items():
            handle[key] = value
    return output.getvalue()


def test_zephir_zero_is_reported_not_silently_removed_or_called_manual():
    annotations = hdf_bytes({"t_idx": [0, 0, 1, 1], "worldline_id": [0, 7, 0, 7],
        "provenance": np.array([b"NEIR", b"JY1", b"NEIR", b"JY1"]),
        "x": [.1, .2, .3, .4], "y": [.2] * 4, "z": [.3] * 4})
    worldlines = hdf_bytes({"id": [0, 7, 8], "name": np.array([b"null", b"neuron", b"unused"])})
    observed = summarize_zephir(annotations, worldlines, {"shape_t": 2})
    roi = observed["temporal_roi"]
    assert roi["complete"] is True
    assert roi["worldline_zero"]["name"] == "null"
    assert roi["worldline_zero"]["rows"] == 2
    assert roi["worldline_count_including_zero"] == 2
    assert roi["named_worldlines_without_annotations"] == [8]
    assert roi["independent_manual_truth_verified"] is False
    assert roi["provenance_counts"] == {"JY1": 2, "NEIR": 2}


def test_exposed_alias_excluded_even_when_metadata_fails():
    assert classify_asset({"prior_exposure": "known_exposed_alias", "error": "timeout"}) == (
        "excluded", ["known_previously_exposed_recording_alias"])


def test_csv_does_not_infer_temporal_identity_from_wide_coordinates():
    result = _csv_summary("x,y,z\n1,2,3\n4,5,6\n")
    assert result["explicit_frame_and_id_columns"] is False
    result = _csv_summary("frame,track_id,x\n0,7,1\n1,7,2\n")
    assert result["distinct_frame_values"] == 2
    assert result["distinct_id_values"] == 1
    assert result["independent_manual_truth_verified"] is False


def verified_sparse_record():
    return {"observation": {"temporal_roi": {"complete": False,
        "frames": [{"frame": 1, "roi_ids": [11, 42], "valid": True},
                   {"frame": 3, "roi_ids": [11, 42], "valid": True}]},
        "time": {"verified": True}, "calibration": {"verified": True}},
        "license_verified": True, "prior_exposure": "not_found_in_project_ledger",
        "alias_review": {"status": "reviewed_no_known_exposure",
                         "project_evidence_paths": ["fixture-exposure-ledger"]},
        "reference_verification": {"status": "documented_expert_verified",
            "production": "algorithm_assisted", "scope": "sparse_documented_subset",
            "release_mapping_verified": True, "roi_ids": [42], "frame_indices": [1, 3],
            "temporal_identity_semantics_documented": True,
            "source_url": "https://public.example/exact-release-verification",
            "retrieved_on": "2026-10-04"}}


def test_expert_verified_algorithm_assisted_sparse_scope_can_be_eligible():
    record = verified_sparse_record()
    assert classify_asset(record) == ("eligible", [])
    assert record["validation_scope"] == "sparse_documented_subset"
    assert record["observation"]["temporal_roi"]["complete"] is False


def test_sparse_paper_claim_without_release_mapping_does_not_promote_asset():
    record = verified_sparse_record()
    record["reference_verification"]["release_mapping_verified"] = False
    assert classify_asset(record)[0] == "conditional"
    record["reference_verification"]["release_mapping_verified"] = True
    record["reference_verification"]["roi_ids"] = [12345]
    assert classify_asset(record)[0] == "conditional"


def test_pretrained_overlap_blocks_only_the_relevant_comparator():
    record = verified_sparse_record()
    assert classify_asset(record)[0] == "eligible"
    status, reasons = classify_asset(record, comparator="pretrained_fdnc")
    assert status == "conditional"
    assert reasons == ["pretrained_fdnc_training_overlap_unresolved"]
    record["comparator_training_overlap"] = {"pretrained_fdnc": "verified_absent"}
    assert classify_asset(record, comparator="pretrained_fdnc") == ("eligible", [])


def test_documented_zephir_sparse_scope_handles_integer_frame_count():
    record = verified_sparse_record()
    record["observation"]["temporal_roi"] = {"schema": "zephir_worldlines",
        "complete": True, "frames": 4, "worldline_ids": [0, 42],
        "identity_semantics_documented": True}
    assert classify_asset(record) == ("eligible", [])
    assert record["validation_scope"] == "sparse_documented_subset"
    record["reference_verification"]["frame_indices"] = [1, 4]
    assert classify_asset(record)[0] == "conditional"
