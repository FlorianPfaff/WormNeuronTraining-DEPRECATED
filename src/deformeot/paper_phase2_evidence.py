"""Bounded public-asset evidence audit; never run a tracker or read image arrays.

Execute on a calculation server. HTTP payload bytes (including repeated reads)
are charged before use. HDF5 access is range-backed; a server ignoring Range is
rejected before its body is read. ZIP64 directories and allowlisted compact
members are inspected without extracting archives. Observed row continuity is
explicitly not treated as independent manual identity ground truth.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

import h5py
import numpy as np
import yaml


MIB = 1024 * 1024
HARD_LIMITS = {"max_seconds": 7200, "max_total_bytes": 512 * MIB,
               "max_asset_bytes": 32 * MIB, "max_member_bytes": 32 * MIB,
               "request_timeout_seconds": 30, "max_json_bytes": 2 * MIB,
               "range_block_bytes": 65536, "cache_blocks": 128}
SEGMENTATIONS = "processing/CalciumActivity/CalciumSeriesSegmentation"
FRAME_RE = re.compile(r"(?:Seg_tpoint_|.*_)(\d+)$")
COMPARATORS = ("untrained_geometry_mht", "pretrained_fdnc", "pretrained_three_dee")


class EvidenceLimitError(RuntimeError):
    """A requested read would exceed the approved compact-data budget."""


def _plain(value):
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()] if value.ndim else _plain(value.item())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _https(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("only credential-free public HTTPS URLs are permitted")
    return url


@dataclass
class ReadBudget:
    limits: dict
    clock: object = time.monotonic
    started: float = field(init=False)
    total_bytes: int = 0
    asset_bytes: dict = field(default_factory=dict)
    requests: list = field(default_factory=list)

    def __post_init__(self):
        self.limits = {**HARD_LIMITS, **self.limits}
        for key, hard in HARD_LIMITS.items():
            if not 0 < self.limits[key] <= hard:
                raise ValueError(f"{key} must be positive and no larger than {hard}")
        self.started = self.clock()

    def check(self, asset, requested=0):
        if self.clock() - self.started >= self.limits["max_seconds"]:
            raise EvidenceLimitError("wall-clock deadline reached")
        if self.total_bytes + requested > self.limits["max_total_bytes"]:
            raise EvidenceLimitError("total HTTP-payload budget reached")
        if self.asset_bytes.get(asset, 0) + requested > self.limits["max_asset_bytes"]:
            raise EvidenceLimitError(f"per-asset HTTP-payload budget reached: {asset}")

    def charge(self, asset, count):
        self.total_bytes += count
        self.asset_bytes[asset] = self.asset_bytes.get(asset, 0) + count

    def timeout(self):
        remaining = self.limits["max_seconds"] - (self.clock() - self.started)
        if remaining <= 0:
            raise EvidenceLimitError("wall-clock deadline reached")
        return min(self.limits["request_timeout_seconds"], remaining)

    def summary(self):
        return {"limits": self.limits, "http_payload_bytes": self.total_bytes,
                "per_asset_http_payload_bytes": self.asset_bytes,
                "wall_seconds": self.clock() - self.started,
                "request_count": len(self.requests),
                "accounting": "HTTP response body bytes; excludes TLS/HTTP headers. Repeated bytes count again.",
                "large_image_arrays_read": False, "archive_extractions": False}


class PublicReader:
    def __init__(self, budget, opener=None):
        self.budget = budget
        self.opener = opener or urlopen

    def _read(self, url, asset, maximum, byte_range=None):
        _https(url)
        self.budget.check(asset, maximum)
        headers = {"Accept-Encoding": "identity", "User-Agent": "deformeot-public-evidence/1"}
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        record = {"asset": asset, "url": url, "requested_range": byte_range,
                  "started_at": _utc(), "bytes": 0}
        began = self.budget.clock()
        self.budget.requests.append(record)
        try:
            with self.opener(Request(url, headers=headers), timeout=self.budget.timeout()) as stream:
                status = getattr(stream, "status", None) or stream.getcode()
                record["status"] = status
                _https(stream.geturl())
                encoding = stream.headers.get("Content-Encoding", "identity")
                if encoding not in ("identity", ""):
                    raise ValueError("encoded HTTP response violates byte-range accounting")
                length = stream.headers.get("Content-Length")
                if byte_range is not None:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                         stream.headers.get("Content-Range", ""))
                    if status != 206 or not match:
                        raise ValueError("range unsupported; full-body fallback is forbidden")
                    start, stop, total = map(int, match.groups())
                    if (start, stop) != byte_range or stop >= total:
                        raise ValueError("server returned an unexpected Content-Range")
                    record["remote_size"] = total
                elif status != 200:
                    raise ValueError(f"unexpected metadata HTTP status {status}")
                if length is not None and int(length) > maximum:
                    raise EvidenceLimitError("response exceeds approved compact read")
                chunks, remaining = [], maximum
                read_chunk = getattr(stream, "read1", stream.read)
                while remaining:
                    self.budget.check(asset)
                    # HTTPResponse.read1 yields available bytes instead of
                    # waiting to fill the entire buffer on a trickling server.
                    chunk = read_chunk(min(65536, remaining))
                    if not chunk:
                        break
                    self.budget.charge(asset, len(chunk))
                    record["bytes"] += len(chunk)
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                if byte_range is not None and len(payload) != maximum:
                    raise ValueError("truncated HTTP range")
                if byte_range is None and length is None and len(payload) == maximum:
                    raise EvidenceLimitError("unbounded metadata response reached compact limit")
                if length is not None and len(payload) != int(length):
                    raise ValueError("truncated metadata response")
                return payload
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["wall_seconds"] = self.budget.clock() - began

    def json(self, url, asset):
        return json.loads(self._read(url, asset, self.budget.limits["max_json_bytes"]))

    def range(self, url, asset, offset, count):
        if count == 0:
            return b""
        if offset < 0 or count < 0:
            raise ValueError("invalid byte range")
        return self._read(url, asset, count, (offset, offset + count - 1))


class RangeFile(io.RawIOBase):
    """Seekable bounded HTTP file; caching affects speed, never accounting."""

    def __init__(self, reader, url, size, asset, *, block_size=None):
        super().__init__()
        self.reader, self.url, self.size, self.asset = reader, url, int(size), asset
        self.position = 0
        self.block_size = block_size or reader.budget.limits["range_block_bytes"]
        self.cache = OrderedDict()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return position

    def read(self, count=-1):
        if count is None or count < 0:
            # zipfile's EOCD parser requests read-to-EOF after a bounded tail
            # seek. Permit that compact tail, never an archive/movie fallback.
            count = max(0, self.size - self.position)
        count = max(0, min(count, self.size - self.position))
        if count > self.reader.budget.limits["max_member_bytes"]:
            raise EvidenceLimitError("single read exceeds compact-data limit")
        output = bytearray()
        while count:
            self.reader.budget.check(self.asset)
            index, local = divmod(self.position, self.block_size)
            if index not in self.cache:
                offset = index * self.block_size
                self.cache[index] = self.reader.range(
                    self.url, self.asset, offset, min(self.block_size, self.size - offset))
                if self.reader.budget.requests[-1].get("remote_size") != self.size:
                    raise ValueError("remote object size changed from the frozen asset metadata")
                while len(self.cache) > self.reader.budget.limits["cache_blocks"]:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(index)
            block = self.cache[index]
            length = min(count, len(block) - local)
            output.extend(block[local:local + length])
            self.position += length
            count -= length
        return bytes(output)

    def readinto(self, target):
        data = self.read(len(target))
        target[:len(data)] = data
        return len(data)


def _compact(dataset, *, maximum=32 * MIB):
    """Only callers' explicitly selected compact datasets may reach this helper."""
    if dataset.ndim > 2 or dataset.size * dataset.dtype.itemsize > maximum:
        raise EvidenceLimitError(f"not a compact scalar/vector/table: {dataset.name}")
    # VLEN allocations are bounded by the remote budget, but refuse large tables.
    if dataset.dtype.hasobject and dataset.size > 100000:
        raise EvidenceLimitError(f"large variable-length table refused: {dataset.name}")
    return dataset[()]


def _dataset_metadata(dataset):
    return {"path": dataset.name, "shape": list(dataset.shape), "dtype": str(dataset.dtype),
            "attrs": {key: _plain(dataset.attrs[key]) for key in ("unit", "description", "rate")
                      if key in dataset.attrs}}


def _text_at(handle, path):
    return _plain(_compact(handle[path], maximum=65536)) if path in handle else None


def inspect_nwb(handle, check=lambda: None, result=None):
    """Inspect compact NWB ROI tables; no interpolation, image reads, or scoring."""
    result = {} if result is None else result
    result.update({"subject_id": _text_at(handle, "general/subject/subject_id"),
              "identifier": _text_at(handle, "identifier"),
              "session_id": _text_at(handle, "session_id"),
              "time": {}, "calibration": {}, "temporal_roi": {}, "semantic_labels": {}})
    series = handle.get("acquisition/CalciumImageSeries")
    if series is None or "data" not in series:
        result["temporal_roi"] = {"schema": "no_calcium_image_series", "complete": False}
        return result
    data = series["data"]
    result["image_metadata_only"] = _dataset_metadata(data)
    frames = int(data.shape[0]) if data.ndim else 0
    result["expected_frames"] = frames
    if "timestamps" in series:
        stamps = np.asarray(_compact(series["timestamps"]), dtype=float).reshape(-1)
        valid = bool(len(stamps) == frames and len(stamps) > 1 and
                     np.all(np.isfinite(stamps)) and np.all(np.diff(stamps) > 0))
        unit = _plain(series["timestamps"].attrs.get("unit"))
        result["time"] = {"source": series["timestamps"].name, "count": len(stamps),
                          "first": _plain(stamps[0]) if len(stamps) else None,
                          "last": _plain(stamps[-1]) if len(stamps) else None,
                          "unit": unit,
                          "strictly_increasing_complete": valid,
                          "verified": valid and unit in ("seconds", "second", "s")}
    elif "starting_time" in series:
        start = float(_compact(series["starting_time"]))
        rate = float(series["starting_time"].attrs.get("rate", float("nan")))
        unit = _plain(series["starting_time"].attrs.get("unit"))
        result["time"] = {"source": series["starting_time"].name, "start": _plain(start),
                          "rate_hz": _plain(rate), "count": frames,
                          "unit": unit,
                          "verified": bool(np.isfinite(start) and np.isfinite(rate) and rate > 0
                                           and unit in ("seconds", "second", "s"))}
    volumes = handle.get("general/optophysiology")
    candidates = []
    if volumes is not None:
        for name in volumes.keys():
            check()
            obj = volumes[name]
            if isinstance(obj, h5py.Group) and "grid_spacing" in obj:
                spacing = np.asarray(_compact(obj["grid_spacing"]), dtype=float).reshape(-1)
                unit = obj["grid_spacing"].attrs.get("unit", obj.attrs.get("grid_spacing_unit"))
                # NWB extension may store the unit in its own dataset.
                if unit is None and "grid_spacing_unit" in obj:
                    unit = _compact(obj["grid_spacing_unit"])
                candidates.append({"volume": name, "spacing": _plain(spacing),
                                   "unit": _plain(unit), "source": obj["grid_spacing"].name})
    result["calibration"]["available_volumes"] = candidates
    linked = series.get("imaging_volume")
    linked_name = next((x["volume"] for x in candidates
        if isinstance(linked, h5py.Group) and volumes[x["volume"]].id == linked.id), None)
    chosen = next((x for x in candidates if x["volume"] == linked_name), None)
    if chosen is None:
        chosen = next((x for x in candidates if x["volume"] == "CalciumImVol"), None)
    result["calibration"].update({"selected": chosen, "selection":
        "series_imaging_volume_link" if linked_name else "explicit_CalciumImVol_name_only"})
    result["calibration"]["verified"] = bool(chosen and len(chosen["spacing"]) == 3 and
        all(x is not None and x > 0 for x in chosen["spacing"]) and chosen["unit"])
    labels = handle.get("processing/CalciumActivity/NeuronIDs/labels")
    if isinstance(labels, h5py.Dataset):
        values = np.asarray(_compact(labels)).reshape(-1)
        names = [str(_plain(x)).strip() for x in values]
        result["semantic_labels"] = {**_dataset_metadata(labels), "entries": len(names),
            "nonempty": sum(x not in ("", "None", "nan") for x in names),
            "used_as_temporal_id": False}
    segmentation = handle.get(SEGMENTATIONS)
    if segmentation is None:
        result["temporal_roi"] = {"schema": "absent", "complete": False}
        return result
    if "data" in segmentation and isinstance(segmentation["data"], h5py.Dataset):
        result["temporal_roi"] = {"schema": "dense_segmentation_or_unknown_data",
            "dataset": _dataset_metadata(segmentation["data"]), "complete": False,
            "reason": "Dense segmentation movies are outside compact-only scope; no image fallback."}
        return result
    groups = []
    for name in segmentation.keys():
        check()
        match = FRAME_RE.fullmatch(name)
        if match and isinstance(segmentation[name], h5py.Group):
            groups.append((int(match.group(1)), name))
    groups.sort()
    if not groups:
        result["temporal_roi"] = {"schema": "unrecognized", "complete": False,
                                  "children": list(segmentation.keys())[:40]}
        return result
    frame_results, first_ids, all_ids = [], None, set()
    result["temporal_roi"] = {"schema": "per_frame_roi_tables", "complete": False,
        "observed_frame_groups": len(groups), "frames": frame_results,
        "inspection_finished": False, "independent_manual_truth_verified": False}
    for t, name in groups:
        check()
        group = segmentation[name]
        if "id" not in group or "voxel_mask" not in group:
            frame_results.append({"frame": t, "valid": False, "reason": "missing_id_or_mask"})
            continue
        ids = np.asarray(_compact(group["id"])).reshape(-1)
        mask = np.asarray(_compact(group["voxel_mask"]))
        indexes = (np.asarray(_compact(group["voxel_mask_index"])).reshape(-1)
                   if "voxel_mask_index" in group else None)
        if mask.dtype.names and all(x in mask.dtype.names for x in ("x", "y", "z")):
            xyz = np.column_stack([mask[x].reshape(-1) for x in ("x", "y", "z")])
        elif mask.ndim == 2 and mask.shape[1] in (3, 4):
            xyz = mask[:, :3]
        else:
            frame_results.append({"frame": t, "valid": False,
                                  "reason": "unrecognized_compact_voxel_mask"})
            continue
        serialized_ids = [_plain(x) for x in ids]
        unique = len(set(map(str, serialized_ids))) == len(ids)
        if first_ids is None:
            first_ids = serialized_ids
        all_ids.update(map(str, serialized_ids))
        valid_index = bool(indexes is None and len(xyz) == len(ids)) or bool(
            indexes is not None and len(indexes) == len(ids) and len(indexes) > 0 and
            indexes[-1] == len(xyz) and np.all(np.diff(np.r_[0, indexes]) > 0))
        finite = bool(np.all(np.isfinite(xyz)))
        item = {"frame": t, "rows": len(ids), "mask_points": len(xyz),
                "roi_ids": serialized_ids,
                "unique_ids": unique, "same_ordered_ids_as_first": serialized_ids == first_ids,
                "mask_index_valid": valid_index, "finite_coordinates": finite,
                "valid": bool(unique and valid_index and finite),
                "id_sha256": hashlib.sha256(json.dumps(serialized_ids).encode()).hexdigest()}
        if "ID_labels" in group:
            semantic = np.asarray(_compact(group["ID_labels"]))
            item["semantic_label_rows"] = int(semantic.shape[0]) if semantic.ndim else 1
        frame_results.append(item)
    times = [x[0] for x in groups]
    zero_based = times == list(range(frames))
    one_based = times == list(range(1, frames + 1))
    # One-based suffixes are recorded but cannot be silently shifted to movie time.
    complete = bool(zero_based and all(x.get("valid") for x in frame_results))
    result["temporal_roi"] = {"schema": "per_frame_roi_tables", "observed_frames": len(groups),
        "frame_index_first": times[0], "frame_index_last": times[-1],
        "zero_based_complete": zero_based, "one_based_complete_unmapped": one_based,
        "unique_roi_ids": len(all_ids), "first_frame_ids": first_ids,
        "all_ordered_ids_stable": bool(all(x.get("same_ordered_ids_as_first") for x in frame_results)),
        "complete": complete, "frames": frame_results,
        "inspection_finished": True,
        "interpretation": "Actual per-frame ROI id datasets inspected; row stability is not biological identity verification.",
        "identity_semantics_documented": False, "independent_manual_truth_verified": False}
    return result


def _verified_reference_scope(record):
    """Resolve documentary verification plus its exact observed release subset.

    Algorithm-assisted expert verification is acceptable. A generic paper claim
    or editable provenance tag is not an asset-level mapping. Sparse eligibility
    needs explicit ROI IDs and observed frames; neither all labels nor every
    frame in the movie must have been annotated by hand.
    """
    verification = record.get("reference_verification", {})
    roi = record.get("observation", {}).get("temporal_roi", {})
    scope = verification.get("scope")
    if (verification.get("status") != "documented_expert_verified" or
            verification.get("release_mapping_verified") is not True or
            verification.get("production") not in ("manual", "algorithm_assisted") or
            scope not in ("dense_all_frames_all_rois", "sparse_documented_subset")):
        return None
    try:
        _https(verification["source_url"])
        date.fromisoformat(verification["retrieved_on"])
    except (KeyError, ValueError, TypeError):
        return None
    if scope == "dense_all_frames_all_rois":
        return scope if roi.get("complete") else None
    ids = verification.get("roi_ids", [])
    frames = verification.get("frame_indices", [])
    if not isinstance(ids, list) or not ids or not isinstance(frames, list) or not frames:
        return None
    if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in frames):
        return None
    if len(set(map(str, ids))) != len(ids) or len(set(frames)) != len(frames):
        return None
    frame_rows = roi.get("frames", [])
    observed = ({row["frame"]: row for row in frame_rows}
                if isinstance(frame_rows, list) else {})
    if observed and all(frame in observed and observed[frame].get("valid") and
        set(map(str, ids)) <= set(map(str, observed[frame].get("roi_ids", []))) for frame in frames):
        return scope
    # A complete ZephIR table establishes subset coverage, not expert provenance;
    # the latter still requires the explicit verified documentary mapping above.
    if (roi.get("schema") == "zephir_worldlines" and roi.get("complete") and
            set(map(str, ids)) <= set(map(str, roi.get("worldline_ids", []))) and
            all(isinstance(t, int) and 0 <= t < roi.get("frames", 0) for t in frames)):
        return scope
    return None


def classify_asset(record, *, comparator="untrained_geometry_mht"):
    """Suitability for dense or explicitly scoped sparse external validation."""
    if comparator not in COMPARATORS:
        raise ValueError("unknown comparator profile")
    reasons = []
    if record.get("prior_exposure") == "known_exposed_alias":
        return "excluded", ["known_previously_exposed_recording_alias"]
    observation = record.get("observation", {})
    roi = observation.get("temporal_roi", {})
    if roi.get("schema") in ("no_calcium_image_series", "absent"):
        return "excluded", ["no_extractable_temporal_ROI_reference_in_asset"]
    if record.get("error"):
        reasons.append("bounded_inspection_incomplete")
    scope = _verified_reference_scope(record)
    record["validation_scope"] = scope
    if not roi.get("complete") and scope != "sparse_documented_subset":
        reasons.append("temporal_coordinate_coverage_not_verified_for_declared_scope")
    verification = record.get("reference_verification", {})
    if not (roi.get("identity_semantics_documented") or
            (scope and verification.get("temporal_identity_semantics_documented") is True)):
        reasons.append("persistent_biological_identity_semantics_not_verified")
    if scope is None:
        reasons.append("asset_or_subset_expert_verification_and_release_mapping_not_verified")
    if not observation.get("calibration", {}).get("verified"):
        reasons.append("physical_calibration_not_verified")
    if not observation.get("time", {}).get("verified"):
        reasons.append("asset_timing_not_verified")
    if not record.get("license_verified"):
        reasons.append("asset_license_not_verified")
    if comparator != "untrained_geometry_mht":
        overlap = record.get("comparator_training_overlap", {}).get(comparator, "unknown")
        if overlap not in ("verified_absent", "not_applicable_no_training"):
            reasons.append(f"{comparator}_training_overlap_unresolved")
    alias_review = record.get("alias_review", {})
    scoped_alias_clearance = bool(alias_review.get("status") == "reviewed_no_known_exposure" and
                                  alias_review.get("project_evidence_paths"))
    if not record.get("cross_release_aliases_resolved") and not scoped_alias_clearance:
        reasons.append("asset_alias_review_against_project_exposure_not_complete")
    return ("conditional", reasons) if reasons else ("eligible", [])


def _finalize_status(record):
    record["comparator_eligibility"] = {}
    for comparator in COMPARATORS:
        status, reasons = classify_asset(record, comparator=comparator)
        record["comparator_eligibility"][comparator] = {"status": status, "reasons": reasons}
    selected = record["comparator_eligibility"]["untrained_geometry_mht"]
    record["status"], record["reasons"] = selected["status"], selected["reasons"]
    record["status_scope"] = "untrained_geometry_mht_external_validation"
    record["evidence_interpretation"] = (
        "Unknown means this bounded inspection did not verify the required source linkage; "
        "it does not establish that suitable annotations or provenance do not exist.")


def _inspect_dandi(candidate, config, reader, metadata_cache):
    family = config["dandisets"][candidate["family"]]
    base = family["source_url"]
    key = candidate["family"]
    if key not in metadata_cache:
        version = reader.json(base, f"metadata_{key}")
        listing = reader.json(base + "assets/?page_size=100", f"metadata_{key}")
        if listing.get("next") or listing.get("count") != len(listing.get("results", [])):
            raise ValueError("incomplete DANDI asset listing; automatic expansion forbidden")
        metadata_cache[key] = (version, listing["results"])
    version, assets = metadata_cache[key]
    selected = [x for x in assets if x["path"] == candidate["path"]]
    if len(selected) != 1:
        raise ValueError("frozen asset path did not match exactly one public asset")
    row = selected[0]
    detail_url = f"https://api.dandiarchive.org/api/assets/{row['asset_id']}/"
    detail = reader.json(detail_url, candidate["id"])
    urls = [x for x in detail.get("contentUrl", [])
            if x.startswith("https://") and "api.dandiarchive.org" not in x]
    if not urls or detail.get("path") != candidate["path"]:
        raise ValueError("DANDI asset detail lacks consistent public content metadata")
    size = int(detail["contentSize"])
    if size != row["size"]:
        raise ValueError("DANDI listing and content size disagree")
    license_value = version.get("license", version.get("metadata", {}).get("license"))
    info = {"asset_id": row["asset_id"], "content_size": size,
            "declared_digest": detail.get("digest"), "content_url": urls[0],
            "source_url": detail_url, "retrieved_at": _utc(), "license": license_value,
            "license_verified": bool(license_value and "CC-BY-4.0" in str(license_value))}
    # Preserve metadata even if later bounded HDF access fails.
    candidate.update(info)
    candidate["observation"] = {}
    with RangeFile(reader, urls[0], size, candidate["id"]) as stream:
        with h5py.File(stream, "r") as handle:
            candidate["observation"] = inspect_nwb(
                handle, lambda: reader.budget.check(candidate["id"]), candidate["observation"])
    subject = candidate["observation"].get("subject_id")
    candidate["aliases"] = list(dict.fromkeys([candidate["path"].split("/")[0],
        Path(candidate["path"]).stem, str(subject) if subject is not None else "subject_id_missing"]))


def _zip_member(archive, name, maximum):
    info = archive.getinfo(name)
    if info.file_size > maximum or info.compress_size > maximum or info.flag_bits & 1:
        raise EvidenceLimitError("ZIP member is oversized or encrypted")
    with archive.open(info) as stream:
        data = stream.read(maximum + 1)
    if len(data) != info.file_size or len(data) > maximum:
        raise EvidenceLimitError("ZIP member violated compact-data bounds")
    return data


def _csv_summary(text_value):
    """Describe explicit long-form frame/id columns; never invent identity axes."""
    rows = list(csv.reader(io.StringIO(text_value)))
    if not rows:
        return {"rows": 0, "explicit_frame_and_id_columns": False}
    headers = [x.strip().lower() for x in rows[0]]
    frame_names = {"frame", "frame_id", "t", "t_idx", "timepoint", "time_point"}
    id_names = {"id", "track_id", "neuron_id", "cell_id", "worldline_id", "roi_id"}
    frame_col = next((i for i, x in enumerate(headers) if x in frame_names), None)
    id_col = next((i for i, x in enumerate(headers) if x in id_names), None)
    result = {"rows_after_first": len(rows) - 1, "first_row": rows[0],
              "row_width_counts": dict(Counter(len(row) for row in rows)),
              "explicit_frame_and_id_columns": frame_col is not None and id_col is not None}
    if result["explicit_frame_and_id_columns"]:
        pairs = [(row[frame_col], row[id_col]) for row in rows[1:]
                 if len(row) > max(frame_col, id_col)]
        result.update({"distinct_frame_values": len({x[0] for x in pairs}),
                       "distinct_id_values": len({x[1] for x in pairs}),
                       "duplicate_frame_id_rows": len(pairs) - len(set(pairs)),
                       "complete_temporal_coverage_verified": False,
                       "independent_manual_truth_verified": False})
    return result


def summarize_zephir(annotations, worldlines, metadata):
    """Keep worldline zero and provenance codes explicit; never guess manual truth."""
    with h5py.File(io.BytesIO(annotations), "r") as handle:
        arrays = {key: np.asarray(_compact(handle[key])).reshape(-1)
                  for key in ("t_idx", "worldline_id", "provenance", "x", "y", "z")}
    lengths = {len(x) for x in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("ZephIR column lengths differ")
    frames, counts = np.unique(arrays["t_idx"], return_counts=True)
    identities, id_counts = np.unique(arrays["worldline_id"], return_counts=True)
    pairs = np.column_stack([arrays["t_idx"], arrays["worldline_id"]])
    unique_pairs = len(np.unique(pairs, axis=0))
    codes, code_counts = np.unique(arrays["provenance"], return_counts=True)
    with h5py.File(io.BytesIO(worldlines), "r") as handle:
        ids = np.asarray(_compact(handle["id"])).reshape(-1)
        names = np.asarray(_compact(handle["name"])).reshape(-1)
    if len(ids) != len(names):
        raise ValueError("ZephIR worldline ID/name lengths differ")
    name_map = {int(k): str(_plain(v)) for k, v in zip(ids, names)}
    expected = int(metadata["shape_t"])
    zero_mask = arrays["worldline_id"] == 0
    zero_codes, zero_counts = np.unique(arrays["provenance"][zero_mask], return_counts=True)
    complete = bool(np.array_equal(frames, np.arange(expected)) and
        np.all(counts == len(identities)) and np.all(id_counts == expected) and
        unique_pairs == len(pairs))
    finite = all(bool(np.all(np.isfinite(arrays[k]))) for k in ("x", "y", "z"))
    return {"temporal_roi": {"schema": "zephir_worldlines", "complete": complete and finite,
        "rows": len(pairs), "frames": len(frames), "worldline_count_including_zero": len(identities),
        "worldline_ids": [int(x) for x in identities],
        "duplicate_frame_worldline_pairs": len(pairs) - unique_pairs,
        "rows_per_frame_min": int(counts.min()) if len(counts) else 0,
        "rows_per_frame_max": int(counts.max()) if len(counts) else 0,
        "finite_coordinates": finite, "coordinate_unit": "normalized_image_coordinates",
        "provenance_counts": {str(_plain(k)): int(v) for k, v in zip(codes, code_counts)},
        "worldline_zero": {"name": name_map.get(0), "rows": int(zero_mask.sum()),
            "provenance_counts": {str(_plain(k)): int(v) for k, v in zip(zero_codes, zero_counts)},
            "interpretation": "Schema reserves default name null, but release membership is reported, not silently removed or counted as a verified neuron."},
        "annotation_worldline_ids_missing_from_name_table": [int(x) for x in identities if int(x) not in name_map],
        "named_worldlines_without_annotations": [int(x) for x in ids if int(x) not in set(identities)],
        "identity_semantics_documented": True, "independent_manual_truth_verified": False},
        "calibration": {"verified": False, "metadata": metadata,
                        "reason": "No physical spacing established by inspected metadata."},
        "time": {"verified": False, "frame_indices": True,
                 "reason": "Paper reports 4 Hz; compact metadata has no verified timestamps or rate."}}


def _inspect_zephir(config, reader):
    setting = config["zephir"]
    result = {"id": setting["id"], "family": "zephir_public", "aliases": setting["aliases"],
              "prior_exposure": "not_found_in_project_ledger", "training_overlap": "unknown",
              "source_url": setting["record_url"], "retrieved_at": _utc()}
    for key in ("reference_verification", "comparator_training_overlap", "alias_review",
                "cross_release_aliases_resolved", "verified_animal_id"):
        if key in setting:
            result[key] = setting[key]
    try:
        record = reader.json(setting["record_url"], result["id"])
        result["license"] = record.get("metadata", {}).get("license")
        result["license_verified"] = "cc-by-4.0" in str(result["license"]).lower()
        matching = [x for x in record.get("files", []) if x.get("size") == setting["archive_size"]]
        if len(matching) != 1:
            raise ValueError("frozen ZephIR archive size no longer matches record")
        result["declared_archive_checksum"] = matching[0].get("checksum")
        with RangeFile(reader, setting["archive_url"], setting["archive_size"], result["id"]) as stream:
            with zipfile.ZipFile(stream) as archive:
                result["archive_member_count"] = len(archive.infolist())
                blobs = {name: _zip_member(archive, setting["prefix"] + name,
                    reader.budget.limits["max_member_bytes"])
                    for name in ("metadata.json", "worldlines.h5", "annotations.h5")}
        result["member_sha256"] = {name: hashlib.sha256(blob).hexdigest()
                                   for name, blob in blobs.items()}
        result["observation"] = summarize_zephir(blobs["annotations.h5"], blobs["worldlines.h5"],
                                                   json.loads(blobs["metadata.json"]))
        result["provenance_semantics"] = {"verified_code_map": setting["known_provenance_codes"],
                                         "unresolved": setting["unresolved_codes_note"]}
        result["sparse_validation_route"] = setting.get("sparse_validation_route")
        result["excluded_archive_aliases"] = {"nguyen2017_AL_w1": "previously_exposed_NeRVE",
            "churchland_CSHL049_2020_01_20_001": "mouse_not_approved_worm_cohort"}
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    _finalize_status(result)
    return result


def _inspect_three_dee(config, reader):
    setting = config["three_dee"]
    outputs = []
    try:
        listing = reader.json(setting["listing_url"], "metadata_three_dee")
        if listing.get("links", {}).get("next"):
            raise ValueError("OSF listing is paginated; automatic expansion forbidden")
        license_record = reader.json(setting["license_url"], "metadata_three_dee")
        node = reader.json(setting["node_url"], "metadata_three_dee")
        license_related = (node.get("data", {}).get("relationships", {}).get("license", {})
                           .get("links", {}).get("related", {}))
        related_url = (license_related.get("href") if isinstance(license_related, dict)
                       else license_related)
        license_name = license_record.get("data", {}).get("attributes", {}).get("name")
        related_data = (node.get("data", {}).get("relationships", {}).get("license", {})
                        .get("data") or {})
        related_id = related_data.get("id")
        linked_license = (str(related_url).rstrip("/") == setting["license_url"].rstrip("/") or
                          (related_id is not None and
                           related_id == license_record.get("data", {}).get("id")))
        verified = bool(linked_license and license_name and
                        "Attribution" in license_name and "4.0" in license_name)
        init_error = None
    except Exception as exc:
        listing, license_name, verified = {}, None, False
        init_error = f"{type(exc).__name__}: {exc}"
    for specification in setting["archives"]:
        result = {**specification, "family": "three_dee_cell_tracker_public",
            "source_url": setting["listing_url"], "retrieved_at": _utc(),
            "training_overlap": "unknown_recording_specific_segmentation_training_possible",
            "license": license_name, "license_verified": verified}
        try:
            if init_error:
                raise ValueError(init_error)
            rows = [x for x in listing.get("data", [])
                    if x.get("attributes", {}).get("name") == specification["filename"]]
            if len(rows) != 1:
                raise ValueError("OSF filename does not resolve uniquely")
            row = rows[0]
            result.update({"osf_file_id": row["id"], "content_size": row["attributes"]["size"],
                           "download_url": row["links"]["download"]})
            # Already exposed worm4 needs alias documentation, not another content read.
            if result["prior_exposure"] == "known_exposed_alias":
                result["content_inspection_skipped"] = "already_exposed_alias"
            else:
                with RangeFile(reader, result["download_url"], result["content_size"], result["id"]) as stream:
                    with zipfile.ZipFile(stream) as archive:
                        manifest = [{"name": item.filename, "bytes": item.file_size,
                                     "compressed_bytes": item.compress_size}
                                    for item in archive.infolist()]
                        result["manifest"] = manifest
                        texts = []
                        for item in archive.infolist():
                            reader.budget.check(result["id"])
                            if item.is_dir() or Path(item.filename).suffix.lower() not in setting["allowed_compact_extensions"]:
                                continue
                            if item.file_size > 2 * MIB:
                                texts.append({"name": item.filename, "skipped": "text_member_over_2MiB"})
                                continue
                            payload = _zip_member(archive, item.filename, 2 * MIB)
                            text_value = payload.decode("utf-8", errors="replace")
                            text_info = {"name": item.filename, "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "line_count": len(text_value.splitlines()),
                                "first_nonempty_lines": [line[:500] for line in text_value.splitlines() if line.strip()][:5],
                                "interpretation": "Compact text is evidence of a file, not independent manual temporal reference provenance."}
                            if Path(item.filename).suffix.lower() == ".csv":
                                text_info["table_summary"] = _csv_summary(text_value)
                            if Path(item.filename).suffix.lower() == ".json":
                                try:
                                    value = json.loads(text_value)
                                    text_info["json_top_level_keys"] = list(value)[:100] if isinstance(value, dict) else None
                                except (ValueError, TypeError):
                                    text_info["json_parse_failed"] = True
                            texts.append(text_info)
                        result["compact_text_members"] = texts
                result["observation"] = {"temporal_roi": {"schema": "archive_manifest_only",
                    "complete": False, "independent_manual_truth_verified": False},
                    "calibration": {"verified": False}, "time": {"verified": False}}
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        _finalize_status(result)
        outputs.append(result)
    return outputs


def _validate_config(config):
    if config.get("schema_version") != 1 or config.get("scope") != "public_compact_asset_evidence_only":
        raise ValueError("unsupported evidence configuration")
    date.fromisoformat(config["reviewed_on"])
    candidates = config["candidates"]
    if len(candidates) != 15 or len({x["path"] for x in candidates}) != 15:
        raise ValueError("exactly 15 unique approved asset paths are required")
    if [x["family"] for x in candidates] != ["SK1"] * 4 + ["SK2"] * 2 + ["KK"] * 9:
        raise ValueError("inspection order must be six SK candidates followed by nine KK")
    if len({x["id"] for x in candidates}) != 15:
        raise ValueError("candidate ids must be unique")
    papers = config["related_work"]["papers"]
    if not 1 <= len(papers) <= 10:
        raise ValueError("related-work review is bounded to ten primary papers")
    ids = {paper["id"] for paper in papers}
    for paper in papers:
        _https(paper["url"])
        date.fromisoformat(paper["retrieved_on"])
        for column in ("anchoring", "deformation", "future_frame_access", "hypothesis_retention",
                       "annotation_requirements", "evaluation_scope"):
            if not isinstance(paper.get(column), str) or not paper[column].strip():
                raise ValueError(f"paper {paper['id']} lacks comparison column {column}")
    for claim in config["related_work"]["claims"]:
        if not claim["sources"] or not set(claim["sources"]) <= ids:
            raise ValueError("claim table must cite reviewed papers")
    return config


def run_evidence(root, output_dir, *, opener=None, limits=None, progress=True):
    """Audit approved public compact assets; writes checkpoints and evidence.json.

    ``limits`` can only tighten the YAML/hard ceilings. ``opener`` allows offline
    HTTP fixtures in tests. No previous result is silently reused. Parent runner
    should additionally set a process timeout of 7200 seconds and run on server.
    """
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    path = root / "configs/paper_phase2_evidence.yaml"
    source = path.read_bytes()
    config = _validate_config(yaml.safe_load(source))
    configured_limits = {**config["limits"]}
    for key, value in (limits or {}).items():
        if key not in configured_limits or value > configured_limits[key]:
            raise ValueError("runtime overrides may only tighten reviewed bounds")
        configured_limits[key] = value
    budget = ReadBudget(configured_limits)
    reader = PublicReader(budget, opener)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema_version": 1, "started_at": _utc(), "source_config": str(path),
        "source_config_sha256": hashlib.sha256(source).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_identity_classification": config["reference_identity"]["classification"],
        "related_work": config["related_work"], "eligibility_policy": config["eligibility_policy"],
        "candidate_selection": config["candidate_selection"], "assets": [], "supplementary_assets": [],
        "execution": {"public_only": True, "external_test_scoring": False,
                      "training": False, "new_annotations": False, "author_contact": False,
                      "raw_movies": False},
        "limitations": ["Complete compact row coverage does not establish correct biological identities.",
                        "Unknown training overlap and cross-release aliases remain explicit.",
                        "HTTP byte accounting excludes protocol overhead; range-cache blocks may overlap adjacent archive bytes, but no image array is decoded.",
                        "Budget or network failures mean incomplete evidence, not absence of annotations.",
                        "This bounded closest-paper audit is not an exhaustive novelty search."]}

    def checkpoint():
        result["budget"] = budget.summary()
        (output_dir / "evidence.json").write_text(json.dumps(_plain(result), indent=2,
            sort_keys=True, allow_nan=False) + "\n")
        (output_dir / "http_reads.json").write_text(json.dumps(budget.requests, indent=2,
            sort_keys=True, allow_nan=False) + "\n")

    cache = {}
    for position, raw in enumerate(config["candidates"], 1):
        if progress:
            print(f"[{position}/15] compact evidence: {raw['id']}", flush=True)
        item = {"prior_exposure": "not_found_in_project_ledger",
                "training_overlap": "unknown", "cross_release_aliases_resolved": False,
                "annotation_provenance": config["dandisets"][raw["family"]]["annotation_provenance"],
                "provenance_source": config["dandisets"][raw["family"]]["source_url"],
                "provenance_retrieved_on": config["dandisets"][raw["family"]]["retrieved_on"],
                **raw}
        began = budget.clock()
        try:
            budget.check(item["id"])
            _inspect_dandi(item, config, reader, cache)
        except Exception as exc:
            item["error"] = f"{type(exc).__name__}: {exc}"
        _finalize_status(item)
        item["wall_seconds"] = budget.clock() - began
        item["http_payload_bytes"] = budget.asset_bytes.get(item["id"], 0)
        result["assets"].append(item)
        checkpoint()
    if progress:
        print("[supplementary] ZephIR ZIP64 compact annotations and 3DeeCellTracker manifests", flush=True)
    result["supplementary_assets"].append(_inspect_zephir(config, reader))
    checkpoint()
    result["supplementary_assets"].extend(_inspect_three_dee(config, reader))
    subject_sources = {}
    for item in result["assets"]:
        subject = item.get("observation", {}).get("subject_id")
        if subject is not None:
            subject_sources.setdefault(str(subject), []).append(item["id"])
    result["alias_audit"] = {
        "unique_frozen_asset_paths": len({x["path"] for x in result["assets"]}),
        "distinct_observed_subject_ids": len(subject_sources),
        "observed_subject_id_collisions": {key: values for key, values in subject_sources.items()
                                            if len(values) > 1},
        "independent_animal_count_verified": False,
        "interpretation": "Asset paths and subject identifiers are observed aliases; distinct strings do not prove distinct biological animals across releases."}
    counts = Counter(x["status"] for x in result["assets"])
    result["status_counts"] = {key: counts[key] for key in ("eligible", "conditional", "excluded")}
    eligible = [x for x in [*result["assets"], *result["supplementary_assets"]]
                if x["status"] == "eligible"]
    result["validation_scope_counts"] = {scope: sum(x.get("validation_scope") == scope for x in eligible)
        for scope in ("dense_all_frames_all_rois", "sparse_documented_subset")}
    verified_animals = {x["verified_animal_id"] for x in eligible if x.get("verified_animal_id")}
    dense_animals = {x["verified_animal_id"] for x in eligible if x.get("verified_animal_id")
                     and x.get("validation_scope") == "dense_all_frames_all_rois"}
    result["external_cohort_status"] = "eligible" if len(verified_animals) >= 2 else "conditional"
    result["verified_eligible_new_animals"] = len(verified_animals)
    result["eligible_asset_count"] = len(eligible)
    result["dense_external_cohort_verified"] = len(dense_animals) >= 2
    result["sparse_external_validation_available"] = result["validation_scope_counts"]["sparse_documented_subset"] > 0
    result["finished_at"] = _utc()
    checkpoint()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run_evidence(args.root, args.output_dir)
    print(json.dumps({"status_counts": result["status_counts"], "budget": result["budget"]}), flush=True)


if __name__ == "__main__":
    main()
