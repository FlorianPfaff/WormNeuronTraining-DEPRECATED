"""Range-backed access to WormID coordinate trajectories on DANDI.

The published EY NWB assets are about 1.4 GB each because they contain the
calcium image movie.  The prospective geometry benchmark needs only the small
``voxel_mask`` centre tables.  This module opens the immutable DANDI object with
HTTP byte ranges, reads those tables through h5py, and writes a compact local
coordinate cache.  Semantic neuron names are never used as persistent IDs;
stable ROI row indices define the initialized neuron-instance cohort.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import fsspec
import h5py
import numpy as np

from .wormid import WormIDRecording


_TIMEPOINT_RE = re.compile(r"^Seg_tpoint_(\d+)$")
_SEGMENTATION_PATH = "processing/CalciumActivity/CalciumSeriesSegmentation"
_LABEL_PATH = "processing/CalciumActivity/NeuronIDs/labels"
_SPACING_PATH = "general/optophysiology/CalciumImVol/grid_spacing"
_STARTING_TIME_PATH = "acquisition/CalciumImageSeries/starting_time"


@dataclass(frozen=True)
class DandiAsset:
    """Immutable source identity for one published DANDI NWB asset."""

    recording_id: str
    asset_id: str
    blob_id: str
    path: str
    content_size: int
    sha256: str
    dandi_etag: str
    content_url: str

    def __post_init__(self) -> None:
        if Path(self.path).stem != self.recording_id:
            raise ValueError("recording_id must equal the DANDI asset filename stem")
        if self.content_size <= 0:
            raise ValueError("DANDI asset size must be positive")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("DANDI SHA-256 must contain 64 lowercase hex digits")
        if not self.content_url.startswith("https://"):
            raise ValueError("DANDI content URL must use HTTPS")


def _json_request(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    """Read one DANDI JSON response with fsspec's shared HTTP implementation."""

    with fsspec.open(url, mode="rt", encoding="utf-8", timeout=timeout_seconds) as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return payload


def _direct_content_url(values: Iterable[object]) -> str:
    candidates = [str(value) for value in values]
    direct = [
        value
        for value in candidates
        if value.startswith("https://") and "api.dandiarchive.org" not in value
    ]
    if not direct:
        raise ValueError("DANDI asset metadata has no direct HTTPS content URL")
    return sorted(direct)[0]


def fetch_dandi_inventory(
    dandiset: str,
    version: str,
    *,
    timeout_seconds: float = 60.0,
) -> tuple[DandiAsset, ...]:
    """Fetch and validate the complete published NWB inventory from DANDI."""

    if not re.fullmatch(r"\d{6}", dandiset):
        raise ValueError("dandiset must be a six-digit identifier")
    listing_url = (
        "https://api.dandiarchive.org/api/dandisets/"
        f"{dandiset}/versions/{version}/assets/?page_size=100"
    )
    listing = _json_request(listing_url, timeout_seconds=timeout_seconds)
    results = listing.get("results")
    if not isinstance(results, list) or int(listing.get("count", -1)) != len(results):
        raise ValueError("DANDI asset listing was incomplete or malformed")
    if listing.get("next") is not None:
        raise ValueError("DANDI inventory exceeded the frozen single-page query")

    assets: list[DandiAsset] = []
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("DANDI listing contains a non-object asset")
        asset_id = str(row["asset_id"])
        detail = _json_request(
            f"https://api.dandiarchive.org/api/assets/{asset_id}/",
            timeout_seconds=timeout_seconds,
        )
        digest = detail.get("digest")
        if not isinstance(digest, dict):
            raise ValueError(f"DANDI asset {asset_id} has no digest object")
        path = str(row["path"])
        detail_path = str(detail.get("path", ""))
        if path != detail_path:
            raise ValueError(f"listing/detail path mismatch for DANDI asset {asset_id}")
        content_size = int(row["size"])
        if content_size != int(detail.get("contentSize", -1)):
            raise ValueError(f"listing/detail size mismatch for DANDI asset {asset_id}")
        if not path.endswith(".nwb"):
            raise ValueError(f"unexpected non-NWB asset in frozen DANDI set: {path}")
        content_urls = detail.get("contentUrl")
        if not isinstance(content_urls, list):
            raise ValueError(f"DANDI asset {asset_id} has no content URL list")
        assets.append(
            DandiAsset(
                recording_id=Path(path).stem,
                asset_id=asset_id,
                blob_id=str(row["blob"]),
                path=path,
                content_size=content_size,
                sha256=str(digest["dandi:sha2-256"]),
                dandi_etag=str(digest["dandi:dandi-etag"]),
                content_url=_direct_content_url(content_urls),
            )
        )
    ordered = tuple(sorted(assets, key=lambda asset: asset.recording_id))
    identifiers = [asset.recording_id for asset in ordered]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("DANDI inventory has duplicate recording identifiers")
    return ordered


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_coordinate_sha256(
    positions_zyx_um: np.ndarray,
    times_seconds: np.ndarray,
    spacing_zyx_um: np.ndarray,
) -> str:
    """Hash coordinate content independent of NPZ compression and timestamps."""

    digest = hashlib.sha256(b"wormid-coordinate-cache-semantic-v1\0")
    for name, values in (
        ("positions_zyx_um", positions_zyx_um),
        ("times_seconds", times_seconds),
        ("spacing_zyx_um", spacing_zyx_um),
    ):
        array = np.ascontiguousarray(values, dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _decode_label(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _extract_coordinate_arrays(nwb: h5py.File) -> dict[str, object]:
    """Read only the coordinate and timing objects needed by the benchmark."""

    for required in (
        _SEGMENTATION_PATH,
        _LABEL_PATH,
        _SPACING_PATH,
        _STARTING_TIME_PATH,
    ):
        if required not in nwb:
            raise ValueError(f"remote NWB is missing {required}")

    semantic_labels = tuple(_decode_label(value) for value in nwb[_LABEL_PATH][:])
    if not semantic_labels:
        raise ValueError("remote NWB contains no ROI rows")
    neuron_count = len(semantic_labels)
    spacing_xyz = np.asarray(nwb[_SPACING_PATH][:], dtype=float)
    if spacing_xyz.shape != (3,) or np.any(~np.isfinite(spacing_xyz)) or np.any(
        spacing_xyz <= 0
    ):
        raise ValueError("remote grid spacing must contain three positive values")

    segmentation = nwb[_SEGMENTATION_PATH]
    parsed: list[tuple[int, str]] = []
    for key in segmentation.keys():
        match = _TIMEPOINT_RE.match(key)
        if match:
            parsed.append((int(match.group(1)), key))
    parsed.sort()
    if not parsed or [index for index, _ in parsed] != list(range(len(parsed))):
        raise ValueError("remote segmentation time points must be contiguous from zero")

    expected_ids = np.arange(neuron_count, dtype=np.int64)
    expected_indices = np.arange(1, neuron_count + 1, dtype=np.int64)
    xyz_frames: list[np.ndarray] = []
    for _, key in parsed:
        plane = segmentation[key]
        for field in ("id", "voxel_mask", "voxel_mask_index"):
            if field not in plane:
                raise ValueError(f"{plane.name} is missing {field}")
        ids = np.asarray(plane["id"][:], dtype=np.int64)
        indices = np.asarray(plane["voxel_mask_index"][:], dtype=np.int64)
        mask = np.asarray(plane["voxel_mask"][:])
        required_fields = {"x", "y", "z", "weight"}
        if mask.dtype.names is None or not required_fields.issubset(mask.dtype.names):
            raise ValueError(f"{plane.name}/voxel_mask has unexpected fields")
        if not np.array_equal(ids, expected_ids):
            raise ValueError(f"{plane.name} ROI ids are not stable row indices")
        if len(mask) != neuron_count or not np.array_equal(indices, expected_indices):
            raise ValueError(f"{plane.name} is not one centre per persistent ROI")
        if not np.allclose(mask["weight"], 1.0):
            raise ValueError(f"{plane.name} has non-unit centre weights")
        xyz = np.column_stack([mask["x"], mask["y"], mask["z"]]).astype(float)
        if np.any(~np.isfinite(xyz)):
            raise ValueError(f"{plane.name} contains non-finite centres")
        xyz_frames.append(xyz)

    source_xyz = np.stack(xyz_frames)
    positions_zyx_um = source_xyz[..., ::-1] * spacing_xyz[::-1]
    starting_time = nwb[_STARTING_TIME_PATH]
    rate = float(starting_time.attrs.get("rate", np.nan))
    start = float(starting_time[()])
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(start):
        raise ValueError("remote starting_time must define a positive rate")
    times = start + np.arange(len(parsed), dtype=float) / rate
    semantic_serialization = json.dumps(
        semantic_labels, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "positions_zyx_um": positions_zyx_um,
        "source_xyz_voxels": source_xyz,
        "times_seconds": times,
        "spacing_zyx_um": spacing_xyz[::-1],
        "semantic_label_count": neuron_count,
        "semantic_blank_count": sum(not value.strip() for value in semantic_labels),
        "semantic_labels_sha256": hashlib.sha256(semantic_serialization).hexdigest(),
    }


def _cache_metadata(cache_path: Path) -> dict[str, object]:
    with np.load(cache_path, allow_pickle=False) as cached:
        raw = cached["metadata_json"]
        metadata = json.loads(str(raw.item()))
    if not isinstance(metadata, dict):
        raise ValueError(f"coordinate cache {cache_path} has invalid metadata")
    return metadata


def load_coordinate_cache(path: str | Path) -> tuple[WormIDRecording, dict[str, object]]:
    """Load and validate a compact coordinate cache as a WormID recording."""

    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=False) as cached:
        metadata = json.loads(str(cached["metadata_json"].item()))
        positions = np.asarray(cached["positions_zyx_um"], dtype=float)
        source_xyz = np.asarray(cached["source_xyz_voxels"], dtype=float)
        times = np.asarray(cached["times_seconds"], dtype=float)
        spacing = tuple(float(value) for value in cached["spacing_zyx_um"])
    if not isinstance(metadata, dict) or int(metadata.get("cache_schema_version", -1)) != 1:
        raise ValueError(f"unsupported coordinate cache schema in {cache_path}")
    neurons = int(positions.shape[1])
    labels = tuple(f"roi_{index:03d}" for index in range(neurons))
    recording = WormIDRecording(
        path=str(metadata["source_content_url"]),
        recording_id=str(metadata["recording_id"]),
        positions_zyx_um=positions,
        labels=labels,
        times_seconds=times,
        spacing_zyx_um=spacing,
        source_xyz_voxels=source_xyz,
    )
    if recording.frames != int(metadata["frames"]) or recording.neurons != int(
        metadata["neurons"]
    ):
        raise ValueError(f"coordinate cache dimensions disagree with metadata: {cache_path}")
    observed_semantic_hash = semantic_coordinate_sha256(positions, times, np.asarray(spacing))
    expected_semantic_hash = str(metadata.get("semantic_coordinate_sha256", ""))
    if observed_semantic_hash != expected_semantic_hash:
        raise ValueError(f"coordinate cache semantic digest mismatch: {cache_path}")
    return recording, metadata


def extract_coordinate_cache(
    asset: DandiAsset,
    cache_dir: str | Path,
    *,
    block_size_bytes: int = 256 * 1024,
    max_memory_blocks: int = 128,
    max_source_fraction: float = 0.10,
) -> tuple[WormIDRecording, dict[str, object]]:
    """Create or reuse a provenance-rich cache using HTTP byte-range reads only."""

    if block_size_bytes < 64 * 1024 or max_memory_blocks < 1:
        raise ValueError("range-cache block configuration is invalid")
    if not 0 < max_source_fraction < 1:
        raise ValueError("max_source_fraction must lie strictly between zero and one")
    destination = Path(cache_dir) / f"{asset.recording_id}.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        metadata = _cache_metadata(destination)
        expected = {
            "recording_id": asset.recording_id,
            "source_asset_id": asset.asset_id,
            "source_sha256_declared_by_dandi": asset.sha256,
            "source_content_size": asset.content_size,
        }
        if all(metadata.get(key) == value for key, value in expected.items()):
            recording, loaded_metadata = load_coordinate_cache(destination)
            return recording, {
                **loaded_metadata,
                "cache_path": str(destination.resolve()),
                "cache_size": destination.stat().st_size,
                "cache_sha256": sha256_file(destination),
                "cache_reused": True,
            }
        raise ValueError(
            f"existing cache provenance does not match immutable asset {asset.recording_id}"
        )

    filesystem = fsspec.filesystem("http")
    with filesystem.open(
        asset.content_url,
        mode="rb",
        size=asset.content_size,
        block_size=block_size_bytes,
        cache_type="blockcache",
        cache_options={"maxblocks": max_memory_blocks},
    ) as remote:
        if int(remote.size) != asset.content_size:
            raise ValueError(f"remote size changed for {asset.recording_id}")
        with h5py.File(remote, "r") as nwb:
            arrays = _extract_coordinate_arrays(nwb)
        requested_bytes = int(remote.cache.total_requested_bytes)
        range_misses = int(remote.cache.miss_count)
        range_hits = int(remote.cache.hit_count)

    source_fraction = requested_bytes / asset.content_size
    if source_fraction > max_source_fraction:
        raise RuntimeError(
            f"coordinate extraction requested {source_fraction:.1%} of the source asset"
        )
    positions = np.asarray(arrays["positions_zyx_um"], dtype=float)
    times = np.asarray(arrays["times_seconds"], dtype=float)
    spacing = np.asarray(arrays["spacing_zyx_um"], dtype=float)
    metadata: dict[str, object] = {
        "cache_schema_version": 1,
        "identity_semantics": "persistent stable ROI/neuron-instance row index",
        "semantic_labels_used": False,
        "recording_id": asset.recording_id,
        "source_asset_id": asset.asset_id,
        "source_blob_id": asset.blob_id,
        "source_path": asset.path,
        "source_content_url": asset.content_url,
        "source_content_size": asset.content_size,
        "source_sha256_declared_by_dandi": asset.sha256,
        "source_full_digest_locally_verified": False,
        "source_dandi_etag": asset.dandi_etag,
        "frames": int(positions.shape[0]),
        "neurons": int(positions.shape[1]),
        "semantic_label_count": int(arrays["semantic_label_count"]),
        "semantic_blank_count": int(arrays["semantic_blank_count"]),
        "semantic_labels_sha256": str(arrays["semantic_labels_sha256"]),
        "semantic_coordinate_sha256": semantic_coordinate_sha256(
            positions, times, spacing
        ),
        "range_block_size_bytes": int(block_size_bytes),
        "range_requests": range_misses,
        "range_cache_hits": range_hits,
        "range_requested_bytes": requested_bytes,
        "range_requested_fraction_of_asset": source_fraction,
    }
    temporary = destination.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            positions_zyx_um=positions,
            source_xyz_voxels=np.asarray(arrays["source_xyz_voxels"], dtype=float),
            times_seconds=times,
            spacing_zyx_um=spacing,
        )
    temporary.replace(destination)
    recording, loaded_metadata = load_coordinate_cache(destination)
    return recording, {
        **loaded_metadata,
        "cache_path": str(destination.resolve()),
        "cache_size": destination.stat().st_size,
        "cache_sha256": sha256_file(destination),
        "cache_reused": False,
    }


def asset_inventory_row(asset: DandiAsset) -> dict[str, object]:
    """Return a canonical JSON-ready source inventory row."""

    return asdict(asset)
