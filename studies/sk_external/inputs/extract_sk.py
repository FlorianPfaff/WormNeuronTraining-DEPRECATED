"""Extract six frozen WormID SK inputs. No tracking or label-based selection.

Requires the project's pinned Python environment plus numcodecs==0.16.3.
A private codec installation, if needed, belongs in inputs/.deps.
"""
import base64
import hashlib
import io
import json
import os
import platform
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT / ".deps"))
import numcodecs
from numcodecs import get_codec

SUBJECTS = (
    ("SK1_20230322_185704", "20230322-18-57-04", "3523d059-5d5c-4767-ac50-28cdbdb7ee25", 118),
    ("SK1_20230322_201650", "20230322-20-16-50", "36715c41-aa5c-4a05-affd-66a9a6089f7f", 117),
    ("SK1_20230506_150145", "20230506-15-01-45", "64d4ce85-1c87-4444-8758-61bb20d6dabc", 93),
    ("SK1_20230506_153351", "20230506-15-33-51", "bde173ba-5801-4008-baee-8d30b6341586", 103),
    ("SK2_20220426_w00", "2022-04-26-w00-NP1", "6d7709d2-0ae0-430c-ba4d-c7800496259f", 121),
    ("SK2_20220426_w01", "2022-04-26-w01-NP1", "8525ab3c-4a5d-4893-a2c8-4ac7ccbd87bd", 140),
)
SERIES = "acquisition/CalciumImageSeries"
SEG = "processing/CalciumActivity/CalciumSeriesSegmentation"
LABELS = "processing/CalciumActivity/NeuronIDs/labels"
CONVERTER = "https://github.com/focolab/NWBelegans/blob/74e976ac80c38facd396dadbf2f35d913dd0ed6d/NWB_convert.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()

def session():
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    return s

def zarray(refs, key, structured=False):
    meta = refs[key + "/.zarray"]
    raw = refs[key + "/0"]
    assert isinstance(raw, str), key
    encoded = base64.b64decode(raw[7:]) if raw.startswith("base64:") else raw.encode("latin1")
    raw_bytes = get_codec(meta["compressor"]).decode(encoded) if meta["compressor"] else encoded
    if structured:
        values = json.loads(raw_bytes)
        assert values[-1] == meta["shape"]
        return np.array(values[:-2])
    for filt in reversed(meta["filters"] or []):
        raw_bytes = get_codec(filt).decode(raw_bytes)
    if meta["filters"]:
        return np.asarray(raw_bytes).reshape(meta["shape"])
    return np.frombuffer(raw_bytes, dtype=meta["dtype"]).reshape(meta["shape"])

class RangeReader(io.RawIOBase):
    """Fetch only small byte ranges for native metadata and ROI-table checks."""
    def __init__(self, client, url, size):
        self.client = client
        self.url = url
        self.size = size
        self.pos = 0
        self.cache = {}
        self.payload = 0
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos
    def readinto(self, buffer):
        n = min(len(buffer), max(0, self.size - self.pos))
        if n > 512 * 1024:
            raise ValueError("Unexpectedly large native table read")
        i = 0
        block_size = 16384
        while i < n:
            block, within = divmod(self.pos, block_size)
            if block not in self.cache:
                if self.payload >= 6_000_000:
                    raise ValueError("Native spot-check byte budget exceeded")
                start = block * block_size
                stop = min(start + block_size, self.size)
                with self.client.get(self.url, headers={"Range": f"bytes={start}-{stop - 1}"}, stream=True, timeout=(12, 30)) as r:
                    assert r.status_code == 206 and r.headers["Content-Range"] == f"bytes {start}-{stop-1}/{self.size}"
                    payload = r.raw.read(stop - start + 1)
                assert len(payload) == stop - start
                self.payload += len(payload)
                self.cache[block] = payload
            data = self.cache[block][within:within + n - i]
            buffer[i:i + len(data)] = data
            i += len(data)
            self.pos += len(data)
        return n

def native_check(s, meta, subject, xyz_pixels, expected_ids, scale, rate, start, volume_path):
    source = next(u for u in meta["contentUrl"] if "s3." in u)
    frames = [0, len(xyz_pixels) // 2, len(xyz_pixels) - 1]
    with RangeReader(s, source, meta["contentSize"]) as reader:
        with h5py.File(reader, "r") as f:
            assert f["general/subject/subject_id"].asstr()[()] == subject
            series = f[SERIES]
            unit = series["starting_time"].attrs["unit"]
            assert unit in ("seconds", b"seconds")
            assert series["starting_time"].attrs["rate"] == rate and series["starting_time"][()] == start
            volume = series["imaging_volume"]
            assert volume.id == f[volume_path].id
            np.testing.assert_array_equal(volume["grid_spacing"][...], scale)
            assert volume["grid_spacing"].attrs["unit"] in ("micrometers", b"micrometers")
            for t in frames:
                group = f[f"{SEG}/Seg_tpoint_{t}"]
                np.testing.assert_array_equal(group["id"][...], expected_ids)
                mask = group["voxel_mask"][...]
                assert mask.dtype.names == ("x", "y", "z", "weight")
                xyz = np.column_stack([mask[axis] for axis in ("x", "y", "z")])
                np.testing.assert_array_equal(xyz, xyz_pixels[t])
                assert np.all(mask["weight"] == 1)
        return frames, reader.payload

def extract(subject_spec):
    key, subject, uuid, n = subject_spec
    family = key[:3]
    ds, version, nframes = ("000565", "0.241009.1504", 1500) if family == "SK1" else ("000472", "0.241009.1502", 3000)
    official = f"https://api.dandiarchive.org/api/dandisets/{ds}/versions/{version}/assets/{uuid}/"
    lindi = f"https://lindi.neurosift.org/dandi/dandisets/{ds}/assets/{uuid}/nwb.lindi.json"
    with session() as s:
        r = s.get(official, timeout=25)
        r.raise_for_status()
        meta = r.json()
        assert meta["identifier"] == uuid and meta["path"].startswith(f"sub-{subject}/")
        with s.get(lindi, headers={"Accept-Encoding": "gzip"}, stream=True, timeout=(12, 35)) as r:
            r.raise_for_status()
            payload = r.raw.read(32_000_001, decode_content=True)
            assert len(payload) <= 32_000_000
        ref_sha = digest(payload)
        index = json.loads(payload)
        gen, refs = index["generationMetadata"], index["refs"]
        assert gen["assetId"] == uuid and gen["assetPath"] == meta["path"] and gen["assetSize"] == meta["contentSize"]
        shape = refs[f"{SERIES}/data/.zarray"]["shape"]
        assert shape[0] == nframes
        volume_path = refs[f"{SERIES}/imaging_volume/.zattrs"]["_SOFT_LINK"]["path"].lstrip("/")
        assert volume_path == "general/optophysiology/CalciumImVol"
        grid = volume_path + "/grid_spacing"
        assert refs[grid + "/.zattrs"]["unit"] == "micrometers"
        scale = zarray(refs, grid)
        assert scale.shape == (3,) and np.all(scale > 0)
        tm = refs[f"{SERIES}/starting_time/.zattrs"]
        assert tm["unit"] == "seconds" and np.isfinite(tm["rate"]) and tm["rate"] > 0
        start = float(zarray(refs, f"{SERIES}/starting_time").item())
        assert start == 0 and tm["rate"] == (1.04 if family == "SK1" else 3.3)
        timestamps = start + np.arange(nframes, dtype=np.float64) / tm["rate"]
        pattern = re.compile(rf"{SEG}/Seg_tpoint_(\d+)/id/\.zarray")
        found = sorted(int(m[1]) for k in refs if (m := pattern.fullmatch(k)))
        assert found == list(range(nframes))
        ids = np.arange(n, dtype=np.int64)
        one_voxel_each = ids + 1
        xyz_pixels = np.empty((nframes, n, 3), dtype=np.int32)
        labels = zarray(refs, LABELS).tolist()
        assert len(labels) == n
        for t in found:
            group = f"{SEG}/Seg_tpoint_{t}/"
            np.testing.assert_array_equal(zarray(refs, group + "id"), ids)
            np.testing.assert_array_equal(zarray(refs, group + "voxel_mask_index"), one_voxel_each)
            mask = zarray(refs, group + "voxel_mask", structured=True)
            assert mask.shape == (n, 4) and np.all(mask[:, 3] == 1)
            xyz = mask[:, :3]
            assert np.isfinite(xyz).all() and np.all(xyz >= 0) and np.all(xyz < np.asarray(shape[1:]))
            assert np.all(xyz == np.floor(xyz)) and not np.any(np.all(xyz == 0, axis=1))
            xyz_pixels[t] = xyz
        sample_frames, spot_bytes = native_check(s, meta, subject, xyz_pixels, ids, scale, tm["rate"], start, volume_path)
    points = xyz_pixels.astype(np.float64) * scale
    out = OUT / f"{key}.npz"
    tmp = tempfile.NamedTemporaryFile(suffix=".npz", dir=OUT, delete=False)
    try:
        with tmp:
            np.savez_compressed(tmp, points=points, times=timestamps, roi_ids=ids, spacing_xyz_um=scale)
        with np.load(tmp.name, allow_pickle=False) as saved:
            assert saved.files == ["points", "times", "roi_ids", "spacing_xyz_um"]
            for name, expected in (("points", points), ("times", timestamps), ("roi_ids", ids), ("spacing_xyz_um", scale)):
                np.testing.assert_array_equal(saved[name], expected)
        os.replace(tmp.name, out)
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    print(key, nframes, n, "published", out.name, flush=True)
    return dict(id=key, family=family, subject_id=subject, asset_uuid=uuid, dandiset_version=version,
                dandiset_id=ds, asset_metadata_url=official, nwb_path=meta["path"], nwb_bytes=meta["contentSize"],
                dandi_reported_nwb_sha256=meta["digest"]["dandi:sha2-256"],
                lindi_url=lindi, lindi_json_sha256=ref_sha, frames_zero_based_inclusive=[0,nframes-1],
                roi_ids_zero_based_inclusive=[0,n-1], xyz_spacing_um=scale.tolist(), rate_hz=tm["rate"],
                time_start_s=start, last_nominal_time_s=timestamps[-1], points_shape=list(points.shape),
                named_roi_count=sum(bool(x) for x in labels), named_roi_count_with_question_mark=sum("?" in x for x in labels),
                native_nwb_spot_checked_frames=sample_frames, native_range_bytes=spot_bytes,
                npz=out.name, output_sha256=digest(out.read_bytes()),
                arrays_sha256={name:digest(a.tobytes(order="C")) for name,a in (("points",points),("times",timestamps),("roi_ids",ids),("spacing_xyz_um",scale))})

def main():
    assert platform.node()=="workstation2", "Run on gpuserver6000"
    with ProcessPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(extract, SUBJECTS))
    manifest = dict(schema="wha-sk-coordinate-inputs-v1", host=platform.node(), source_converter=CONVERTER,
                    reference="Persistent author-supplied Kato NWB ROI IDs. Source tracking was algorithm-assisted. Individual expert-checked frames are not identified.",
                    selection="All original rows and frames in the six inventoried SK1/SK2 NWBs, with no outcome or anatomical-name filtering.",
                    geometry="Source compound x, y, z times imaging-volume-linked grid spacing in micrometres; no origin shift, axis permutation, or registration.",
                    time_axis="Frame index divided by NWB nominal rate plus NWB starting time in seconds.",
                    checks="All frame keys consecutive; same IDs in original order for every frame; one in-bounds finite integer center with weight 1 per ROI per frame; no all-zero centers. Native DANDI NWB timestamps, calibration, subject and coordinate/ID tables at first, middle and last frame exact-checked against LINDI for each subject. All NPZ arrays round-trip checked.",
                    array_repr="points, times and spacing_xyz_um float64; roi_ids int64. Array SHA256 calculated on contiguous little-endian C-order bytes.",
                    versions=dict(python=platform.python_version(), numpy=np.__version__, h5py=h5py.__version__,numcodecs=numcodecs.__version__),
                    extraction_script_sha256=digest(Path(__file__).read_bytes()), recordings=outcomes)
    tmp=OUT / "input_manifest.json.partial"
    tmp.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    tmp.replace(OUT/"input_manifest.json")
    print("Input manifest complete",flush=True)

if __name__=="__main__":
    main()


