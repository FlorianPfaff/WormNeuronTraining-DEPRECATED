"""Build a deterministic, raw-data-free reproduction bundle for the paper artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "paper"
ARCHIVE_PREFIX = Path("deformeot-paper")
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 64 * 1024 * 1024
FORBIDDEN_BUNDLE_SUFFIXES = {".bin", ".h5", ".hdf5", ".nwb", ".npz", ".pth", ".zip"}

PACKAGE_NAMES = (
    "aiohttp",
    "fsspec",
    "h5py",
    "matplotlib",
    "networkx",
    "numpy",
    "pandas",
    "Pillow",
    "pypdf",
    "PyYAML",
    "reportlab",
    "scikit-learn",
    "scipy",
    "pytest",
    "ruff",
)

# These are the canonical, compact computational records behind the paper.  Large
# raw inputs and derived feature caches are identified in configs/data_manifest.yaml
# but are intentionally excluded from the archive.
REQUIRED_RESULT_PATHS = (
    "work/results/ascent_end_to_end_metrics.json",
    "work/results/ascent_official_evaluator.json",
    "work/results/fdnc_sequential_baselines.json",
    "work/results/targettrack_extent_benchmark.json",
    "work/results/wormid_downstream.json",
    "work/results/wormid_prospective_expansion.json",
    "work/results/wormid_reference_benchmark.json",
    "work/results/wormid_reference_summary.json",
    "work/results/wormid_reference_tuning.json",
    "work/results/wormid_severe_switches.json",
    "work/results/wormid_extent/benchmark_reproducible.json",
    "work/results/wormid_extent/density_definition_sensitivity.json",
    "work/results/wormid_extent/dual_anchor_extent_calibration_heldout.json",
    "work/results/wormid_extent/extent_component_ablation.json",
)

REQUIRED_PAPER_OUTPUT_PATHS = (
    "README.md",
    "reproducibility.md",
    "references.bib",
    "manuscript.md",
    "manuscript.pdf",
    "environment.txt",
    "figures/figure-1-method.png",
    "figures/figure-1-method.pdf",
    "figures/figure-2-dual-anchor.png",
    "figures/figure-2-dual-anchor.pdf",
    "figures/figure-3-extent.png",
    "figures/figure-3-extent.pdf",
    "figures/figure-4-calibration.png",
    "figures/figure-4-calibration.pdf",
    "figures/figure-5-external.png",
    "figures/figure-5-external.pdf",
    "figures/figure-6-downstream.png",
    "figures/figure-6-downstream.pdf",
    "figures/figure-s1-density-sensitivity.png",
    "figures/figure-s1-density-sensitivity.pdf",
    "source-results/ascent_current.json",
    "source-results/ascent_official.json",
    "source-results/calibration.json",
    "source-results/density_sensitivity.json",
    "source-results/downstream.json",
    "source-results/extent_components.json",
    "source-results/extent_reproducible.json",
    "source-results/fdnc.json",
    "source-results/prospective.json",
    "source-results/reference.json",
    "source-results/reference_raw.json",
    "source-results/severe_switches.json",
    "source-results/targettrack.json",
    "tables/results-summary.json",
    "tables/table-density-sensitivity.csv",
    "tables/table-prospective-by-stride.csv",
    "tables/table-prospective-recordings.csv",
    "tables/table-internal-target-ledger.csv",
    "tables/table-reference-by-stride.csv",
)

SOURCE_PATTERNS = (
    "src/**/*.py",
    "tests/**/*.py",
    "scripts/*.py",
    "configs/*.yaml",
    "configs/*.sha256",
    "configs/*.py",
    "docs/*.md",
    "paper/*.md",
    "paper/*.bib",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_files(paths: list[Path], *, label: str) -> list[Path]:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"missing required {label}: {rendered}")
    return paths


def _contains_symlink(path: Path, base: Path) -> bool:
    """Return whether *path* or a component below *base* is a symlink."""

    relative = path.relative_to(base)
    cursor = base
    if cursor.is_symlink():
        return True
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return True
    return False


def _validate_files(files: list[Path], output_dir: Path) -> None:
    root_resolved = ROOT.resolve()
    output_resolved = output_dir.resolve()
    archive_names: set[str] = set()
    total_bytes = 0
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"bundle input is not a regular file: {path}")
        if path.is_relative_to(output_dir):
            base = output_dir
            allowed = output_resolved
        elif path.is_relative_to(ROOT):
            base = ROOT
            allowed = root_resolved
        else:
            raise ValueError(f"bundle input is outside the repository and output: {path}")
        if _contains_symlink(path, base):
            raise ValueError(f"symlinked bundle input is not allowed: {path}")
        resolved = path.resolve()
        if not resolved.is_relative_to(allowed):
            raise ValueError(f"bundle input escapes its allowed root: {path}")
        if path.suffix.lower() in FORBIDDEN_BUNDLE_SUFFIXES:
            raise ValueError(f"raw-data-like bundle input is not allowed: {path}")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"bundle input exceeds {MAX_FILE_BYTES} bytes: {path}")
        total_bytes += size
        archive_name = relative_archive_path(path, output_dir).as_posix()
        if archive_name in archive_names:
            raise ValueError(f"duplicate archive path: {archive_name}")
        archive_names.add(archive_name)
    if total_bytes > MAX_TOTAL_INPUT_BYTES:
        raise ValueError(
            f"bundle inputs total {total_bytes} bytes, above {MAX_TOTAL_INPUT_BYTES}"
        )


def source_files(output_dir: Path) -> list[Path]:
    fixed = _require_files(
        [
            ROOT / "README.md",
            ROOT / "pyproject.toml",
            ROOT / "requirements-lock.txt",
        ],
        label="repository files",
    )
    selected = list(fixed)
    for pattern in SOURCE_PATTERNS:
        selected.extend(path for path in ROOT.glob(pattern) if path.is_file())

    selected.extend(
        _require_files(
            [ROOT / relative for relative in REQUIRED_RESULT_PATHS],
            label="canonical result JSON files",
        )
    )
    selected.extend(
        _require_files(
            [output_dir / relative for relative in REQUIRED_PAPER_OUTPUT_PATHS],
            label="paper outputs",
        )
    )
    files = sorted(set(selected), key=lambda path: path.as_posix())
    _validate_files(files, output_dir)
    return files


def write_environment(path: Path) -> None:
    lines = [
        "Paper artifact environment",
        "frozen_on=2026-07-20",
        "dependency_note=requirements-lock.txt pins direct and verification dependencies; "
        "it is not a transitive lock",
        f"python={platform.python_version()}",
        f"implementation={platform.python_implementation()}",
        f"platform={platform.platform()}",
        f"executable={sys.executable}",
    ]
    for package in PACKAGE_NAMES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "NOT INSTALLED"
        lines.append(f"{package}={version}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relative_archive_path(path: Path, output_dir: Path) -> Path:
    if path.is_relative_to(output_dir):
        return Path("outputs/paper") / path.relative_to(output_dir)
    if path.is_relative_to(ROOT):
        return path.relative_to(ROOT)
    raise ValueError(f"cannot assign an archive path outside the allowed roots: {path}")


def write_manifest(path: Path, files: list[Path], output_dir: Path) -> None:
    entries = [
        {
            "path": relative_archive_path(source, output_dir).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        }
        for source in files
    ]
    manifest = {
        "schema_version": 2,
        "frozen_on": "2026-07-20",
        "raw_data_included": False,
        "bundle_boundary": "explicit allowlist plus repository source patterns",
        "note": (
            "Canonical compact work/results JSON files are included. Use "
            "configs/data_manifest.yaml to verify separately downloaded raw data and "
            "derived feature caches."
        ),
        "files": entries,
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_deterministic_archive(path: Path, files: list[Path], output_dir: Path) -> None:
    _validate_files(files, output_dir)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                for source in files:
                    arcname = ARCHIVE_PREFIX / relative_archive_path(source, output_dir)
                    info = archive.gettarinfo(str(source), arcname=arcname.as_posix())
                    if not info.isreg():
                        raise ValueError(f"non-regular archive member is not allowed: {source}")
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mode = 0o644
                    with source.open("rb") as handle:
                        archive.addfile(info, handle)


def write_checksums(path: Path, files: list[Path], output_dir: Path) -> None:
    _validate_files(files, output_dir)
    lines = [
        f"{sha256(candidate)}  {candidate.relative_to(output_dir).as_posix()}"
        for candidate in sorted(
            files, key=lambda candidate: candidate.relative_to(output_dir).as_posix()
        )
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def build(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {output_dir}")
    reference_source = ROOT / "paper" / "references.bib"
    if not reference_source.is_file():
        raise FileNotFoundError(f"missing reference source: {reference_source}")
    (output_dir / "references.bib").write_bytes(reference_source.read_bytes())

    environment_path = output_dir / "environment.txt"
    manifest_path = output_dir / "reproduction-manifest.json"
    archive_path = output_dir / "reproduction-bundle.tar.gz"
    checksum_path = output_dir / "SHA256SUMS.txt"

    write_environment(environment_path)
    manifest_sources = source_files(output_dir)
    write_manifest(manifest_path, manifest_sources, output_dir)
    archive_sources = sorted(
        [*manifest_sources, manifest_path], key=lambda path: path.as_posix()
    )
    write_deterministic_archive(archive_path, archive_sources, output_dir)

    checksummed_outputs = [
        source for source in archive_sources if source.is_relative_to(output_dir)
    ]
    checksummed_outputs.append(archive_path)
    write_checksums(checksum_path, checksummed_outputs, output_dir)
    return {
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "files_in_archive": len(archive_sources),
        "checksummed_outputs": len(checksummed_outputs),
        "raw_data_included": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.output_dir.expanduser().absolute()), sort_keys=True))


if __name__ == "__main__":
    main()
