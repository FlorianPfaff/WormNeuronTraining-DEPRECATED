"""Verify paper deliverables, source identities, and the safe bundle boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import BinaryIO

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "paper"
ARCHIVE_PREFIX = PurePosixPath("deformeot-paper")
MAX_ARCHIVE_MEMBER_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
FORBIDDEN_RAW_SUFFIXES = {".bin", ".h5", ".hdf5", ".nwb", ".npz", ".pth", ".zip"}
FIGURE_STEMS = (
    "figure-1-method",
    "figure-2-dual-anchor",
    "figure-3-extent",
    "figure-4-calibration",
    "figure-5-external",
    "figure-6-downstream",
    "figure-s1-density-sensitivity",
)
CANONICAL_RESULT_PATHS = (
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
PROSPECTIVE_RESULT_SHA256 = (
    "c210e6fe2350fa1e2988d4ad20579b3afc76ae33a7a49e82a10ff27235945b3a"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    while block := handle.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"absolute manifest/archive path: {value}")
    require(value == path.as_posix(), f"non-canonical manifest/archive path: {value}")
    require(path.parts and ".." not in path.parts, f"unsafe manifest/archive path: {value}")
    return path


def _manifest_local_path(relative: PurePosixPath, output_dir: Path) -> Path:
    output_prefix = PurePosixPath("outputs/paper")
    if relative == output_prefix:
        raise AssertionError("manifest path identifies the output directory, not a file")
    if relative.is_relative_to(output_prefix):
        return output_dir.joinpath(*relative.relative_to(output_prefix).parts)
    return ROOT.joinpath(*relative.parts)


def _load_manifest(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "reproduction-manifest.json"
    require(manifest_path.is_file(), "missing reproduction-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(isinstance(manifest, dict), "reproduction manifest is not a JSON object")
    return manifest


def verify_source_hashes(output_dir: Path) -> int:
    summary = json.loads((output_dir / "tables" / "results-summary.json").read_text())
    sources = summary["generated_from"]
    checked = 0
    for name, record in sources.items():
        path = ROOT / record["path"]
        frozen_copy = output_dir / "source-results" / f"{name}.json"
        require(path.is_file(), f"missing result source for {name}: {path}")
        require(frozen_copy.is_file(), f"missing copied result source for {name}")
        require(path.stat().st_size == record["bytes"], f"size mismatch for {name}")
        require(sha256(path) == record["sha256"], f"SHA-256 mismatch for {name}")
        require(
            frozen_copy.read_bytes() == path.read_bytes(),
            f"copied result source differs from canonical source for {name}",
        )
        checked += 1
    prospective = sources.get("prospective")
    require(isinstance(prospective, dict), "prospective result is absent from summary")
    require(
        prospective.get("sha256") == PROSPECTIVE_RESULT_SHA256,
        "canonical prospective result digest changed",
    )
    return checked


def verify_figures(output_dir: Path) -> int:
    checked = 0
    for stem in FIGURE_STEMS:
        png_path = output_dir / "figures" / f"{stem}.png"
        pdf_path = output_dir / "figures" / f"{stem}.pdf"
        require(png_path.is_file(), f"missing PNG figure: {png_path}")
        require(pdf_path.is_file(), f"missing PDF figure: {pdf_path}")
        with Image.open(png_path) as image:
            width, height = image.size
            image.verify()
        require(width >= 1200 and height >= 650, f"unexpectedly small figure: {stem}")
        require(
            len(PdfReader(str(pdf_path)).pages) == 1,
            f"figure PDF is not one page: {stem}",
        )
        checked += 1
    return checked


def _normalize_text(value: str) -> str:
    return " ".join(value.replace("*", "").replace("`", "").split()).casefold()


def verify_manuscript(output_dir: Path) -> int:
    repository_source = ROOT / "paper" / "manuscript.md"
    source_path = output_dir / "manuscript.md"
    pdf_path = output_dir / "manuscript.pdf"
    require(repository_source.is_file(), "missing paper/manuscript.md")
    require(source_path.is_file(), "missing outputs/paper/manuscript.md")
    require(pdf_path.is_file(), "missing manuscript.pdf")
    source_bytes = source_path.read_bytes()
    require(
        source_bytes == repository_source.read_bytes(),
        "outputs/paper/manuscript.md is stale relative to paper/manuscript.md",
    )
    try:
        source_text = source_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise AssertionError("manuscript source is not ASCII") from error
    for phrase in (
        "Anonymous draft - authorship to be supplied",
        "internally frozen analysis plan",
        "protocol deviation",
        "99.5%",
        "inconclusive",
        "split-evaluation",
    ):
        require(phrase.casefold() in source_text.casefold(), f"missing manuscript phrase: {phrase}")

    heading = next(
        (line[2:].strip() for line in source_text.splitlines() if line.startswith("# ")),
        "",
    )
    require(bool(heading), "manuscript has no level-one title")
    reader = PdfReader(str(pdf_path))
    require(len(reader.pages) >= 8, "manuscript PDF has fewer than eight pages")
    source_digest = hashlib.sha256(source_bytes).hexdigest()
    metadata = reader.metadata
    subject = str(metadata.subject or "") if metadata is not None else ""
    keywords = str(metadata.get("/Keywords", "")) if metadata is not None else ""
    require(
        f"source-sha256={source_digest}" in subject
        and f"source-sha256={source_digest}" in keywords,
        "manuscript PDF metadata does not identify the current Markdown source",
    )
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    require(
        _normalize_text(heading) in _normalize_text(text),
        "current Markdown title is missing from PDF text",
    )
    require("references" in text.casefold(), "references missing from PDF text")
    return len(reader.pages)


def verify_artifact_readme(output_dir: Path) -> None:
    path = output_dir / "README.md"
    require(path.is_file(), "missing outputs/paper/README.md")
    text = path.read_text(encoding="utf-8").casefold()
    for phrase in ("license", "build_reproduction_bundle.py", "sha256sum -c"):
        require(phrase in text, f"artifact README is missing: {phrase}")


def verify_manifest(output_dir: Path) -> int:
    manifest = _load_manifest(output_dir)
    require(manifest.get("schema_version") == 2, "unexpected manifest schema version")
    require(manifest.get("raw_data_included") is False, "manifest raw-data flag is not false")
    records = manifest.get("files")
    require(isinstance(records, list), "manifest files field is not a list")
    checked = 0
    seen: set[PurePosixPath] = set()
    for record in records:
        require(isinstance(record, dict), "manifest file record is not an object")
        relative = _safe_relative_path(str(record["path"]))
        require(relative not in seen, f"duplicate manifest path: {relative}")
        seen.add(relative)
        require(
            relative.suffix.lower() not in FORBIDDEN_RAW_SUFFIXES,
            f"raw-data-like manifest entry: {relative}",
        )
        path = _manifest_local_path(relative, output_dir)
        require(path.is_file(), f"manifest file missing: {relative}")
        require(not path.is_symlink(), f"manifest file is a symlink: {relative}")
        require(path.stat().st_size == record["bytes"], f"manifest size mismatch: {relative}")
        require(sha256(path) == record["sha256"], f"manifest SHA-256 mismatch: {relative}")
        checked += 1

    missing_results = set(map(PurePosixPath, CANONICAL_RESULT_PATHS)) - seen
    require(not missing_results, f"canonical result JSON absent from manifest: {missing_results}")
    return checked


def verify_bundle(output_dir: Path) -> int:
    archive_path = output_dir / "reproduction-bundle.tar.gz"
    require(archive_path.is_file(), "missing reproduction bundle")
    manifest = _load_manifest(output_dir)
    manifest_records = manifest["files"]
    expected_relative = {
        _safe_relative_path(str(record["path"])) for record in manifest_records
    }
    expected_relative.add(PurePosixPath("outputs/paper/reproduction-manifest.json"))
    expected_names = {ARCHIVE_PREFIX / relative for relative in expected_relative}

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        require(len(names) == len(set(names)), "bundle contains duplicate member names")
        for member, name in zip(members, names):
            _safe_relative_path(member.name)
            require(name.is_relative_to(ARCHIVE_PREFIX), f"bundle member escapes prefix: {name}")
            require(member.isreg(), f"bundle contains non-regular member: {name}")
            require(member.size <= MAX_ARCHIVE_MEMBER_BYTES, f"oversized bundle member: {name}")
            require(
                name.suffix.lower() not in FORBIDDEN_RAW_SUFFIXES,
                f"raw-data-like file found in bundle: {name}",
            )
        require(set(names) == expected_names, "bundle members differ from manifest allowlist")
        require(
            sum(member.size for member in members) <= MAX_ARCHIVE_BYTES,
            "bundle uncompressed size exceeds safety boundary",
        )

        records_by_name = {
            ARCHIVE_PREFIX / _safe_relative_path(str(record["path"])): record
            for record in manifest_records
        }
        for member in members:
            name = PurePosixPath(member.name)
            handle = archive.extractfile(member)
            require(handle is not None, f"cannot read bundle member: {name}")
            with handle:
                if name == ARCHIVE_PREFIX / "outputs/paper/reproduction-manifest.json":
                    require(
                        sha256_stream(handle)
                        == sha256(output_dir / "reproduction-manifest.json"),
                        "bundled reproduction manifest differs from delivered manifest",
                    )
                    continue
                record = records_by_name[name]
                require(member.size == record["bytes"], f"bundled size mismatch: {name}")
                require(
                    sha256_stream(handle) == record["sha256"],
                    f"bundled hash mismatch: {name}",
                )
    return len(members)


def _expected_checksum_paths(output_dir: Path) -> set[PurePosixPath]:
    manifest = _load_manifest(output_dir)
    output_prefix = PurePosixPath("outputs/paper")
    expected = {
        relative.relative_to(output_prefix)
        for record in manifest["files"]
        if (relative := _safe_relative_path(str(record["path"]))).is_relative_to(
            output_prefix
        )
    }
    expected.update(
        {
            PurePosixPath("reproduction-manifest.json"),
            PurePosixPath("reproduction-bundle.tar.gz"),
        }
    )
    return expected


def verify_checksums(output_dir: Path) -> int:
    checksum_path = output_dir / "SHA256SUMS.txt"
    require(checksum_path.is_file(), "missing SHA256SUMS.txt")
    checked = 0
    seen: set[PurePosixPath] = set()
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"malformed checksum line: {line}")
        expected, relative_text = match.groups()
        relative = _safe_relative_path(relative_text)
        require(relative not in seen, f"duplicate checksum path: {relative}")
        seen.add(relative)
        path = output_dir.joinpath(*relative.parts)
        require(path.is_file(), f"checksummed file missing: {relative}")
        require(not path.is_symlink(), f"checksummed file is a symlink: {relative}")
        require(sha256(path) == expected, f"checksum mismatch: {relative}")
        checked += 1
    require(
        seen == _expected_checksum_paths(output_dir),
        "checksum paths differ from the output allowlist",
    )
    return checked


def verify(output_dir: Path) -> dict[str, int | str]:
    required = (
        output_dir / "reproducibility.md",
        output_dir / "references.bib",
        output_dir / "source-results" / "prospective.json",
        output_dir / "tables" / "table-internal-target-ledger.csv",
        output_dir / "tables" / "table-reference-by-stride.csv",
        output_dir / "tables" / "table-density-sensitivity.csv",
        output_dir / "tables" / "table-prospective-by-stride.csv",
        output_dir / "tables" / "table-prospective-recordings.csv",
    )
    for path in required:
        require(path.is_file(), f"missing required artifact: {path}")
    verify_artifact_readme(output_dir)
    report: dict[str, int | str] = {
        "status": "PASS",
        "result_sources_checked": verify_source_hashes(output_dir),
        "figures_checked": verify_figures(output_dir),
        "manuscript_pages": verify_manuscript(output_dir),
        "manifest_entries_checked": verify_manifest(output_dir),
        "bundle_entries_checked": verify_bundle(output_dir),
        "checksums_checked": verify_checksums(output_dir),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(verify(args.output_dir.expanduser().absolute()), sort_keys=True))


if __name__ == "__main__":
    main()
