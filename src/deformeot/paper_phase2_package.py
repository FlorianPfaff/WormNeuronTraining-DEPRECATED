"""Build a bounded portable archive of the new phase, never whole work folders.

The detached manifest hashes every regular member of the archive (including the
generated readme). Keeping it detached avoids a self-referential manifest hash.
The coordinator's budget is a live, pre-package snapshot: the closed final ledger
must accompany the archive separately after this worker and coordinator finish.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

import yaml


MAX_INPUT_BYTES = 1024 ** 3
EXCLUDED_NAMES = {'reproduction-bundle.tar.gz', 'reproduction-manifest.json', 'supervisor.lock'}
EXCLUDED_PARTS = {'__pycache__', '.pytest_cache', '.git', '.cache', 'runtime', 'tmp'}
PRIOR_FILES = ('outputs/paper/source-results/prospective.json',
               'work/results/pyrecest_followup_20261002/evaluation_results.json')
OPTIONAL_PRIOR_FILES = ('outputs/paper/reproduction-manifest.json', 'outputs/paper/environment.txt',
                        'work/results/wormid_prospective_expansion.json',
                        'work/results/wormid_reference_benchmark.json')


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_name(value):
    path = PurePosixPath(str(value).replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('archive names must be safe relative paths')
    return str(path)


def _allowed(path):
    return (not any(part in EXCLUDED_PARTS for part in path.parts)
            and path.name not in EXCLUDED_NAMES
            and path.suffix not in {'.tmp', '.lock', '.pyc', '.pyo'})


def _walk_files(folder):
    """Do not follow arbitrary directory or file symlinks during tree walks."""
    if not folder.is_dir():
        return
    for directory, subdirs, names in os.walk(folder, followlinks=False):
        subdirs[:] = sorted(name for name in subdirs
                           if name not in EXCLUDED_PARTS and not (Path(directory) / name).is_symlink())
        for name in sorted(names):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink() and _allowed(path.relative_to(folder)):
                yield path


def verify_package(bundle, manifest):
    """Read back every archived byte without extraction or executable loading."""
    bundle = Path(bundle)
    manifest = json.loads(Path(manifest).read_text()) if isinstance(manifest, (str, Path)) else manifest
    actual_sha = _hash_file(bundle)
    if actual_sha != manifest.get('archive_sha256') or bundle.stat().st_size != manifest.get('archive_bytes'):
        raise ValueError('archive SHA256 or byte size does not match the detached manifest')
    entries = manifest.get('files', [])
    expected = {}
    for entry in entries:
        name = _archive_name(entry['path'])
        size = entry['bytes']
        if name in expected:
            raise ValueError('duplicate member in detached manifest')
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError('manifest member byte size must be a nonnegative integer')
        expected[name] = entry
    if len(expected) != manifest.get('file_count') or sum(item['bytes'] for item in entries) >= MAX_INPUT_BYTES:
        raise ValueError('manifest file count or aggregate input bound is invalid')
    seen, total = set(), 0
    try:
        with tarfile.open(bundle, mode='r|gz') as archive:
            for member in archive:
                name = _archive_name(member.name)
                if not member.isfile() or member.name != name or name in seen:
                    raise ValueError('archive contains a link, noncanonical path, or duplicate member')
                if name not in expected or member.size != expected[name]['bytes']:
                    raise ValueError(f'unexpected member or incorrect archived size: {name}')
                seen.add(name)
                digest, count = hashlib.sha256(), 0
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError(f'archived member cannot be read: {name}')
                with stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        count += len(chunk)
                        total += len(chunk)
                        if count > member.size or total >= MAX_INPUT_BYTES:
                            raise ValueError('archive readback exceeded its declared safe byte bounds')
                        digest.update(chunk)
                if count != member.size or digest.hexdigest() != expected[name]['sha256']:
                    raise ValueError(f'archived member SHA256 or size mismatch: {name}')
    except (tarfile.TarError, OSError, EOFError) as error:
        raise ValueError(f'archive readback failed: {error}') from error
    if seen != set(expected) or total != manifest.get('total_input_bytes'):
        raise ValueError('archive member set or total payload size is incomplete')
    return {'passed': True, 'archive_sha256': actual_sha, 'verified_members': len(seen),
            'verified_payload_bytes': total, 'extracted_to_disk': False,
            'verification': 'Read back every regular tar member; exact path set, size and SHA256 verified against detached manifest.'}


def build_package(root, output, cache_dir):
    """Archive explicit source/results/data scope with a 1-GiB input ceiling."""
    root, output, cache_dir = map(lambda value: Path(value).resolve(), (root, output, cache_dir))
    config = yaml.safe_load((root / 'configs/paper_phase2.yaml').read_text())
    protocol_name = _archive_name(config['original_protocol'])
    protocol = yaml.safe_load((root / protocol_name).read_text())
    dataset = protocol['dataset']
    recordings = [dataset['tuning_recording'], *dataset['legacy_evaluation_recordings'],
                  *dataset['untouched_recordings']]
    if len(recordings) != 21 or len(set(recordings)) != 21:
        raise ValueError('the package requires exactly 21 unique protocol recordings')
    expected_cache_names = {f'{name}.npz' for name in recordings}
    if {path.name for path in cache_dir.glob('*.npz')} != expected_cache_names:
        raise ValueError('coordinate cache inventory must exactly match the 21 protocol recordings')
    members = {}

    def add(path, name, required=True):
        path, name = Path(path), _archive_name(name)
        if not path.is_file():
            if required:
                raise FileNotFoundError(f'required reproduction input missing: {path}')
            return
        if name in members and members[name].resolve() != path.resolve():
            raise ValueError(f'archive path collision: {name}')
        members[name] = path

    for folder in ('src', 'configs', 'tests', 'scripts'):
        if not (root / folder).is_dir():
            raise FileNotFoundError(f'required source directory missing: {folder}')
        for path in _walk_files(root / folder):
            add(path, path.relative_to(root).as_posix())
    for name in ('docs/paper_phase2_protocol.md', 'docs/prospective_expansion_protocol.md',
                 'pyproject.toml', 'requirements-paper-pilot-lock.txt',
                 protocol_name, _archive_name(config['pyrecest_selection']), *PRIOR_FILES):
        add(root / name, name)
    for name in OPTIONAL_PRIOR_FILES:
        add(root / name, name, required=False)
    model_name = _archive_name(config['fdnc']['checkpoint'])
    model = root / model_name
    if not model.is_file() or _hash_file(model) != config['fdnc']['checkpoint_sha256']:
        raise ValueError('fDNC checkpoint does not match its frozen SHA256')
    add(model, model_name)
    source_name = _archive_name(config['fdnc']['source'])
    source = root / source_name
    for required in ('model.py', 'DNC_predict.py', 'Registration_transformer.py'):
        if not (source / required).is_file():
            raise FileNotFoundError(f'released fDNC source missing: {required}')
    for path in sorted(source.glob('*.py')):
        add(path, f'{source_name}/{path.name}')
    for name in sorted(expected_cache_names):
        add(cache_dir / name, f'inputs/coordinate_cache/{name}')
    try:
        output_prefix = output.relative_to(root).as_posix()
    except ValueError:
        output_prefix = f'work/results/{output.name}'
    if not (output / 'budget.json').is_file():
        raise FileNotFoundError('the live budget snapshot is required')
    for path in _walk_files(output):
        add(path, f'{output_prefix}/{path.relative_to(output).as_posix()}')
    # Include only new-phase presentation outputs, never old full paper bundles.
    for folder in sorted((root / 'outputs').glob('paper_phase2*')):
        for path in _walk_files(folder):
            add(path, path.relative_to(root).as_posix())
    size = sum(path.stat().st_size for path in members.values())
    if size >= MAX_INPUT_BYTES:
        raise ValueError(f'archive input preflight exceeds the 1-GiB safety ceiling: {size} bytes')

    notes = {
        'schema_version': 1,
        'scope': 'new paper phase only; no runtime, raw movies or broad vendor/work directories',
        'coordinate_cache_count': 21,
        'coordinate_cache_path': 'inputs/coordinate_cache',
        'results_path': output_prefix,
        'manifest': 'The detached reproduction-manifest.json lists SHA256 and size for every bundled member.',
        'budget_snapshot': 'Live pre-package snapshot, not final closed accounting. The final budget.json must accompany this bundle separately.',
        'reproduction': 'Install the pinned requirements in a separate environment. Run python -m deformeot.paper_phase2 --help. Use the bundled input cache with --cache-dir inputs/coordinate_cache and a NEW output directory; archived absolute worker paths are provenance, not portable resume paths. The runner is intentionally restricted to gpuserver6000/workstation2.',
        'scientific_status': 'Previously examined development recordings; packaging does not make them an independent test set.',
    }
    readme = (json.dumps(notes, indent=2, sort_keys=True) + '\n').encode()
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / 'reproduction-bundle.tar.gz'
    temporary = output / 'reproduction-bundle.tar.gz.tmp'
    manifest_path = output / 'reproduction-manifest.json'
    entries, actual_total = [], 0
    with temporary.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name in sorted([*members, 'REPRODUCTION_BUNDLE.json']):
                    data = readme if name == 'REPRODUCTION_BUNDLE.json' else members[name].read_bytes()
                    actual_total += len(data)
                    if actual_total >= MAX_INPUT_BYTES:
                        raise ValueError('archive inputs grew beyond the 1-GiB ceiling during packaging')
                    info = tarfile.TarInfo(name)
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    archive.addfile(info, io.BytesIO(data))
                    entries.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    temporary.replace(bundle)
    manifest = {'schema_version': 1, 'files': entries, 'file_count': len(entries),
                'total_input_bytes': actual_total, 'input_safety_ceiling_bytes': MAX_INPUT_BYTES,
                'archive_sha256': _hash_file(bundle), 'archive_bytes': bundle.stat().st_size,
                'manifest_is_detached': True, **notes}
    manifest_tmp = output / 'reproduction-manifest.json.tmp'
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    manifest_tmp.replace(manifest_path)
    verification = verify_package(bundle, manifest_path)
    return {'bundle_path': str(bundle), 'bundle_sha256': manifest['archive_sha256'],
            'bundle_bytes': manifest['archive_bytes'], 'manifest_path': str(manifest_path),
            'manifest_sha256': _hash_file(manifest_path), 'file_count': len(entries),
            'coordinate_cache_count': 21, 'total_input_bytes': actual_total,
            'budget_snapshot': notes['budget_snapshot'], 'detached_manifest_must_accompany_bundle': True,
            'archive_readback_verification': verification}
