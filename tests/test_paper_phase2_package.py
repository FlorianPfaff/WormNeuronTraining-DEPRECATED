import hashlib
import io
import json
import tarfile

import pytest
import yaml

from deformeot import paper_phase2_package as packaging


def fixture_project(tmp_path):
    root, output, cache = tmp_path / 'project', tmp_path / 'project/work/results/paper_phase2_fixture', tmp_path / 'cache'
    for folder in ('src', 'configs', 'tests', 'scripts', 'docs', 'work/data/fdnc', 'outputs/paper/source-results'):
        (root / folder).mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True)
    cache.mkdir()
    names = [f'recording{index:02d}' for index in range(21)]
    protocol = {'dataset': {'tuning_recording': names[0], 'legacy_evaluation_recordings': names[1:5],
                            'untouched_recordings': names[5:]}}
    (root / 'configs/original.yaml').write_text(yaml.safe_dump(protocol))
    model = root / 'work/data/model.bin'
    model.write_bytes(b'pinned fixture checkpoint')
    config = {'original_protocol': 'configs/original.yaml', 'pyrecest_selection': 'configs/selection.json',
              'fdnc': {'checkpoint': 'work/data/model.bin', 'source': 'work/data/fdnc',
                       'checkpoint_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}}
    (root / 'configs/paper_phase2.yaml').write_text(yaml.safe_dump(config))
    for name in ('model.py', 'DNC_predict.py', 'Registration_transformer.py'):
        (root / 'work/data/fdnc' / name).write_text('# fixture\n')
    for name in ('docs/paper_phase2_protocol.md', 'docs/prospective_expansion_protocol.md',
                 'pyproject.toml', 'requirements-paper-pilot-lock.txt',
                 'configs/selection.json', *packaging.PRIOR_FILES):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}\n')
    for name in names:
        (cache / f'{name}.npz').write_bytes(f'fixture cache {name}'.encode())
    (output / 'budget.json').write_text('{"coordinator_running": true}\n')
    (output / 'checkpoints').mkdir()
    (output / 'checkpoints/done.json').write_text('{"status": "complete"}\n')
    (output / 'assignments').mkdir()
    (output / 'assignments/result.npz').write_bytes(b'fixture archived assignments')
    (root / 'outputs/paper_phase2_fixture').mkdir()
    (root / 'outputs/paper_phase2_fixture/report.md').write_text('Fixture report\n')
    return root, output, cache


def test_package_has_exact_caches_every_member_hashed_and_live_budget_caveat(tmp_path):
    root, output, cache = fixture_project(tmp_path)
    result = packaging.build_package(root, output, cache)
    manifest = json.loads((output / 'reproduction-manifest.json').read_text())
    listed = {row['path']: row for row in manifest['files']}
    with tarfile.open(result['bundle_path'], 'r:gz') as archive:
        assert set(archive.getnames()) == set(listed)
        assert len([name for name in archive.getnames() if name.startswith('inputs/coordinate_cache/')]) == 21
        for member in archive.getmembers():
            assert member.isfile() and not member.name.startswith('/') and '..' not in member.name.split('/')
            data = archive.extractfile(member).read()
            assert len(data) == listed[member.name]['bytes']
            assert hashlib.sha256(data).hexdigest() == listed[member.name]['sha256']
        note = json.loads(archive.extractfile('REPRODUCTION_BUNDLE.json').read())
        assert 'not final closed' in note['budget_snapshot']
    assert result['bundle_sha256'] == hashlib.sha256((output / 'reproduction-bundle.tar.gz').read_bytes()).hexdigest()
    assert result['detached_manifest_must_accompany_bundle']
    assert result['archive_readback_verification']['passed']
    assert result['archive_readback_verification']['verified_members'] == len(listed)


def test_package_excludes_old_massive_results_runtime_links_locks_and_itself(tmp_path):
    root, output, cache = fixture_project(tmp_path)
    (root / 'work/unrelated_movie.tif').write_bytes(b'not included')
    (root / 'runtime').mkdir()
    (root / 'runtime/python').write_bytes(b'not included')
    (output / 'supervisor.lock').write_text('live lock')
    (output / 'partial.tmp').write_text('unfinished')
    (root / 'src/linked_secret.py').symlink_to(root / 'work/unrelated_movie.tif')
    (output / 'reproduction-bundle.tar.gz').write_bytes(b'previous package')
    result = packaging.build_package(root, output, cache)
    with tarfile.open(result['bundle_path']) as archive:
        names = archive.getnames()
    assert not any('unrelated_movie' in name or 'linked_secret' in name or name.endswith('.lock')
                   or name.endswith('.tmp') or 'reproduction-bundle' in name for name in names)


def test_package_rejects_missing_extra_caches_and_wrong_model_hash(tmp_path):
    root, output, cache = fixture_project(tmp_path)
    extra = cache / 'extra.npz'
    extra.write_bytes(b'extra')
    with pytest.raises(ValueError, match='exactly match'):
        packaging.build_package(root, output, cache)
    extra.unlink()
    (root / 'work/data/model.bin').write_bytes(b'changed model')
    with pytest.raises(ValueError, match='checkpoint'):
        packaging.build_package(root, output, cache)


def test_preflight_ceiling_rejects_before_creating_bundle(tmp_path, monkeypatch):
    root, output, cache = fixture_project(tmp_path)
    monkeypatch.setattr(packaging, 'MAX_INPUT_BYTES', 1)
    with pytest.raises(ValueError, match='preflight'):
        packaging.build_package(root, output, cache)
    assert not (output / 'reproduction-bundle.tar.gz').exists()


def test_repeated_package_is_deterministic_when_sources_do_not_change(tmp_path):
    root, output, cache = fixture_project(tmp_path)
    first = packaging.build_package(root, output, cache)
    second = packaging.build_package(root, output, cache)
    assert first['bundle_sha256'] == second['bundle_sha256']
    assert first['manifest_sha256'] == second['manifest_sha256']


def test_readback_rejects_archive_or_per_member_hash_changes(tmp_path):
    root, output, cache = fixture_project(tmp_path)
    result = packaging.build_package(root, output, cache)
    manifest = json.loads((output / 'reproduction-manifest.json').read_text())
    manifest['files'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='member SHA256'):
        packaging.verify_package(result['bundle_path'], manifest)
    with (output / 'reproduction-bundle.tar.gz').open('ab') as stream:
        stream.write(b'fixture corruption')
    with pytest.raises(ValueError, match='archive SHA256'):
        packaging.verify_package(result['bundle_path'], output / 'reproduction-manifest.json')


def test_readback_rejects_duplicate_members_even_with_matching_archive_hash(tmp_path):
    root, output, cache = fixture_project(tmp_path)
    result = packaging.build_package(root, output, cache)
    manifest = json.loads((output / 'reproduction-manifest.json').read_text())
    with tarfile.open(result['bundle_path']) as archive:
        first = archive.getmembers()[0]
        data = archive.extractfile(first).read()
    with tarfile.open(result['bundle_path'], 'w:gz') as archive:
        archive.addfile(first, io.BytesIO(data))
        archive.addfile(first, io.BytesIO(data))
    path = output / 'reproduction-bundle.tar.gz'
    manifest.update(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), archive_bytes=path.stat().st_size)
    with pytest.raises(ValueError, match='duplicate member'):
        packaging.verify_package(path, manifest)
