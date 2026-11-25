"""Orchestration contracts; execute with the central server test worker only."""

from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from deformeot import paper_phase2 as phase2
from deformeot.paper_pilot import select_cpd
from deformeot.tracker import DetectionFrame
from deformeot.wormid import IdentityBlindSequence
from deformeot.wormid_benchmark import TrackResult
from deformeot.wormid_pyrecest_mht import PyRecEstMHTConfig


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_FILES = ('paper_phase2.py', 'paper_phase2_mechanism.py',
                 'paper_phase2_rescore.py', 'paper_phase2_evidence.py')


def configuration():
    return yaml.safe_load((ROOT / 'configs/paper_phase2.yaml').read_text())


def source_context(revision=1):
    return {'signature': f'{revision:064x}',
            'source_hashes': {f'src/deformeot/{name}': f'{revision:064x}' for name in ADAPTER_FILES}}


def supervisor_fixture(tmp_path, monkeypatch, revision=1):
    clock = [0.]
    monkeypatch.setattr(phase2.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(phase2, 'storage_bytes', lambda root: 0)
    supervisor = phase2.Supervisor(tmp_path, tmp_path / 'results', tmp_path,
                                  configuration(), source_context(revision))
    return supervisor, clock


def test_cpd_grid_has_72_tasks_and_stable_explicit_tie_order():
    config = configuration()
    specs = phase2.cpd_specs(config)
    assert len(specs) == 24
    assert len(specs) * len(config['cpd']['strides']) == 72
    assert len({row['name'] for row in specs}) == 24
    assert [row['order'] for row in specs] == list(range(24))
    assert specs[0]['name'] == 'cpd_reference_a2_b1'
    assert specs[-1]['name'] == 'cpd_temporal_a10_b8'
    assert all(row['config']['w'] == 0 and row['config']['max_iterations'] == 100
               and row['config']['tolerance'] == 1e-5 for row in specs)
    rows = [{'status': 'complete', 'payload': {'candidate': spec['name'], 'stride': stride,
              'identity_accuracy': .9, 'identity_switch_rate': .01}}
            for spec in specs for stride in config['cpd']['strides']]
    selected = select_cpd(rows, specs, config['cpd']['strides'])
    assert [row['name'] for row in selected['selected']] == ['cpd_reference', 'cpd_temporal']
    assert [row['order'] for row in selected['selected']] == [0, 12]


def test_mechanism_specs_reuse_selected_geometry_config_without_mutating_defaults():
    config = configuration()
    before = asdict(PyRecEstMHTConfig())
    saved = json.loads((ROOT / config['pyrecest_selection']).read_text())
    frozen = next(row for row in saved['selected_candidates'] if row['name'] == 'mht_geometry_lag2')['config']
    specs = phase2.mechanism_specs(ROOT, config)
    assert [row['name'] for row in specs] == config['mechanism']['modes']
    assert len(specs) == 5
    assert all(row['config'] == frozen for row in specs)
    assert all(row['config']['use_gaussian_cost'] is False for row in specs)
    specs[0]['config']['fixed_lag'] = 77
    assert specs[1]['config']['fixed_lag'] == 2
    assert asdict(PyRecEstMHTConfig()) == before
    assert json.loads((ROOT / config['pyrecest_selection']).read_text()) == saved


def test_saved_assignments_independently_rescore_swap_and_recovery(tmp_path):
    points = np.asarray([[0., 0., 0.], [3., 1., 0.], [0., 4., 2.]])
    orders = [np.asarray(row, dtype=np.int64) for row in ([2, 0, 1], [0, 1, 2], [1, 2, 0], [2, 1, 0])]
    frames = tuple(DetectionFrame(points[order], time=float(index * 2)) for index, order in enumerate(orders))
    sequence = IdentityBlindSequence(frames, tuple(orders), ('n2', 'n0', 'n1'), (0, 10, 20, 30))
    identities_by_frame = [[2, 0, 1], [0, 2, 1], [0, 2, 1], [2, 0, 1]]
    assignments = tuple(np.asarray([int(np.flatnonzero(order == identity)[0]) for identity in desired], dtype=np.int64)
                        for order, desired in zip(orders, identities_by_frame))
    result = TrackResult('fixture', assignments, .1, np.asarray([0., 1., 2., 3.]))
    path = tmp_path / 'assignments/fixture.npz'
    saved = phase2.save_assignments(path, sequence, result)
    assert saved['independent_integer_agreement']
    assert saved['correct'] == 8 and saved['total'] == 12
    assert saved['identity_switches'] == 4
    assert saved['wrong_identity_frames'] == 4
    assert saved['wrong_identity_episode_count'] == 2
    assert saved['maximum_wrong_identity_duration_selected_scans'] == 2
    assert len(saved['assignment_file_sha256']) == 64
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays['assignments'].dtype == np.int64
        assert np.array_equal(arrays['frame_indices'], [0, 10, 20, 30])
        assert np.array_equal(arrays['frame_times'], [0., 2., 4., 6.])
        assert np.array_equal(arrays['truth_indices'], orders)


def test_ledger_charges_coordinator_and_each_live_worker_continuously(tmp_path, monkeypatch):
    supervisor, clock = supervisor_fixture(tmp_path, monkeypatch)
    initial = supervisor.ledger['charged_job_seconds']
    try:
        supervisor.mechanism_phase = True
        supervisor.live = {'mechanism_worker': {'mechanism': True}, 'other_worker': {'mechanism': False}}
        clock[0] = 10.
        supervisor.tick()
        assert supervisor.ledger['charged_job_seconds'] == initial + 30.
        assert supervisor.ledger['charged_mechanism_seconds'] == 20.
        supervisor.live.clear()
        supervisor.mechanism_phase = False
        clock[0] = 12.
        supervisor.tick()
        assert supervisor.ledger['charged_job_seconds'] == initial + 32.
        assert supervisor.ledger['charged_mechanism_seconds'] == 20.
    finally:
        supervisor.live.clear()
        supervisor.close()


def test_task_signature_changes_with_inputs_and_context_revision(tmp_path, monkeypatch):
    supervisor, _ = supervisor_fixture(tmp_path, monkeypatch)
    try:
        first = supervisor.task('mechanism', recording_id='animal', stride=1,
                                specification={'name': 'online', 'config': {'fixed_lag': 0}})
        changed = supervisor.task('mechanism', recording_id='animal', stride=10,
                                  specification={'name': 'online', 'config': {'fixed_lag': 0}})
        assert first['source_signature'] == supervisor.context['signature']
        assert first['task_signature'] != changed['task_signature']
        assert first['task_id'].endswith(supervisor.context['signature'][:12])
    finally:
        supervisor.close()


def test_completed_or_failed_checkpoint_is_immutable_and_stale_hash_rejected(tmp_path, monkeypatch):
    supervisor, _ = supervisor_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(phase2.subprocess, 'Popen', lambda *a, **k: pytest.fail('immutable failed task was retried'))
    try:
        task = supervisor.task('verify_adapters')
        checkpoint = {'task_id': task['task_id'], 'task_signature': task['task_signature'],
                      'phase': 'verify_adapters', 'status': 'failed', 'error': 'preserved fixture failure'}
        phase2.atomic_json(task['result_path'], checkpoint)
        assert supervisor.run([task]) == [checkpoint]
        checkpoint['task_signature'] = 'stale'
        phase2.atomic_json(task['result_path'], checkpoint)
        with pytest.raises(RuntimeError, match='signature mismatch'):
            supervisor.run([task])
    finally:
        supervisor.close()


def test_more_than_two_repair_attempts_are_rejected_without_a_worker(tmp_path, monkeypatch):
    supervisor, _ = supervisor_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(phase2.subprocess, 'Popen', lambda *a, **k: pytest.fail('repair limit launched a worker'))
    try:
        task = supervisor.task('verify_adapters')
        prefix = task['task_id'].rsplit('__', 1)[0]
        for revision in range(3):
            supervisor.ledger['jobs'][f'{prefix}__old{revision}'] = {'status': 'failed'}
        with pytest.raises(RuntimeError, match='two repair cycles'):
            supervisor.run([task])
    finally:
        supervisor.close()


def test_adapter_source_is_limited_to_initial_plus_two_repair_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(phase2, 'storage_bytes', lambda root: 0)
    for revision in (1, 2, 3):
        supervisor = phase2.Supervisor(tmp_path, tmp_path / 'results', tmp_path,
                                      configuration(), source_context(revision))
        supervisor.close()
    with pytest.raises(RuntimeError, match='two repair revisions'):
        phase2.Supervisor(tmp_path, tmp_path / 'results', tmp_path,
                          configuration(), source_context(4))


def test_budget_headroom_stops_scientific_launch_without_starting_process(tmp_path, monkeypatch):
    supervisor, _ = supervisor_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(phase2.subprocess, 'Popen', lambda *a, **k: pytest.fail('budget-exhausted task launched'))
    try:
        supervisor.ledger['charged_job_seconds'] = configuration()['limits']['cumulative_job_seconds'] - 1
        task = supervisor.task('mechanism')
        assert supervisor.run([task], mechanism=True) == []
        assert supervisor.ledger['stopped_batches'][-1]['pending'] == [task['task_id']]
    finally:
        supervisor.close()


def test_freeze_context_reuse_preserves_and_checks_source_archive_hash(tmp_path, monkeypatch):
    root, output = tmp_path / 'project', tmp_path / 'results'
    for folder in ('src', 'tests', 'configs', 'scripts'):
        (root / folder).mkdir(parents=True)
    (root / 'pyproject.toml').write_text('# fixture\n')
    (root / 'requirements-paper-pilot-lock.txt').write_text('# fixture\n')
    monkeypatch.setattr(phase2, 'provenance', lambda *args: source_context())
    monkeypatch.setattr(importlib.metadata, 'distributions', lambda: [])
    first = phase2.freeze_context(root, output, tmp_path / 'cache', configuration())
    second = phase2.freeze_context(root, output, tmp_path / 'cache', configuration())
    assert second['source_archive_sha256'] == first['source_archive_sha256']
    assert json.loads((output / 'active_context.json').read_text())['source_archive_sha256'] == first['source_archive_sha256']
    archive = output / 'contexts' / f"{first['signature']}.source.tar.gz"
    archive.write_bytes(b'fixture corruption')
    with pytest.raises((ValueError, RuntimeError), match='archive|SHA|hash'):
        phase2.freeze_context(root, output, tmp_path / 'cache', configuration())
