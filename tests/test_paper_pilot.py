from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from deformeot.paper_pilot import Supervisor, atomic_json, pilot_metrics, select_cpd, storage_bytes
from deformeot.paper_pilot_report import summarize
from deformeot.tracker import DetectionFrame
from deformeot.wormid import IdentityBlindSequence
from deformeot.wormid_benchmark import TrackResult


def limits(seconds=100, size=10**8):
    return {'limits': {'cumulative_job_seconds': seconds, 'additional_storage_bytes': size,
                       'task_timeout_seconds': 60, 'workers': 1}}


def test_storage_deduplicates_hardlinks_and_does_not_follow_symlinks(tmp_path):
    root = tmp_path / 'root'
    root.mkdir()
    item = root / 'data'
    item.write_bytes(b'0123456789')
    (root / 'hardlink').hardlink_to(item)
    other = tmp_path / 'other'
    other.write_bytes(b'a' * 100)
    (root / 'external').symlink_to(other)
    assert storage_bytes(root) == 10


def test_budget_accounts_external_jobs_and_stops_before_launch(tmp_path):
    atomic_json(tmp_path / 'accounting/external.json', {'wall_seconds': 20})
    supervisor = Supervisor(tmp_path, tmp_path / 'output', tmp_path, limits(seconds=15), 'a')
    assert supervisor.ledger['completed_job_seconds'] == 20
    assert supervisor.run([{'task_id': 'never_launch', 'phase': 'select'}]) == []
    assert supervisor.ledger['terminal_reason'] == 'cumulative_job_time_limit'
    assert not supervisor.ledger['jobs']


def test_stale_checkpoint_is_rejected(tmp_path):
    supervisor = Supervisor(tmp_path, tmp_path / 'output', tmp_path, limits(), 'a')
    atomic_json(tmp_path / 'output/checkpoints/x.json', {'task_signature': 'wrong', 'status': 'complete'})
    with pytest.raises(ValueError, match='stale checkpoint'):
        supervisor.run([{'task_id': 'x', 'phase': 'select'}])


def test_unreconciled_running_job_requires_inspection(tmp_path):
    atomic_json(tmp_path / 'output/budget.json', {'signature': 'a', 'jobs': {'x': {'status': 'running'}}})
    with pytest.raises(RuntimeError, match='unreconciled'):
        Supervisor(tmp_path, tmp_path / 'output', tmp_path, limits(), 'a')


def test_storage_limit_stops_without_starting_workers(tmp_path):
    supervisor = Supervisor(tmp_path, tmp_path / 'output', tmp_path, limits(size=1), 'a')
    assert supervisor.run([{'task_id': 'never_launch', 'phase': 'select'}]) == []
    assert supervisor.ledger['terminal_reason'] == 'additional_storage_limit'


def test_timeout_kills_worker_and_accounts_time(tmp_path, monkeypatch):
    original_popen = subprocess.Popen
    def sleeping_worker(command, **kwargs):
        return original_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
    monkeypatch.setattr('deformeot.paper_pilot.subprocess.Popen', sleeping_worker)
    config = limits()
    config['limits']['task_timeout_seconds'] = .1
    supervisor = Supervisor(tmp_path, tmp_path / 'output', tmp_path, config, 'a')
    rows = supervisor.run([{'task_id': 'sleep', 'phase': 'select'}], workers=1)
    assert rows[0]['status'] == 'timeout'
    assert supervisor.ledger['completed_job_seconds'] >= .1
    assert supervisor.ledger['jobs']['sleep']['status'] == 'timeout'


def test_cpd_selection_never_uses_incomplete_config_and_breaks_ties_by_order():
    specs = [{'name': name, 'anchor': 'reference', 'order': order} for order, name in enumerate(('a', 'b', 'c'))]
    rows = [{'status': 'complete', 'payload': {'candidate': name, 'stride': stride,
              'identity_accuracy': 1.0, 'identity_switch_rate': 0.0}}
            for name in ('a', 'b') for stride in (1, 10, 40)]
    winner = select_cpd(rows, specs, [1, 10, 40])
    assert winner['selected'][0]['order'] == 0
    assert not winner['leaderboard'][2]['complete']


def test_metrics_distinguish_stable_wrong_identity_from_recovery():
    points = np.asarray([[0., 0., 0.], [10., 0., 0.]])
    sequence = IdentityBlindSequence(tuple(DetectionFrame(points, time=i) for i in range(4)),
                                     tuple(np.arange(2) for _ in range(4)), ('a', 'b'), tuple(range(4)))
    result = TrackResult('fixture', tuple(np.asarray(row) for row in ([0, 1], [1, 0], [1, 0], [0, 1])),
                         1., np.asarray([0., 1., 1., 1.]))
    metrics = pilot_metrics(sequence, result)
    assert metrics['identity_switches'] == 4
    assert metrics['wrong_identity_frames'] == 4
    assert metrics['mean_wrong_identity_duration_selected_scans'] == 2
    assert metrics['maximum_wrong_identity_duration_selected_scans'] == 2


def test_report_requires_all_pilot_animals_and_uses_native_rate_only():
    rows = []
    for name, accuracy, switches in [('dual_anchor', .9, .02), ('cpd_reference', .9, .04)]:
        for recording in ('a', 'b', 'c'):
            rows.append({'phase': 'benchmark', 'status': 'complete', 'payload': {
                'candidate': name, 'recording_id': recording, 'stride': 1, 'identity_accuracy': accuracy,
                'identity_switch_rate': switches, 'mean_wrong_identity_duration_selected_scans': 2.,
                'runtime_seconds': 1., 'future_selected_scans': 0}})
    config = {'screen': {'minimum_switch_reduction_fraction': .25, 'maximum_accuracy_loss': .0025}}
    result = summarize(rows, ['a', 'b', 'c'], config)
    assert result['screen']['target_met']
    assert result['screen']['relative_switch_rate_reduction'] == .5
    assert not summarize(rows[:-1], ['a', 'b', 'c'], config)['screen']['available']
