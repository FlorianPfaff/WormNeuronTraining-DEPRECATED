"""Resource-bounded, resumable paper-feasibility pilot; all outcomes exploratory.

The supervisor owns one locked budget ledger. Each numerical task runs in its
own process group, so timeouts/budget termination also stop audit subprocesses.
Historical code and results are only read. This module is deliberately separate
from the frozen original and PyRecEst experiment drivers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy.spatial.distance import cdist
import yaml

from .wormid import identity_blind_sequence
from .wormid_benchmark import ReferenceTrackerConfig, score_tracks, track_sequence
from .wormid_dandi import load_coordinate_cache, sha256_file
from .wormid_pyrecest_experiment import _validate_exhaustive_result
from .wormid_switch_strata import benchmark_frame_indices


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def storage_bytes(root):
    """Count unique regular-file bytes, without following links outside scope."""
    total, seen = 0, set()
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            try:
                stat = path.lstat()
            except FileNotFoundError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if path.is_symlink() or key in seen:
                continue
            seen.add(key)
            total += stat.st_size
    return total


def provenance(root, config, cache_dir=None):
    root = Path(root)
    sources = {}
    for folder in ('src/deformeot', 'tests', 'configs'):
        for path in sorted((root / folder).glob('*')):
            if path.suffix in ('.py', '.yaml', '.sha256'):
                sources[str(path.relative_to(root))] = sha256_file(path)
    model = root / config['fdnc']['checkpoint']
    if sha256_file(model) != config['fdnc']['checkpoint_sha256']:
        raise ValueError('fDNC checkpoint differs from the frozen public artifact')
    for path in sorted((root / config['fdnc']['source']).glob('*.py')):
        sources[str(path.relative_to(root))] = sha256_file(path)
    versions = {}
    for name in ('numpy', 'scipy', 'pandas', 'pycpd', 'pyrecest', 'torch', 'h5py', 'PyYAML'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    for name in (config['pyrecest_selection'], 'outputs/paper/source-results/prospective.json',
                 'work/results/pyrecest_followup_20261002/evaluation_results.json'):
        sources[name] = sha256_file(root / name)
    caches = {path.name: sha256_file(path) for path in sorted(Path(cache_dir).glob('*.npz'))} if cache_dir else {}
    value = {'source_hashes': sources, 'cache_hashes': caches, 'packages': versions,
             'python': platform.python_version(), 'platform': platform.platform(),
             'model_sha256': sha256_file(model), 'config': config}
    return {**value, 'signature': digest(value)}


def pilot_metrics(sequence, result):
    _validate_exhaustive_result(result, len(sequence.frames[0].positions), len(sequence.frames))
    metrics = asdict(score_tracks(sequence, result))
    identities = np.stack([truth[assignment] for truth, assignment in
                           zip(sequence.truth_indices, result.assignments)])
    wrong = identities != sequence.truth_indices[0][None, :]
    durations = []
    for column in wrong.T:
        edges = np.diff(np.r_[False, column, False].astype(np.int8))
        durations.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist())
    opportunities = (len(sequence.frames) - 1) * wrong.shape[1]
    metrics.update(identity_switch_opportunities=opportunities,
                   identity_switch_rate=metrics['identity_switches'] / opportunities if opportunities else 0.0,
                   wrong_identity_frames=int(wrong[1:].sum()),
                   accuracy_excluding_initialization=float(1 - wrong[1:].mean()),
                   wrong_identity_episode_count=len(durations),
                   mean_wrong_identity_duration_selected_scans=float(np.mean(durations)) if durations else 0.0,
                   maximum_wrong_identity_duration_selected_scans=max(durations, default=0),
                   assignment_sha256=hashlib.sha256(np.asarray(result.assignments, dtype='<i8').tobytes()).hexdigest(),
                   seeded_idf1_is_not_conventional_remapped_idf1=True,
                   coverage=1.0)
    return metrics


def motion_selection(cache_dir, original):
    ids = sorted([*original['dataset']['legacy_evaluation_recordings'],
                  *original['dataset']['untouched_recordings']])
    rows = []
    for name in ids:
        recording, _ = load_coordinate_cache(Path(cache_dir) / f'{name}.npz')
        values = []
        for previous, current in zip(recording.positions_zyx_um, recording.positions_zyx_um[1:]):
            matrix = cdist(previous, current)
            values.append(.5 * (np.median(matrix.min(axis=0)) + np.median(matrix.min(axis=1))))
        rows.append({'recording_id': name, 'median_motion_um': float(np.median(values)),
                     'cache_sha256': sha256_file(Path(cache_dir) / f'{name}.npz')})
    rows.sort(key=lambda row: (row['median_motion_um'], row['recording_id']))
    selected = [rows[index]['recording_id'] for index in (0, (len(rows) - 1) // 2, len(rows) - 1)]
    return {'selected_recordings': selected, 'ranking': rows,
            'selection_uses_tracking_outcomes': False,
            'interpretation': 'purposive development sample, not random or independent confirmation'}


def sequence_for(task, original):
    name = task['recording_id']
    recording, _ = load_coordinate_cache(Path(task['cache_dir']) / f'{name}.npz')
    ids = sorted([original['dataset']['tuning_recording'],
                  *original['dataset']['legacy_evaluation_recordings'],
                  *original['dataset']['untouched_recordings']])
    seed = original['association']['base_seed'] + 1000 * ids.index(name) + task['stride']
    frames = benchmark_frame_indices(recording.frames, task['stride'])
    sequence = identity_blind_sequence(recording, frame_indices=frames, seed=seed)
    return sequence, {'permutation_seed': seed, 'selected_frame_indices': list(map(int, frames)),
                      'cache_sha256': sha256_file(Path(task['cache_dir']) / f'{name}.npz')}


def run_tracking(sequence, specification, root, config, geometry):
    name = specification['method']
    if name in ('previous_hungarian', 'pairwise_affine', 'first_affine', 'dual_anchor'):
        return track_sequence(sequence, name, config=geometry)
    if name == 'cpd':
        from .paper_pilot_baselines import CPDConfig, track_cpd
        return track_cpd(sequence, anchor=specification['anchor'],
                         config=CPDConfig(**specification['config']), geometry=geometry)
    if name == 'fdnc':
        from .wormid_fdnc import FDNCTransferConfig, track_fdnc_reference
        return track_fdnc_reference(sequence, model_path=root / config['fdnc']['checkpoint'],
                                    source_directory=root / config['fdnc']['source'],
                                    config=FDNCTransferConfig(device='cpu', batch_size=32))
    if name == 'pyrecest_gnn':
        from .wormid_pyrecest_gnn import PyRecEstGNNConfig, track_pyrecest_gnn
        return track_pyrecest_gnn(sequence, config=PyRecEstGNNConfig(**specification['config']), geometry=geometry)
    if name == 'pyrecest_mht':
        from .wormid_pyrecest_mht import PyRecEstMHTConfig, track_pyrecest_mht
        return track_pyrecest_mht(sequence, config=PyRecEstMHTConfig(**specification['config']), geometry=geometry)
    raise ValueError(f'unsupported method {name}')


def execute_worker(task):
    started = time.monotonic()
    root, output = Path(task['root']), Path(task['output_dir'])
    config = yaml.safe_load((root / 'configs/paper_pilot.yaml').read_text())
    original = yaml.safe_load((root / config['original_protocol']).read_text())
    row = {'task_id': task['task_id'], 'task_signature': task['task_signature'],
           'phase': task['phase'], 'started_utc': utc_now(), 'host_load_start': list(os.getloadavg())}
    try:
        os.nice(10)
        if task['phase'] == 'select':
            payload = motion_selection(task['cache_dir'], original)
        elif task['phase'] == 'audit':
            from .paper_pilot_audit import run_audit
            payload = run_audit(root, task['cache_dir'], output / 'audit', task.get('historical_python'))
        elif task['phase'] == 'inventory':
            from .paper_pilot_inventory import run_inventory
            payload = run_inventory(root, output / 'inventory')
        elif task['phase'] == 'validate':
            from .paper_pilot_baselines import validate_fdnc
            validation_task = {**task, 'recording_id': original['dataset']['tuning_recording'], 'stride': 40}
            sequence, _ = sequence_for(validation_task, original)
            payload = validate_fdnc(root / config['fdnc']['checkpoint'], root / config['fdnc']['source'], sequence, device='cpu')
        elif task['phase'] == 'tests':
            result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=root, capture_output=True, text=True)
            (output / 'verification.log').write_text(result.stdout + result.stderr)
            payload = {'exit_code': result.returncode, 'log': str(output / 'verification.log')}
            if result.returncode:
                raise RuntimeError(f'test suite failed; see {output / "verification.log"}')
        else:
            sequence, inputs = sequence_for(task, original)
            geometry = ReferenceTrackerConfig(**original['association']['geometry'])
            result = run_tracking(sequence, task['specification'], root, config, geometry)
            payload = {'recording_id': task['recording_id'], 'stride': task['stride'],
                       'candidate': task['specification']['name'], 'specification': task['specification'],
                       **inputs, **pilot_metrics(sequence, result),
                       'future_selected_scans': task['specification'].get('config', {}).get('fixed_lag', 0)}
        row.update(status='complete', payload=payload)
    except Exception as error:
        row.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    row.update(wall_seconds=time.monotonic() - started, finished_utc=utc_now(),
               host_load_end=list(os.getloadavg()))
    atomic_json(task['result_path'], row)
    print(json.dumps({key: row[key] for key in ('task_id', 'status', 'wall_seconds')}), flush=True)


class Supervisor:
    def __init__(self, root, output, storage_root, config, signature, reconcile=False):
        self.root, self.output, self.storage_root = Path(root), Path(output), Path(storage_root)
        self.config, self.signature = config, signature
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / 'supervisor.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.output / 'budget.json'
        self.ledger = read_json(self.path) if self.path.exists() else {
            'signature': signature, 'completed_job_seconds': 0.0, 'jobs': {},
            'limit_job_seconds': config['limits']['cumulative_job_seconds'],
            'limit_storage_bytes': config['limits']['additional_storage_bytes'], 'created_utc': utc_now()}
        if not self.path.exists():
            external = []
            for path in sorted((self.storage_root / 'accounting').glob('*.json')):
                entry = read_json(path)
                if 'wall_seconds' not in entry or entry['wall_seconds'] < 0:
                    raise ValueError(f'invalid external accounting entry {path}')
                external.append(entry)
            self.ledger['external_jobs'] = external
            self.ledger['completed_job_seconds'] = sum(entry['wall_seconds'] for entry in external)
        if self.ledger['signature'] != signature:
            raise ValueError('pilot provenance changed: use a fresh output directory')
        # An abruptly killed supervisor must not silently discard its reservations.
        stale = [(name, row) for name, row in self.ledger['jobs'].items() if row.get('status') == 'running']
        if stale and not reconcile:
            raise RuntimeError('unreconciled running jobs in budget ledger; inspect before resume')
        for name, row in stale:
            if Path(f"/proc/{row['pid']}").exists():
                raise RuntimeError(f'refusing reconciliation while recorded PID {row["pid"]} exists')
            # Worst-case reservation conservatively accounts for an unobserved
            # end after a hard supervisor crash; no elapsed time is discarded.
            charge = config['limits']['task_timeout_seconds']
            self.ledger['completed_job_seconds'] += charge
            row.update(status='reconciled_interruption', wall_seconds=charge,
                       accounting='conservative_full_timeout_after_unobserved_exit')
        self.persist()

    def persist(self):
        self.ledger['updated_utc'] = utc_now()
        atomic_json(self.path, self.ledger)

    def run(self, tasks, workers=4):
        waiting, results, active = [], [], {}
        for task in tasks:
            identifier = task['task_id']
            path = self.output / 'checkpoints' / f'{identifier}.json'
            full = {**task, 'root': str(self.root), 'output_dir': str(self.output), 'result_path': str(path)}
            full['task_signature'] = digest({'signature': self.signature, 'task': task})
            if path.exists():
                old = read_json(path)
                if old.get('task_signature') != full['task_signature']:
                    raise ValueError(f'stale checkpoint {path}')
                if old.get('status') not in ('interrupted', 'budget_stopped'):
                    results.append(old)
                    continue
                history = path.parent / 'history'
                history.mkdir(exist_ok=True)
                archive = history / f'{identifier}__{time.time_ns()}.json'
                path.replace(archive)
            waiting.append(full)
        stopped = None
        try:
            while waiting or active:
                running_seconds = sum(time.monotonic() - value['start'] for value in active.values())
                self.ledger['storage_bytes'] = storage_bytes(self.storage_root)
                if self.ledger['completed_job_seconds'] + running_seconds >= self.ledger['limit_job_seconds'] - 5:
                    stopped = 'cumulative_job_time_limit'
                if self.ledger['storage_bytes'] >= self.ledger['limit_storage_bytes']:
                    stopped = 'additional_storage_limit'
                if stopped:
                    break
                while waiting and len(active) < workers:
                    task = waiting.pop(0)
                    path = self.output / 'tasks' / f"{task['task_id']}.json"
                    atomic_json(path, task)
                    log_path = self.output / 'logs' / f"{task['task_id']}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open('w')
                    environment = {**os.environ, 'PYTHONPATH': str(self.root / 'src'),
                                   'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                                   'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'deformeot.paper_pilot',
                                                '--worker-task', str(path)], cwd=self.root,
                                               env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    active[task['task_id']] = {'process': process, 'start': time.monotonic(), 'log': log, 'task': task}
                    self.ledger['jobs'][task['task_id']] = {'status': 'running', 'pid': process.pid,
                                                           'started_utc': utc_now(),
                                                           'attempt': self.ledger['jobs'].get(task['task_id'], {}).get('attempt', 0) + 1}
                    self.persist()
                for identifier, running in list(active.items()):
                    elapsed = time.monotonic() - running['start']
                    process = running['process']
                    timeout = elapsed >= self.config['limits']['task_timeout_seconds']
                    if timeout and process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    if process.poll() is None:
                        continue
                    running['log'].close()
                    task = running['task']
                    path = Path(task['result_path'])
                    row = read_json(path) if path.exists() else {
                        'task_id': identifier, 'task_signature': task['task_signature'], 'phase': task['phase'],
                        'status': 'timeout' if timeout else 'failed', 'exit_code': process.returncode,
                        'error': 'worker did not return a complete result', 'wall_seconds': elapsed}
                    atomic_json(path, row)
                    self.ledger['completed_job_seconds'] += elapsed
                    self.ledger['jobs'][identifier].update(status=row['status'], wall_seconds=elapsed)
                    results.append(row)
                    del active[identifier]
                    self.persist()
                    print(f"{identifier}: {row['status']} ({elapsed:.1f}s)", flush=True)
                if active:
                    time.sleep(.5)
        finally:
            for identifier, running in active.items():
                process = running['process']
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                elapsed = time.monotonic() - running['start']
                running['log'].close()
                task = running['task']
                row = {'task_id': identifier, 'task_signature': task['task_signature'], 'phase': task['phase'],
                       'status': 'budget_stopped' if stopped else 'interrupted',
                       'error': stopped or 'supervisor interrupted', 'wall_seconds': elapsed}
                atomic_json(task['result_path'], row)
                results.append(row)
                self.ledger['completed_job_seconds'] += elapsed
                self.ledger['jobs'][identifier].update(status=row['status'], wall_seconds=elapsed)
            if stopped:
                self.ledger['terminal_reason'] = stopped
            self.persist()
        return results


def select_cpd(rows, specifications, strides):
    winners, leaderboard = [], []
    for specification in specifications:
        complete = [row['payload'] for row in rows if row['status'] == 'complete'
                    and row['payload'].get('candidate') == specification['name']]
        valid = len(complete) == len(strides) and {row['stride'] for row in complete} == set(strides)
        entry = {'specification': specification, 'complete': valid}
        if valid:
            entry.update(accuracy=float(np.mean([row['identity_accuracy'] for row in complete])),
                         switch_rate=float(np.mean([row['identity_switch_rate'] for row in complete])))
        leaderboard.append(entry)
    for anchor in ('reference', 'temporal'):
        valid = [entry for entry in leaderboard if entry['complete'] and entry['specification']['anchor'] == anchor]
        if valid:
            best = min(valid, key=lambda entry: (-entry['accuracy'], entry['switch_rate'], entry['specification']['order']))
            winners.append({**best['specification'], 'name': f'cpd_{anchor}'})
    return {'selected': winners, 'leaderboard': leaderboard, 'selection_uses_pilot_outcomes': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('audit', 'validate', 'benchmark', 'inventory', 'report', 'all'), default='all')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('work/results/paper_pilot_20261003'))
    parser.add_argument('--storage-root', type=Path)
    parser.add_argument('--historical-python', type=str)
    parser.add_argument('--reconcile-interrupted', action='store_true',
                        help='reconcile dead orphan jobs, conservatively charging their full timeout')
    parser.add_argument('--worker-task', type=Path)
    args = parser.parse_args()
    if args.worker_task:
        execute_worker(read_json(args.worker_task))
        return
    root, output = args.root.resolve(), args.output_dir.resolve()
    config = yaml.safe_load((root / 'configs/paper_pilot.yaml').read_text())
    if args.phase == 'report':
        from .paper_pilot_report import generate_report
        print(json.dumps(generate_report(root, output, config), indent=2))
        return
    if args.cache_dir is None or args.storage_root is None:
        parser.error('--cache-dir and --storage-root are required for execution')
    context = provenance(root, config, args.cache_dir)
    supervisor = Supervisor(root, output, args.storage_root, config, context['signature'], args.reconcile_interrupted)
    atomic_json(output / 'context.json', context)
    def finish_if_stopped():
        if supervisor.ledger.get('terminal_reason'):
            from .paper_pilot_report import generate_report
            print(json.dumps(generate_report(root, output, config), indent=2), flush=True)
            return True
        return False
    def prerequisite(path):
        return path.is_file() and read_json(path).get('status') == 'complete'
    common = {'cache_dir': str(args.cache_dir.resolve())}
    def phase_task(phase):
        return {**common, 'phase': phase, 'task_id': phase,
                **({'historical_python': args.historical_python} if phase == 'audit' else {})}
    if args.phase in ('audit', 'all'):
        supervisor.run([phase_task('audit')], workers=1)
        if finish_if_stopped():
            return
    if args.phase in ('inventory', 'all'):
        supervisor.run([phase_task('inventory')], workers=1)
        if finish_if_stopped():
            return
    if args.phase in ('validate', 'all'):
        supervisor.run([phase_task('tests'), phase_task('validate')], workers=1)
        if finish_if_stopped():
            return
    if args.phase in ('benchmark', 'all'):
        validation = output / 'checkpoints' / 'validate.json'
        tests = output / 'checkpoints' / 'tests.json'
        if not prerequisite(tests):
            from .paper_pilot_report import generate_report
            print(json.dumps(generate_report(root, output, config), indent=2), flush=True)
            return
        supervisor.run([phase_task('select')], workers=1)
        if finish_if_stopped():
            return
        if not prerequisite(output / 'checkpoints/select.json'):
            from .paper_pilot_report import generate_report
            print(json.dumps(generate_report(root, output, config), indent=2), flush=True)
            return
        selection = read_json(output / 'checkpoints/select.json')['payload']
        atomic_json(output / 'pilot_selection.json', selection)
        original = yaml.safe_load((root / config['original_protocol']).read_text())
        tuning = original['dataset']['tuning_recording']
        specifications = []
        for anchor in config['cpd']['anchors']:
            for alpha in config['cpd']['alpha']:
                for beta in config['cpd']['beta']:
                    name = f'cpd_{anchor}_a{alpha:g}_b{beta:g}'
                    specifications.append({'name': name, 'method': 'cpd', 'anchor': anchor, 'order': len(specifications),
                                           'config': {'alpha': alpha, 'beta': beta, 'w': 0.0,
                                                      'max_iterations': 100, 'tolerance': 1e-5}})
        def task_for(phase, recording, stride, specification):
            return {**common, 'phase': phase, 'recording_id': recording, 'stride': stride,
                    'specification': specification,
                    'task_id': f'{phase}__{recording}__s{stride}__{specification["name"]}'}
        tune_tasks = [task_for('tune', tuning, stride, spec)
                      for stride in config['strides'] for spec in specifications]
        rows = supervisor.run(tune_tasks, workers=config['limits']['workers'])
        if finish_if_stopped():
            return
        selected = select_cpd(rows, specifications, config['strides'])
        atomic_json(output / 'cpd_selection.json', selected)
        candidates = [{'name': name, 'method': name, 'config': {}} for name in
                      ('previous_hungarian', 'pairwise_affine', 'first_affine', 'dual_anchor')]
        candidates += selected['selected']
        if validation.exists() and read_json(validation)['status'] == 'complete':
            payload = read_json(validation)['payload']
            if payload.get('passed', payload.get('status') == 'passed'):
                candidates.append({'name': 'fdnc_reference', 'method': 'fdnc', 'config': {}})
        candidates += read_json(root / config['pyrecest_selection'])['selected_candidates']
        atomic_json(output / 'candidates.json', {'candidates': candidates})
        tasks = [task_for('benchmark', recording, stride, spec) for recording in selection['selected_recordings']
                 for stride in config['strides'] for spec in candidates]
        supervisor.run(tasks, workers=config['limits']['workers'])
        if finish_if_stopped():
            return
        # Repeat the median-motion native-rate case in isolation from other pilot jobs.
        timing_tasks = [task_for('timing', selection['selected_recordings'][1], 1,
                                {**spec, 'name': spec['name'] + f'__repeat{repeat}'})
                        for spec in candidates for repeat in (1, 2)]
        supervisor.run(timing_tasks, workers=1)
    from .paper_pilot_report import generate_report
    print(json.dumps(generate_report(root, output, config), indent=2), flush=True)


if __name__ == '__main__':
    main()
