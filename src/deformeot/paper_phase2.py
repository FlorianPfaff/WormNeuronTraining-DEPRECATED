"""Opt-in version 1 paper-feasibility execution; does not alter frozen trackers.

All workers and the live coordinator are charged wall time. Immutable attempt
files preserve failures. Run this module on gpuserver6000, never on the laptop.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import yaml

from .paper_pilot import (atomic_json, digest, pilot_metrics, provenance, read_json,
                          run_tracking, select_cpd, sequence_for, storage_bytes, utc_now)
from .wormid_benchmark import ReferenceTrackerConfig
from .wormid_dandi import sha256_file

VERSION = 1
THREAD_ENV = {key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
              'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
FROZEN_SHA256 = {
    'configs/wormid_prospective.yaml': 'fdca2e18c61bcb3544106c4cd14e0945be789e6818ecd5e8904d9990c3b42bf0',
    'docs/prospective_expansion_protocol.md': 'b99c18526c104993aced172231607e0588c4e84e0b0cce6919d94af7407b56c3',
    'outputs/paper/source-results/prospective.json': 'c210e6fe2350fa1e2988d4ad20579b3afc76ae33a7a49e82a10ff27235945b3a',
    'configs/wormid_pyrecest.yaml': 'b0379dbf8f4f8552474720d6d694260187722b31c67832fe45dca7a183a64a00',
    'src/deformeot/wormid_pyrecest_gnn.py': '2edbffe77113e8a066f66fa6f576dc59d0f1fc0fb324790d56b2a919d2124af5',
    'src/deformeot/wormid_pyrecest_mht.py': 'e640ce4441926fbc53bf415dc511f467d2bce6b666c6d7b070c7c6d7c75cc2b7',
    'src/deformeot/wormid_pyrecest_experiment.py': 'fda050ff91dcf26eebc921ac8923b7df7fdd908a871c04241d364a1718b2159c',
}


def cpd_specs(config):
    """Order is part of the prespecified deterministic tie breaker."""
    specifications = [{'name': f'cpd_{anchor}_a{alpha:g}_b{beta:g}', 'method': 'cpd',
             'anchor': anchor, 'config': {'alpha': alpha, 'beta': beta,
             **{key: config['cpd'][key] for key in ('w', 'max_iterations', 'tolerance')}}}
            for anchor in config['cpd']['anchors']
            for alpha in config['cpd']['alpha'] for beta in config['cpd']['beta']]
    return [{**spec, 'order': order} for order, spec in enumerate(specifications)]


def mechanism_specs(root, config):
    selected = read_json(Path(root) / config['pyrecest_selection'])['selected_candidates']
    if isinstance(selected, dict):
        selected = list(selected.values())
    chosen = next(item for item in selected if item['name'] == 'mht_geometry_lag2')
    return [{'name': name, 'method': 'dual_anchor' if name == 'dual_anchor' else 'mechanism',
             'mode': name, 'config': dict(chosen['config'])}
            for name in config['mechanism']['modes']]


def dispatch(sequence, specification, root, config, geometry, trace=None):
    if specification['method'] == 'mechanism':
        from .paper_phase2_mechanism import track_mechanism
        from .wormid_pyrecest_mht import PyRecEstMHTConfig
        return track_mechanism(sequence, mode=specification['mode'],
                               config=PyRecEstMHTConfig(**specification['config']),
                               geometry=geometry, trace_callback=trace)
    return run_tracking(sequence, specification, root, config, geometry)


def save_assignments(path, sequence, result):
    """The evaluation channel is archived only after the tracker has returned."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.tmp').open('wb') as stream:
        np.savez_compressed(stream, assignments=np.asarray(result.assignments, dtype=np.int64),
                            truth_indices=np.asarray(sequence.truth_indices, dtype=np.int64),
                            frame_indices=np.asarray(sequence.source_frame_indices, dtype=np.int64),
                            frame_times=np.asarray([frame.time for frame in sequence.frames], dtype=np.float64))
    path.with_suffix('.tmp').replace(path)
    from .paper_phase2_rescore import rescore_file
    independent = rescore_file(path)['metrics']
    existing = pilot_metrics(sequence, result)
    for key in ('correct', 'total', 'identity_switches', 'identity_switch_opportunities',
                'wrong_identity_frames', 'wrong_identity_episode_count',
                'maximum_wrong_identity_duration_selected_scans'):
        if independent[key] != existing[key]:
            raise AssertionError(f'independent scoring disagrees on {key}: '
                                 f'{independent[key]} versus {existing[key]}')
    return {**existing, **independent, 'assignment_file': str(path),
            'assignment_file_sha256': sha256_file(path), 'independent_integer_agreement': True}


def execute_worker(task):
    started = time.monotonic()
    root, output = Path(task['root']), Path(task['output_dir'])
    config = yaml.safe_load((root / 'configs/paper_phase2.yaml').read_text())
    original = yaml.safe_load((root / config['original_protocol']).read_text())
    record = {'task_id': task['task_id'], 'task_signature': task['task_signature'],
              'source_signature': task['source_signature'], 'phase': task['phase'],
              'started_utc': utc_now(), 'host_load_start': list(os.getloadavg()),
              'host': socket.gethostname(), 'python': sys.version, 'thread_environment': THREAD_ENV}
    try:
        os.nice(10)
        phase = task['phase']
        if phase == 'tests':
            # A spawned pytest process belongs to this worker's process group;
            # its whole interval is already charged by the supervisor.
            done = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-o', 'addopts='],
                                  cwd=root, capture_output=True, text=True)
            log = output / f'{task["task_id"]}.pytest.log'
            log.write_text(done.stdout + done.stderr)
            if done.returncode:
                raise RuntimeError(f'tests failed (exit {done.returncode}); see {log}')
            payload = {'exit_code': done.returncode, 'log': str(log), 'summary': done.stdout[-3000:]}
        elif phase == 'evidence':
            from .paper_phase2_evidence import run_evidence
            payload = run_evidence(root, output / task['task_id'])
        elif phase == 'rescore':
            from .paper_phase2_rescore import rescore_file
            files = sorted((output / 'assignments').glob('*.npz'))
            payload = {'files': [], 'count': len(files)}
            for path in files:
                metrics = rescore_file(path)['metrics']
                payload['files'].append({'path': str(path.relative_to(output)),
                                         'sha256': sha256_file(path), 'metrics': metrics})
            atomic_json(output / 'assignment_manifest.json', {'files': payload['files']})
        elif phase == 'report':
            from .paper_phase2_report import make_report
            payload = make_report(root, output, config)
        elif phase == 'package':
            from .paper_phase2_package import build_package
            payload = build_package(root, output, task['cache_dir'])
        elif phase == 'verify_adapters':
            from .paper_pilot_baselines import validate_fdnc
            integrity = {name: sha256_file(root / name) for name in FROZEN_SHA256}
            if integrity != FROZEN_SHA256:
                raise RuntimeError('a frozen historical source or artifact has changed')
            validation_task = {**task, 'recording_id': original['dataset']['tuning_recording'], 'stride': 40}
            sequence, inputs = sequence_for(validation_task, original)
            sequence = replace(sequence, frames=sequence.frames[:12], truth_indices=sequence.truth_indices[:12],
                               source_frame_indices=sequence.source_frame_indices[:12])
            inputs['selected_frame_indices'] = list(sequence.source_frame_indices)
            geometry = ReferenceTrackerConfig(**original['association']['geometry'])
            fdnc = validate_fdnc(root / config['fdnc']['checkpoint'], root / config['fdnc']['source'], sequence)
            if fdnc.get('status') != 'passed':
                raise RuntimeError(f'released fDNC adapter validation failed: {fdnc}')
            checks = []
            specs = [{'name': name, 'method': name} for name in ('pairwise_affine', 'first_affine')]
            specs += [cpd_specs(config)[0], next(s for s in cpd_specs(config) if s['anchor'] == 'temporal')]
            specs += [{'name': 'fdnc_reference', 'method': 'fdnc'}, *mechanism_specs(root, config)]
            for spec in specs:
                results = [dispatch(sequence, spec, root, config, geometry) for _ in range(2)]
                if not np.array_equal(results[0].assignments, results[1].assignments):
                    raise AssertionError(f'non-deterministic assignments for {spec["name"]}')
                for repeat, result in enumerate(results, 1):
                    path = output / 'assignments' / f'{task["task_id"]}__{spec["name"]}__r{repeat}.npz'
                    metrics = save_assignments(path, sequence, result)
                    checks.append({'candidate': spec['name'], 'repeat': repeat, **inputs, **metrics})
            payload = {'fdnc_released_adapter_check': fdnc, 'repeat_checks': checks,
                       'deterministic_all_methods': True, 'frames': len(sequence.frames),
                       'frozen_artifacts_unchanged': True, 'frozen_artifact_hashes': integrity}
        else:
            sequence, inputs = sequence_for(task, original)
            geometry = ReferenceTrackerConfig(**original['association']['geometry'])
            diagnostics = []
            # Common boundary includes adapter initialization/imports, filtering,
            # and association, excluding cache load, scoring and artifact writing.
            clock_start = time.perf_counter()
            result = dispatch(sequence, task['specification'], root, config, geometry,
                              trace=None if phase == 'timing' else diagnostics.append)
            call_seconds = time.perf_counter() - clock_start
            path = output / 'assignments' / f'{task["task_id"]}.npz'
            metrics = save_assignments(path, sequence, result)
            spec = task['specification']
            future = 2 if spec.get('mode') in ('rolling_best_path', 'beam_best_path', 'beam_marginal') else 0
            payload = {'recording_id': task['recording_id'], 'stride': task['stride'],
                       'candidate': spec['name'], 'specification': spec, **inputs, **metrics,
                       'tracker_call_seconds': call_seconds,
                       'timing_boundary': 'full_adapter_call_including_initialization_excluding_input_loading_scoring_writing',
                       'future_selected_scans': future, 'maximum_future_source_frames': future * task['stride'],
                       'repeat': task.get('repeat'), 'lookahead_tail': 'shortened_at_sequence_end'}
            if diagnostics:
                atomic_json(output / 'diagnostics' / f'{task["task_id"]}.json', diagnostics)
        record.update(status='complete', payload=payload)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    record.update(wall_seconds=time.monotonic() - started, finished_utc=utc_now(),
                  host_load_end=list(os.getloadavg()))
    atomic_json(task['result_path'], record)
    print(json.dumps({key: record[key] for key in ('task_id', 'status', 'wall_seconds')}), flush=True)
    return 0 if record['status'] == 'complete' else 1


class Supervisor:
    """Single coordinator, continuously charged; no hidden free waiting time."""
    def __init__(self, root, output, storage_root, config, context):
        self.root, self.output, self.storage_root = map(Path, (root, output, storage_root))
        self.config, self.context = config, context
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / 'supervisor.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.output / 'budget.json'
        self.ledger = read_json(self.path) if self.path.exists() else {
            'schema_version': VERSION, 'created_utc': utc_now(), 'jobs': {},
            'charged_job_seconds': float(config['limits']['initial_inspection_charge_seconds']),
            'charged_mechanism_seconds': 0.0,
            'initial_charge_reason': 'conservative allowance for bounded server metadata inspection and transfer setup',
            'limits': config['limits'], 'repair_cycles': {}}
        if self.ledger.get('coordinator_running'):
            raise RuntimeError('stale coordinator ledger: explicit reconciliation is required; refusing to undercharge')
        previous_allowance = self.ledger['limits']['initial_inspection_charge_seconds']
        allowance = config['limits']['initial_inspection_charge_seconds']
        if allowance > previous_allowance:
            self.ledger['charged_job_seconds'] += allowance - previous_allowance
            self.ledger.setdefault('allowance_adjustments', []).append({
                'at': utc_now(), 'added_seconds': allowance - previous_allowance,
                'reason': 'conservative expanded allowance for setup, status inspections, and final synchronization'})
        self.ledger['limits'] = config['limits']
        for name in ('paper_phase2.py', 'paper_phase2_mechanism.py',
                     'paper_phase2_rescore.py', 'paper_phase2_evidence.py'):
            key = f'src/deformeot/{name}'
            versions = self.ledger.setdefault('adapter_source_versions', {}).setdefault(key, [])
            current = context['source_hashes'][key]
            if current not in versions:
                if len(versions) >= 3:
                    raise RuntimeError(f'initial implementation plus two repair revisions exhausted for {key}')
                versions.append(current)
        self.ledger['coordinator_running'] = True
        self.ledger['coordinator_pid'] = os.getpid()
        self.last_tick = time.monotonic()
        self.last_storage_check = 0.0
        self.live = {}
        self.mechanism_phase = False
        self.closed = False
        self.tick(force_storage=True)

    def tick(self, force_storage=False):
        now = time.monotonic()
        elapsed = now - self.last_tick
        self.last_tick = now
        # Charges the entire live interval even if a just-completed process was
        # observed one poll late. This is conservative, never an underestimate.
        self.ledger['charged_job_seconds'] += elapsed * (1 + len(self.live))
        self.ledger['charged_mechanism_seconds'] += elapsed * (
            int(self.mechanism_phase) + sum(item['mechanism'] for item in self.live.values()))
        self.ledger['updated_utc'] = utc_now()
        self.ledger['live_task_ids'] = list(self.live)
        if force_storage or now - self.last_storage_check > 15:
            used = storage_bytes(self.storage_root)
            self.ledger['storage_bytes'] = used
            self.ledger['peak_storage_bytes'] = max(used, self.ledger.get('peak_storage_bytes', 0))
            self.last_storage_check = now
        atomic_json(self.path, self.ledger)

    def remaining(self, mechanism=False):
        key = 'mechanism_job_seconds' if mechanism else 'cumulative_job_seconds'
        used = 'charged_mechanism_seconds' if mechanism else 'charged_job_seconds'
        return self.config['limits'][key] - self.ledger[used]

    def capacity(self, mechanism=False, finalizing=False):
        reserve = 30 * (len(self.live) + 2) + (0 if finalizing else 600)
        return (self.remaining() > reserve and (not mechanism or self.remaining(True) > reserve)
                and self.ledger['storage_bytes'] < self.config['limits']['additional_storage_bytes']
                    - self.config['limits']['storage_headroom_bytes'])

    def task(self, phase, *, suffix='', **kwargs):
        revision = self.context['signature'][:12]
        label = '__'.join(str(item) for item in (phase, kwargs.get('recording_id'),
                     f's{kwargs["stride"]}' if 'stride' in kwargs else None,
                     kwargs.get('specification', {}).get('name'), suffix) if item)
        identifier = label + '__' + revision
        task = {'task_id': identifier, 'phase': phase, 'root': str(self.root),
                'output_dir': str(self.output), 'source_signature': self.context['signature'],
                'result_path': str(self.output / 'checkpoints' / f'{identifier}.json'), **kwargs}
        task['task_signature'] = digest(task)
        return task

    def run(self, tasks, *, workers=None, mechanism=False):
        self.mechanism_phase = mechanism
        waiting = list(tasks)
        count = workers or self.config['limits']['workers']
        finalizing = all(task['phase'] in ('rescore', 'report', 'package') for task in tasks)
        while waiting or self.live:
            self.tick()
            if not self.capacity(mechanism, finalizing):
                reason = 'resource_limit_or_reserved_headroom'
                for identifier, item in list(self.live.items()):
                    self.stop(item, reason)
                    self.finish(identifier, item, reason)
                self.ledger.setdefault('stopped_batches', []).append(
                    {'at': utc_now(), 'reason': reason, 'pending': [t['task_id'] for t in waiting]})
                self.tick()
                break
            while waiting and len(self.live) < count:
                task = waiting.pop(0)
                path = Path(task['result_path'])
                if path.exists():
                    old = read_json(path)
                    if old['task_signature'] != task['task_signature']:
                        raise RuntimeError(f'checkpoint signature mismatch: {path}')
                    continue  # Immutable failures are not silently retried.
                identifier = task['task_id']
                old_versions = [job for name, job in self.ledger['jobs'].items()
                                if name.rsplit('__', 1)[0] == identifier.rsplit('__', 1)[0]]
                if sum(job.get('status') != 'complete' for job in old_versions) > 2:
                    raise RuntimeError(f'two repair cycles exhausted for {identifier}')
                task_file = self.output / 'tasks' / f'{identifier}.json'
                atomic_json(task_file, task)
                log_path = self.output / 'logs' / f'{identifier}.log'
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open('ab', buffering=0)
                command = [sys.executable, '-u', '-m', 'deformeot.paper_phase2', '--worker', str(task_file)]
                environment = {**os.environ, **THREAD_ENV, 'PYTHONPATH': str(self.root / 'src'),
                               'PYTHONHASHSEED': '0', 'PYTHONDONTWRITEBYTECODE': '1',
                               'TMPDIR': str(self.storage_root / 'tmp')}
                process = subprocess.Popen(command, cwd=self.root, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.live[identifier] = {'process': process, 'start': time.monotonic(),
                                         'task': task, 'log': log, 'mechanism': mechanism}
                self.ledger['jobs'][identifier] = {'status': 'running', 'pid': process.pid,
                    'started_utc': utc_now(), 'phase': task['phase'], 'mechanism': mechanism}
                print(f'start {identifier}', flush=True)
            for identifier, item in list(self.live.items()):
                elapsed = time.monotonic() - item['start']
                timeout = self.config['limits']['evidence_timeout_seconds'] if item['task']['phase'] == 'evidence' else self.config['limits']['task_timeout_seconds']
                if elapsed > timeout:
                    self.stop(item, 'task_timeout')
                    self.finish(identifier, item, 'task_timeout')
                elif item['process'].poll() is not None:
                    self.finish(identifier, item)
            if waiting or self.live:
                time.sleep(.5)
        self.tick()
        self.mechanism_phase = False
        return [read_json(t['result_path']) for t in tasks if Path(t['result_path']).exists()]

    def stop(self, item, reason):
        try:
            os.killpg(item['process'].pid, signal.SIGTERM)
            item['process'].wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(item['process'].pid, signal.SIGKILL)
            item['process'].wait(timeout=3)
        except ProcessLookupError:
            pass

    def finish(self, identifier, item, forced=None):
        self.tick()
        path = Path(item['task']['result_path'])
        if not path.exists():
            atomic_json(path, {'task_id': identifier, 'task_signature': item['task']['task_signature'],
                              'source_signature': self.context['signature'], 'phase': item['task']['phase'],
                              'status': 'failed', 'error': forced or 'worker exited without checkpoint'})
        row = read_json(path)
        self.ledger['jobs'][identifier].update(status=row['status'],
            wall_seconds=time.monotonic() - item['start'], finished_utc=utc_now(), forced_stop=forced)
        item['log'].close()
        del self.live[identifier]
        print(f'end {identifier} {row["status"]}', flush=True)

    def close(self):
        if self.closed:
            return
        for identifier, item in list(self.live.items()):
            self.stop(item, 'coordinator_finally')
            self.finish(identifier, item, 'coordinator_finally')
        self.tick(force_storage=True)
        self.ledger['coordinator_running'] = False
        atomic_json(self.path, self.ledger)
        self.closed = True
        self.lock.close()


def freeze_context(root, output, cache, config):
    context = provenance(root, config, cache)
    context.update(runner_version=VERSION, hostname=socket.gethostname(),
                   runtime_executable=sys.executable, cpu_count=os.cpu_count(),
                   runtime_reuse='existing pinned pilot environment, read-only; no dependency upgrades',
                   full_environment=sorted(f'{item.metadata["Name"]}=={item.version}'
                       for item in __import__('importlib.metadata', fromlist=['distributions']).distributions()))
    runtime_sources = {}
    for package in ('pyrecest', 'pycpd'):
        specification = importlib.util.find_spec(package)
        if specification and specification.origin:
            directory = Path(specification.origin).parent
            runtime_sources[package] = {str(path.relative_to(directory)): sha256_file(path)
                                       for path in sorted(directory.rglob('*.py'))}
    context['runtime_dependency_source_hashes'] = runtime_sources
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    context['numpy_build_and_simd_configuration'] = buffer.getvalue()
    context['thread_environment'] = {key: os.environ.get(key) for key in THREAD_ENV}
    for path in sorted((root / 'scripts').glob('*.py')):
        context['source_hashes'][str(path.relative_to(root))] = sha256_file(path)
    protocol = root / 'docs/paper_phase2_protocol.md'
    if protocol.exists():
        context['source_hashes']['docs/paper_phase2_protocol.md'] = sha256_file(protocol)
    context['signature'] = digest({key: value for key, value in context.items() if key != 'signature'})
    # The signature covers the scientific sources, config, cache and package
    # versions; the complete source snapshot is also independently hashed.
    signature = context['signature']
    path = output / 'contexts' / f'{signature}.json'
    if path.exists():
        saved = read_json(path)
        archive = output / 'contexts' / f'{signature}.source.tar.gz'
        if sha256_file(archive) != saved['source_archive_sha256']:
            raise RuntimeError('archived execution sources failed integrity verification')
        context = saved
    else:
        archive = output / 'contexts' / f'{signature}.source.tar.gz'
        archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, 'w:gz') as bundle:
            for folder in ('src', 'tests', 'configs', 'scripts'):
                bundle.add(root / folder, arcname=folder,
                           filter=lambda item: None if '__pycache__' in item.name or '.pytest_cache' in item.name else item)
            for name in ('pyproject.toml', 'requirements-paper-pilot-lock.txt'):
                bundle.add(root / name, arcname=name)
            if protocol.exists():
                bundle.add(protocol, arcname='docs/paper_phase2_protocol.md')
        context['source_archive_sha256'] = sha256_file(archive)
        atomic_json(path, context)
    atomic_json(output / 'active_context.json', context)
    return context


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--storage-root', type=Path)
    parser.add_argument('--phase', choices=('verify', 'all', 'report'), default='all')
    args = parser.parse_args(argv)
    if args.worker:
        return execute_worker(read_json(args.worker))
    if socket.gethostname() != 'workstation2':
        parser.error('This version is restricted to gpuserver6000 (workstation2).')
    if not all((args.output_dir, args.cache_dir, args.storage_root)):
        parser.error('--output-dir, --cache-dir and --storage-root are required')
    root, output = args.root.resolve(), args.output_dir.resolve()
    config = yaml.safe_load((root / 'configs/paper_phase2.yaml').read_text())
    original = yaml.safe_load((root / config['original_protocol']).read_text())
    output.mkdir(parents=True, exist_ok=True)
    (args.storage_root / 'tmp').mkdir(exist_ok=True)
    context = freeze_context(root, output, args.cache_dir, config)
    supervisor = Supervisor(root, output, args.storage_root, config, context)
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        common = {'cache_dir': str(args.cache_dir)}
        if args.phase != 'report':
            checks = supervisor.run([supervisor.task('tests', **common)], workers=1, mechanism=True)
            if not checks or any(row['status'] != 'complete' for row in checks):
                return 2
            checks = supervisor.run([supervisor.task('verify_adapters', **common)], workers=1, mechanism=True)
            if not checks or any(row['status'] != 'complete' for row in checks):
                return 2
        if args.phase == 'verify':
            return 0
        if args.phase == 'all':
            supervisor.run([supervisor.task('evidence', **common)], workers=1)
            specs = cpd_specs(config)
            tasks = [supervisor.task('tuning', recording_id=original['dataset']['tuning_recording'],
                      stride=stride, specification=spec, **common) for spec in specs for stride in config['cpd']['strides']]
            rows = supervisor.run(tasks)
            complete = [row['payload'] for row in rows if row['status'] == 'complete']
            if len(complete) != 72:
                raise RuntimeError('CPD grid incomplete; no winners frozen and no reduced-grid substitution')
            selected = select_cpd(rows, specs, config['cpd']['strides'])
            selected['source_signature'] = context['signature']
            selected['boundary_optima'] = [{
                'candidate': item['name'],
                'alpha_upper_boundary': item['config']['alpha'] == max(config['cpd']['alpha']),
                'beta_upper_boundary': item['config']['beta'] == max(config['cpd']['beta'])}
                for item in selected['selected']]
            selected['frozen_after_complete_72_case_grid'] = True
            atomic_json(output / 'cpd_selection.json', selected)
            winners = selected['selected']
            methods = [{'name': name, 'method': name} for name in ('pairwise_affine', 'first_affine', 'dual_anchor')]
            methods += [*winners, {'name': 'fdnc_reference', 'method': 'fdnc'}]
            animals = sorted([*original['dataset']['legacy_evaluation_recordings'], *original['dataset']['untouched_recordings']])
            tasks = [supervisor.task('benchmark', recording_id=animal, stride=1, specification=spec, **common)
                     for animal in animals for spec in methods]
            supervisor.run(tasks)
            mechanisms = mechanism_specs(root, config)
            tasks = [supervisor.task('mechanism', recording_id=animal, stride=stride, specification=spec, **common)
                     for animal in config['mechanism']['recordings'] for stride in config['mechanism']['strides']
                     for spec in mechanisms]
            supervisor.run(tasks, mechanism=True)
            # No evidence downloads or other phase workers run during timing.
            tasks = [supervisor.task('timing', recording_id=config['mechanism']['median_motion_recording'],
                     stride=1, specification=spec, repeat=repeat, suffix=f'r{repeat}', **common)
                     for spec in mechanisms if spec['name'] in config['mechanism']['timing_modes']
                     for repeat in (1, 2)]
            supervisor.run(tasks, workers=1, mechanism=True)
        supervisor.run([supervisor.task('rescore', **common)], workers=1)
        supervisor.run([supervisor.task('report', **common)], workers=1)
        supervisor.run([supervisor.task('package', **common)], workers=1)
        return 0
    except Exception:
        atomic_json(output / f'coordinator_failure_{int(time.time())}.json', {'at': utc_now(), 'traceback': traceback.format_exc()})
        if args.phase != 'verify':
            supervisor.run([supervisor.task('rescore', **common)], workers=1)
            supervisor.run([supervisor.task('report', **common)], workers=1)
            supervisor.run([supervisor.task('package', **common)], workers=1)
        raise
    finally:
        supervisor.close()


if __name__ == '__main__':
    raise SystemExit(main())
