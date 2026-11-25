"""Verify a reporting-only revision without changing completed numerical evidence.

Run on the calculation server after the supervisor exits. A fixed two-minute
reservation bounds and conservatively accounts for verification and rendering.
The original execution source archive and context are retained unchanged.
"""
import argparse
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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def stored_bytes(root):
    seen, total = set(), 0
    for directory, _, files in os.walk(root, followlinks=False):
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            key = (stat.st_dev, stat.st_ino)
            if key not in seen:
                seen.add(key)
                total += stat.st_size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    args = parser.parse_args()
    root, output = Path.cwd().resolve(), args.output_dir.resolve()
    with (output / 'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        budget_path = output / 'budget.json'
        budget = json.loads(budget_path.read_text())
        if any(row.get('status') == 'running' for row in budget['jobs'].values()):
            raise RuntimeError('Numerical jobs require reconciliation before finalization')
        context = json.loads((output / 'context.json').read_text())
        if platform.python_version() != context['python'] or platform.platform() != context['platform']:
            raise RuntimeError('Finalization must use the recorded execution runtime and host')
        for name, expected in context['packages'].items():
            try:
                observed = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                observed = None
            if observed != expected:
                raise RuntimeError(f'Finalization package differs: {name}')
        renderer = 'src/deformeot/paper_pilot_report.py'
        unexpected = [name for name, expected in context['source_hashes'].items()
                      if name != renderer and sha(root / name) != expected]
        if unexpected:
            raise RuntimeError(f'Non-reporting numerical inputs changed: {unexpected}')
        archive = output / 'execution_source.tar.gz'
        if not archive.is_file():
            raise RuntimeError('Preserve the original execution source archive first')
        reservation = 120.0
        overhead = None
        if not any(row.get('name') == 'pilot_supervision_overhead' for row in budget.get('external_jobs', [])):
            launch = json.loads((output / 'launch.json').read_text())
            launched = datetime.fromisoformat(launch['started_utc'])
            overhead = {'name': 'pilot_supervision_overhead',
                        'wall_seconds': max(0., (datetime.now(timezone.utc) - launched).total_seconds()),
                        'measurement': 'conservative_launch_to_finalizer_wall_time_in_addition_to_worker_jobs',
                        'note': 'Includes supervisor CPU/storage polling and orchestration; not a CPU-time measurement.'}
        additional = reservation + (overhead['wall_seconds'] if overhead else 0.)
        size = stored_bytes(args.storage_root)
        if budget['completed_job_seconds'] + additional >= budget['limit_job_seconds']:
            raise RuntimeError('No finalization reservation fits within the time limit')
        if size >= budget['limit_storage_bytes']:
            raise RuntimeError('Additional storage limit reached')
        entry = {'name': 'reporting_revision_verification', 'wall_seconds': reservation,
                 'measurement': 'conservative_two_minute_reservation_with_enforced_timeout',
                 'started_utc': datetime.now(timezone.utc).isoformat()}
        if overhead:
            budget.setdefault('external_jobs', []).append(overhead)
        budget.setdefault('external_jobs', []).append(entry)
        budget['completed_job_seconds'] += additional
        budget['storage_bytes'] = size
        save(budget_path, budget)
        started = time.monotonic()
        temporary_root = args.storage_root.resolve() / 'finalization-tmp'
        temporary_root.mkdir(exist_ok=True)
        environment = {**os.environ, 'PYTHONPATH': str(root / 'src'),
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                       'TMPDIR': str(temporary_root)}

        def run(command, log_name):
            with (output / log_name).open('w') as stream:
                process = subprocess.Popen(command, env=environment, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    while process.poll() is None:
                        if time.monotonic() - started >= reservation - 2:
                            raise TimeoutError('Finalization time reservation exhausted')
                        if stored_bytes(args.storage_root) >= budget['limit_storage_bytes']:
                            raise RuntimeError('Finalization storage limit reached')
                        time.sleep(.25)
                    code = process.returncode
                finally:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
                if code:
                    raise RuntimeError(f'{command} failed ({code}); see {log_name}')

        record = {'execution_signature': context['signature'],
                  'execution_source_archive_sha256': sha(archive),
                  'original_renderer_sha256': context['source_hashes'][renderer],
                  'final_renderer_sha256': sha(root / renderer),
                  'new_reporting_tests_sha256': sha(root / 'tests/test_paper_pilot_report.py'),
                  'finalization_script_sha256': sha(Path(__file__).resolve()),
                  'numerical_source_inputs_unchanged': True,
                  'reservation_seconds': reservation, 'status': 'running'}
        try:
            run([sys.executable, '-m', 'pytest', '-q', '-o', 'addopts='], 'final_verification.log')
            record['status'] = 'passed'
            save(output / 'final_verification.json', record)
            run([sys.executable, '-m', 'deformeot.paper_pilot', '--phase', 'report',
                 '--root', str(root), '--output-dir', str(output)], 'final_report.log')
            changed = [name for name, expected in context['source_hashes'].items()
                       if name != renderer and sha(root / name) != expected]
            if changed or sha(archive) != record['execution_source_archive_sha256']:
                raise RuntimeError(f'Execution evidence changed during finalization: {changed}')
            if json.loads((output / 'context.json').read_text()) != context:
                raise RuntimeError('Execution context changed during finalization')
        except BaseException as error:
            record.update(status='failed', error=str(error))
            raise
        finally:
            record['measured_wall_seconds'] = time.monotonic() - started
            entry['measured_wall_seconds'] = record['measured_wall_seconds']
            entry['status'] = record['status']
            budget['storage_bytes'] = stored_bytes(args.storage_root)
            budget['updated_utc'] = datetime.now(timezone.utc).isoformat()
            save(output / 'final_verification.json', record)
            save(budget_path, budget)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
