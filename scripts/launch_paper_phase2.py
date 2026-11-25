"""Server-only detached phase-2 launch; all numerical work is supervised."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--cache-dir', required=True)
parser.add_argument('--storage-root', type=Path, required=True)
parser.add_argument('--phase', choices=('verify', 'all', 'report'), default='all')
args = parser.parse_args()
if socket.gethostname() != 'workstation2':
    parser.error('Use gpuserver6000; this launcher does not run on the laptop.')
root = Path.cwd().resolve()
output = root / 'work/results/paper_phase2_20261004'
output.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-u', '-m', 'deformeot.paper_phase2', '--phase', args.phase,
           '--root', str(root), '--output-dir', str(output), '--cache-dir', args.cache_dir,
           '--storage-root', str(args.storage_root)]
environment = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'PYTHONHASHSEED': '0',
               'PYTHONDONTWRITEBYTECODE': '1', **{name: '1' for name in (
               'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
with (output / f'launch_{stamp}.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True, env=environment)
record = {'started_utc': stamp, 'pid': process.pid, 'command': command,
          'limits': {'cumulative_job_hours': 24, 'mechanism_job_hours': 12,
                     'additional_storage_GiB': 10}, 'server': 'gpuserver6000'}
(output / f'launch_{stamp}.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
