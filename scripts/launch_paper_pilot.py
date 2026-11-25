"""Launch the bounded pilot in persistent storage, surviving SSH disconnects."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--cache-dir', required=True)
parser.add_argument('--storage-root', type=Path, required=True)
parser.add_argument('--historical-python', required=True)
args = parser.parse_args()
root = Path.cwd().resolve()
output = root / 'work/results/paper_pilot_20261003'
output.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-u', '-m', 'deformeot.paper_pilot', '--phase', 'all',
           '--root', str(root), '--cache-dir', args.cache_dir, '--output-dir', str(output),
           '--storage-root', str(args.storage_root), '--historical-python', args.historical_python]
environment = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'OPENBLAS_NUM_THREADS': '1',
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
with (output / 'run.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, env=environment)
record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'pid': process.pid,
          'command': command, 'persistent_storage': str(args.storage_root),
          'limits': {'cumulative_job_hours': 24, 'additional_storage_GiB': 10, 'workers': 4}}
(output / 'launch.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
