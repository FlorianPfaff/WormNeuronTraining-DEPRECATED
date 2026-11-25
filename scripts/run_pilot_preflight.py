"""Run server-only pilot verification and preserve measured setup/check time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--attempt', type=int, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
environment = {**os.environ, 'PYTHONPATH': str(Path('src').resolve()), 'OPENBLAS_NUM_THREADS': '1',
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
started = time.monotonic()
result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True, env=environment)
elapsed = time.monotonic() - started
log = args.output_dir / f'preflight_{args.attempt}.log'
log.write_text(result.stdout + result.stderr)
payload = {'name': f'preflight_{args.attempt}', 'wall_seconds': elapsed, 'exit_code': result.returncode, 'log': str(log)}
(args.output_dir / f'preflight_{args.attempt}.json').write_text(json.dumps(payload, indent=2) + '\n')
print(result.stdout + result.stderr)
print(json.dumps(payload), flush=True)
sys.exit(result.returncode)
