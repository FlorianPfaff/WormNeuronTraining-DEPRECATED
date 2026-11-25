"""Resume the frozen evaluation as a detached job in persistent server storage."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from deformeot.wormid_pyrecest_experiment import load_experiment_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    protocol = Path("configs/wormid_pyrecest.yaml").resolve()
    context = load_experiment_context(Path.cwd(), protocol)
    original = json.loads((output / "experiment_context.json").read_text())
    selection_bytes = (output / "selection.json").read_bytes()
    selection = json.loads(selection_bytes)
    signature = context["implementation_signature"]
    if signature != original["implementation_signature"] or signature != selection["implementation_signature"]:
        raise ValueError("recovery must exactly preserve the frozen implementation and selection")
    command = [sys.executable, "-u", "-m", "deformeot.wormid_pyrecest_experiment",
               "--protocol", str(protocol), "--cache-dir", str(args.cache_dir.resolve()),
               "--output-dir", str(output), "--phase", "evaluate", "--workers", str(args.workers)]
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve()),
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with (output / "recovery_run.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=environment, start_new_session=True)
    audit = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "pid": process.pid,
        "persistent_project": str(Path.cwd()), "command": command,
        "implementation_signature": signature,
        "preserved_selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "reason": "Server reboot removed the earlier temporary evaluation directory; saved tuning and control results were recovered locally.",
        "parameter_selection_repeated": False, "scientific_configuration_changed": False,
        "operational_worker_override": {
            "protocol_provenance_workers": 12, "recovery_workers": args.workers,
            "reason": "Increase independent-task parallelism on the 64-logical-CPU server; each numerical worker remains single-threaded.",
        },
    }
    (output / "recovery.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
