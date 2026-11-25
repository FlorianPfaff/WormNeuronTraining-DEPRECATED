"""Check GNN objective-equivalence against rerun baselines, without tuning."""

import argparse
import json
from pathlib import Path

from deformeot.wormid_pyrecest_experiment import (
    _atomic_json,
    _execute_tasks,
    load_experiment_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    context = load_experiment_context(Path.cwd(), args.protocol)
    control = next(candidate for candidate in context["candidates"] if candidate.get("control"))
    tasks = [{"context": context, "phase": "control_audit", "recording_id": recording,
              "cohort": "previously_evaluated_control", "cache_dir": str(args.cache_dir.resolve()),
              "output_dir": str(args.output_dir.resolve()), "specifications": [control]}
             for recording in sorted((*context["legacy_recordings"], *context["previous_prospective_recordings"]))]
    rows = _execute_tasks(tasks, args.workers, progress=None)
    _atomic_json(args.output_dir / "control_audit.json", {
        "implementation_signature": context["implementation_signature"],
        "selection_uses_control_outcomes": False,
        "results": rows,
    })
    print(json.dumps({"control_tasks": len(rows), "failed": sum(row["status"] != "complete" for row in rows)}))


if __name__ == "__main__":
    main()
