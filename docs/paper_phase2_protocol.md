# Version 1: bounded paper-feasibility study

This is an opt-in extension. The original trackers, their defaults, the July
artifacts and the first development pilot are not modified. None of the 20
evaluation recordings is an untouched external test animal.

## Frozen questions

1. Does dual anchoring retain its practical advantage over reference CPD after
   the one permitted expansion to 72 tuning cases? Reference CPD is the primary
   comparator regardless of the observed rankings of the six methods.
2. Does the existing marginal-commitment MHT beat an information-matched rolling
   best-path search, or can lookahead, bounded search, and commitment policy
   explain the earlier result? The rolling control reconsiders earlier
   assignments within an admitted top-three tree; it is not a delayed greedy
   result or a globally exact tracker.
3. Is a distinctive contribution and a credible public external-validation
   route supported? Library integration, MHT, reference anchoring and
   deformation correction alone are not novelty claims.

The machine-readable specification is `configs/paper_phase2.yaml`. Its fixed
bootstrap seed is 20261004, with 20,000 paired animal resamples. Acquisition-date
sensitivity is separate from the equal-animal primary analysis. Initialization
is included to match the pilot, with excluding-initialization sensitivity.
Coverage means assignments within the supplied initialized, exhaustive ROI
cohort, not detector recall or coverage of all biological neurons.

## Execution and safety

Run calculations and tests only on gpuserver6000, using the recorded pinned
Python environment. The runner checks the server hostname. The supervisor
charges its own complete wall interval plus each live worker interval, including
verification, metadata inspection, timing and reporting. Parallel worker times
are summed, not treated as a single elapsed interval. A 600-second conservative
allowance covers transfer setup, source freezing, status inspections and final
artifact synchronization outside the supervisor.
The limits are 24 cumulative job-hours, 12 mechanism-package job-hours, and
10 GiB additional storage. A finalization reserve and storage headroom stop
scientific work early. Existing cached coordinates and the existing pinned
runtime are read-only; they are not new downloads or newly installed storage.

Failed attempts and their logs remain immutable. Each new adapter has at most
three archived source revisions: its initial implementation and two repairs.
An incomplete control remains inconclusive. Do not replace failed controls,
expand the tuning grid again, or score external outcomes in this phase.

Example, from a separately staged project directory on gpuserver6000:

```bash
PINNED_PYTHON scripts/launch_paper_phase2.py --phase all \
  --cache-dir EXISTING_COORDINATE_CACHE --storage-root NEW_PHASE2_DIRECTORY
```

`--phase verify` runs the test suite, released fDNC adapter fidelity checks, and
repeated deterministic assignments. `--phase report` independently rescores
saved arrays and regenerates the report without rerunning scientific cases.
Do not launch another coordinator while the recorded coordinator is alive.
Crash recovery requires explicit, conservative reconciliation, not silently
discarding unaccounted time.

## Saved evidence and independent scoring

Each scientific case saves an NPZ containing assignments, anonymous-row to
canonical-ROI mappings, original frame indices, and physical timestamps. The
JSON checkpoint records the input hash, seed, selected frames, model and source
context, method settings, runtime/load, and maximum future-scan access.
Scoring occurs after tracking. The independent scorer imports no tracker or
historical evaluator and checks exact integer correctness and switch counts.

```bash
PINNED_PYTHON -m deformeot.paper_phase2_rescore RESULT_DIRECTORY/assignments \
  --manifest RESULT_DIRECTORY/assignment_manifest.json --output rescored.json
```

Wrong-identity episodes count consecutive wrong observations, even when the
wrong biological identity changes inside a run. Elapsed-time measures describe
observed spans or right-endpoint-weighted exposure; they do not infer biological
error onset or invent observations beyond a sparse sequence's tail.

Timing uses two serial, native-rate runs for each of four methods on the frozen
median-motion animal. A common timer wraps the full adapter call, including
initialization but excluding input loading, scoring and artifact writing.
Server load is recorded; shared-server timings are descriptive, not dedicated
hardware performance claims.

The original July discrepancy remains explicitly unresolved. The one bounded
archive search is closed in `configs/paper_phase2_history.yaml`; no new effect
size is justified using unavailable July assignment arrays.

## Decision limits

A methods route requires a distinct contribution, competitive advantage,
reproducible evidence and a credible external-validation route. A complete
control package alone is not a substantive benchmark paper. A positive screen
only supports proposing a subsequent frozen validation study; it never means
the work is already paper-ready. No broad neuroscience-utility, calibrated
uncertainty or extended-object-tracking claims are made.
