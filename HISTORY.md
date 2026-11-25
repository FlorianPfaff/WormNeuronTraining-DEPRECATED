> Historical project notes, retained from before the six-animal external comparison. See the [closure decision](README.md#why-we-stopped) for the final interpretation.

# Reproducible artifact for dual-anchor neuron tracking

This repository contains an outcome-untouched, frozen-protocol replication of
identity-preserving neuron tracking through deforming whole-brain microscopy.
After the tracker, comparator, parameters, five temporal strides, 16-recording
cohort, seeds, aggregation, bootstrap, and decision rule were frozen, all 16
remaining structurally eligible adult *C. elegans* EY recordings in immutable
DANDI version `0.241009.1457` were evaluated with no outcome-based exclusions.
The coordinate-only dual-anchor tracker improved equal-recording seeded identity
accuracy by **1.824 percentage points** over direct first-frame affine
registration (95% paired recording-bootstrap interval 1.245 to 2.510). The
frozen replication criterion, an interval lower endpoint above zero, was met;
15 of 16 recording effects were positive.

Across the same untouched cohort and five strides, pooled identity switches
fell from 257,830 to 79,920 (**69.0%**). The exact top-quartile label-free
high-motion analysis gained 1.498 identity-accuracy points (95% interval 1.075
to 1.963) and reduced pooled switches by 59.3%. Equal-date hierarchical
resampling across six acquisition dates gave a 2.230-point accuracy effect (95%
interval 1.149 to 3.583), and excluding the sole untouched recording acquired on
the tuning date gave 1.884 points (95% interval 1.271 to 2.598).

This was an internally frozen before-outcome protocol, not an externally
registered or third-party-timestamped preregistration. Direct first-frame affine
was locked as the primary comparator because it had been selected on the four
earlier evaluation recordings; the untouched result is therefore a replication
of that fixed comparison, not an untouched comparator search.

In the earlier four-recording split evaluation, a 15-dimensional local GCaMP
patch cue improved identity accuracy in a tuning-Q25 dense stratum by **2.260
points** (95% descriptive interval 1.039 to 4.220). The cue combines
image-centroid offsets with covariance/axis/scale features and should not be
interpreted as an extent-only or intrinsic-morphology effect. Moreover, the Q25
stratum differed from the radius-based wording in the internally frozen
analysis plan, making this supportive evidence under a protocol deviation
rather than a passed confirmatory endpoint.

The validated implementation is deterministic dual anchoring with an optional
local patch descriptor. It is not a validated Bayesian multi-hypothesis
posterior. The core benchmark is association-only: it uses exhaustive supplied
centers, a fixed neuron cohort, first-frame identity initialization, and no
missing/false/merge/split observations. The primary expansion exhausts every
eligible recording not already assigned to tuning or legacy evaluation in the
21-asset DANDI version, but all recordings still come from one device-restricted
or semi-immobilized source collection. The result does not establish
cross-laboratory or general freely moving performance.

## PyRecEst follow-up comparison

The October 2026 follow-up integrates native PyRecEst global nearest-neighbor
association, Gaussian filtering, and multi-hypothesis tracking with the same
shared affine transport and first-frame reference. The candidate grid is in
`configs/wormid_pyrecest.yaml`; code/configuration hashes are recorded in
`configs/wormid_pyrecest.freeze.sha256`. Settings are selected using only the
original tuning animal. Online filtering and two-selected-scan MHT are evaluated
separately on the other 20 recordings at all five original strides.

These 20 recordings were already examined in the earlier study, so this is a
follow-up comparison. Positive MHT lag uses additional future observations.
The zero-lag MHT adapter commits each scan immediately, so its gains test
recursive filtering/scoring rather than persistent alternative histories.

Completed results: GNN increased mean identity accuracy from 92.514% to
92.695% (+0.180 percentage points; recording-bootstrap 95% interval
+0.084 to +0.278), with 1.6% more identity switches and 7.5x runtime. Lag-2
MHT reached 92.758% (+0.244 points), with 3.7% more switches, 54.9x runtime,
and two future selected scans. The online Gaussian-filtering adapter gained
0.171 points, with 1.9% more switches and 17.4x runtime. These are small
accuracy gains, not an across-the-board improvement; the current dual-anchor
tracker remains the better default when speed and identity continuity matter.
See the [complete report](outputs/pyrecest_followup_20261002/report.md) and
[comparison table](outputs/pyrecest_followup_20261002/comparison.csv).
All 500 evaluation cases completed on `gpuserver6000`; 103 tests passed.
Comparisons use freshly rerun baselines. Historical results were not exactly
reproduced (dual-anchor mean drift -0.021 percentage points); the report
quantifies the discrepancies, and the original results remain unchanged.

Use Python 3.11 or later, install `.[pyrecest]`, and clone
`work/vendor/PyRecEst` at commit
`7eff44c5f4d62c54a8394270fc924cb1302f171b`. Run the calculations on an SSH server:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m deformeot.wormid_pyrecest_experiment \
  --protocol configs/wormid_pyrecest.yaml \
  --cache-dir /path/to/wormid_ey_geometry \
  --output-dir work/results/pyrecest_followup_20261002 --phase all --workers 12
```

The evaluator saves the tuning selection, per-task checkpoints, source/data
hashes, paired recording-bootstrap intervals, switches, runtime, lookahead,
and comparison with the canonical baseline outcomes. Resuming requires matching
code, configuration, package versions, and data provenance. The original paper
artifact and source-result JSON are preserved.

## Bounded paper-feasibility pilot

The first implementation of the [paper-readiness plan](docs/paper_readiness_plan.md)
is a separate exploratory pilot, configured in `configs/paper_pilot.yaml`.
It does not implement the later independent-confirmation or downstream-utility
stages, and it never reclassifies previously examined animals as untouched.

The pilot audits six representative numerical cases with three repetitions in
each of two environments; validates the released learned fDNC inference against
its official source; and tunes nonrigid CPD/Hungarian reference and temporal
baselines on the original tuning animal only. Three other, already-examined
recordings are selected by minimum, lower-median, and maximum identity-blind
native-rate motion. Ten methods are compared at strides 1, 10, and 40, followed
by two serial native-rate timing repetitions on the median-motion recording.
The original geometric and PyRecEst implementations remain frozen.

Limits are 24 cumulative server job-hours, 10 GiB of additional storage, four
single-threaded workers, and three hours per task. The supervisor saves exact
code/data/model/environment signatures, per-task results, logs, and a locked
resource ledger. Public metadata and compact annotations from three candidate
external source families are reviewed without scoring external identities,
training models, or downloading large raw movies. Failures remain visible.

Run every numerical phase, including tests and report aggregation, on an SSH
calculation server. The observed Python 3.12.3 CPU environment is pinned in
`requirements-paper-pilot-lock.txt`; the historical reconstruction uses Python
3.10.20 and the original NumPy/SciPy/pandas pins, not the exact original 3.10.12
WSL host. Set up isolated environments and verify all cached input hashes first.
From the server-side project directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /path/to/current-runtime/bin/python -m deformeot.paper_pilot \
  --phase all --root "$PWD" \
  --cache-dir /path/to/existing/wormid-coordinate-cache \
  --output-dir work/results/paper_pilot_20261003 \
  --storage-root /path/to/dedicated/pilot-root \
  --historical-python /path/to/historical-runtime/bin/python
```

The dedicated storage root must include the added environments, dependencies,
and results, not just the output directory; an existing read-only coordinate
cache may remain outside it. Save measured setup/preflight charges as JSON files
with `wall_seconds` under its `accounting/` directory before the first launch.
`scripts/run_pilot_preflight.py` records a server test run, and
`scripts/launch_paper_pilot.py` launches a persistent detached run. Matching
completed checkpoints are reused; provenance changes are rejected. A hard-crash
resume requires inspection of recorded processes before using
`--reconcile-interrupted`, which conservatively charges a full timeout for each
dead unobserved job. Do not delete the ledger to evade a resource stop.

Outputs are retained under `work/results/paper_pilot_20261003/` and summarized in
[the feasibility report](outputs/paper_pilot_20261003/report.md). The native-rate
25% switch-reduction / 0.25-point accuracy-loss screen is descriptive, not a
significance test on three purposively selected animals. Reproduction of adapter
arithmetic does not establish training-data independence or biological label
validity; those limitations must remain in any publication claim.

Completed on `gpuserver6000`: all 36 CPD tuning cases, 90 benchmark cases,
and 20 serial timing repeats; the final verification suite passed 160 tests.
At native rate, equal-recording accuracy was 92.316% for dual anchoring versus
89.341% for the development-selected strongest new baseline, reference CPD:
**+2.976 percentage points and a 76.6% lower mean identity-switch rate**.
The provisional practical screen passed, but the decision is **conditional go**,
not publication readiness: no independent external multi-animal label cohort
has been fully verified, and the original historical numerical discrepancy
remains unexplained. Both reconstructed environments were deterministic and
agreed in all six audit cases; only the two matching controls reproduced the
original historical scores. The pilot conservatively accounted for 2.418
server job-hours and 1.695 GiB of added storage. All calculations and tests ran
on the server; the three selected animals remain development evidence.

## Earlier internally frozen target ledger

| Analysis-plan target | Status | Frozen evidence |
| --- | --- | --- |
| Severe/long-gap identity: +3 identity-accuracy points or at least 25% fewer switches | **Numeric criterion met** | In the exact top-quartile label-free cloud-motion stratum, dual anchors reduced switches by 40.2-71.6% across strides 1/5/10/20/40 and gained 1.624 identity-accuracy points (95% descriptive interval 0.925-2.458). |
| Local extent: +2 identity-accuracy points in radius-defined dense scenes | **Protocol deviation** | The tuning-Q25 stratum gained 2.260 points (95% descriptive interval 1.039-4.220), but the planned radius definition was not executed literally; radius-proxy gains were positive but below 2 points. |
| At 80% coverage: at least 99.5% precision and ECE below 0.05 | **Failed** | Precision was 98.459%; fixed/adaptive ECE was 0.0167/0.0116. |
| Gap recovery: +10 points without a localization penalty | **Inconclusive** | The single Targettrack movie had sparse, method-dependent gap-event denominators. WormID has no missing detections. |
| Raw trace correlation +0.03 or behavior decoding | **Failed** | The planned raw median Pearson gain was 0.0071 (95% descriptive interval 0.0022-0.0123); no behavior decoder was run. |

The original, broader paper-worthiness gate in the earlier internally frozen
plan remains unmet: the Bayesian model and its required comparator/ablation set
were not implemented. The later 16-recording protocol closes the
outcome-untouched replication gap for the fixed center-only dual-anchor claim,
but it is still a same-source replication rather than an independent laboratory
or acquisition-system validation. Post-hoc raw-mean, first-difference,
derivative-event, no-intensity-gate, and density-sensitivity analyses do not
rescue failed or deviated earlier endpoints.

## Repository map

- `src/deformeot/`: tracking, extent, confidence, downstream, Targettrack, fDNC,
  and evaluator code.
- `tests/`: unit and integration tests, including row-permutation and
  intensity-invariance checks.
- `configs/data_manifest.yaml`: frozen input paths, byte sizes, roles, and SHA-256
  identities.
- `docs/preregistration.md`: the internally frozen analysis plan; no external
  timestamped registration is claimed.
- `docs/prospective_expansion_protocol.md` and
  `configs/wormid_prospective.freeze.sha256`: the before-outcome protocol and
  integrity record for the 16-recording expansion.
- `docs/results_audit.md`: complete claim, failure, counterexample, and scope
  audit.
- `requirements-lock.txt`: direct and verification dependency pins observed for
  the paper artifact; this is not a fully resolved transitive lock.
- `work/results/`: canonical machine-generated compact result JSON used as
  figure inputs and included at the same paths in the reproduction bundle.
- `outputs/paper/`: paper figures, tables, copied source results, manuscript, and
  the self-contained reproducibility guide.

Raw public data are not committed. See `outputs/paper/reproducibility.md` for
official acquisition locations, every required file hash, and expected outputs.

## Environment

Python 3.10 or later is required. The paper artifact used Python 3.10.12 and the
direct dependency pins in `requirements-lock.txt`. Transitive resolution can
vary by Python version and platform; `outputs/paper/environment.txt` records the
key versions observed in the verified build.

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
```

The commands below use the frozen machine-local data root. Change only this
variable if the verified files are stored elsewhere.

```bash
export WHA_DATA=/mnt/d/CodexData/wha_neuron_data
```

## Reproduce the frozen untouched-cohort geometry replication

First verify that the before-outcome protocol and configuration have not
changed. The runner repeats this check before reading its configuration:

```bash
sha256sum -c configs/wormid_prospective.freeze.sha256
```

The runner queries the immutable 21-asset DANDI version, extracts only the
trajectory tables through HTTP byte ranges, caches the compact coordinate
arrays outside the repository, and evaluates the four frozen center-only
methods on all 20 non-tuning recordings. No multi-gigabyte image download is
needed for this geometry-only analysis.

```bash
python -m deformeot.wormid_prospective \
  --repository-root . \
  --cache-dir "$WHA_DATA/wormid_ey_geometry" \
  --output work/results/wormid_prospective_expansion.json \
  --workers 4
```

The primary cohort is the exact 16-recording set never used for tuning or the
first split evaluation. `first_affine`, the five strides, parameters, full-asset
seed indices, recording-level bootstrap, acquisition-date sensitivity, and
decision rule are locked in the verified configuration.

The supplied result must have SHA-256
`c210e6fe2350fa1e2988d4ad20579b3afc76ae33a7a49e82a10ff27235945b3a`.
Its primary effect must be 0.01824306499 seeded identity accuracy with 95%
interval 0.01244507215 to 0.02510429042 and
`replication_criterion_met: true`.

## Reproduce the WormID reference result

Tune the reference weight on the declared tuning animal, run all five recordings
with that weight frozen, then summarize only the other four recordings:

```bash
python -m deformeot.wormid_tune \
  "$WHA_DATA/wormid_ey/sub-20190925-04_ses-20190925_ophys.nwb" \
  --weights 0.5 1 2 3 4 5 6 8 12 16 \
  --strides 5 10 20 40 \
  --output work/results/wormid_reference_tuning.json \
  --seed 20260720

python -m deformeot.wormid_benchmark \
  "$WHA_DATA/wormid_ey/sub-20190925-04_ses-20190925_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-02_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-05_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-07_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20191030-07_ses-20191030_ophys.nwb" \
  --strides 1 5 10 20 40 \
  --reference-weight 6 \
  --descriptor-weight 0 \
  --tuning-recording-id sub-20190925-04_ses-20190925_ophys \
  --output work/results/wormid_reference_benchmark.json \
  --seed 20260720

python -m deformeot.wormid_report \
  work/results/wormid_reference_benchmark.json \
  --output work/results/wormid_reference_summary.json \
  --samples 20000 \
  --seed 20260720
```

## Reproduce extent, confidence, and downstream results

The unified patch-cue command runs the 8 by 7 tuning grid on the tuning animal,
the frozen four-recording split-evaluation comparison, the
unchanged-configuration no-intensity-gate robustness arm, and paired
recording-level bootstraps. It reuses the checked feature caches by default; add
`--regenerate-features` to stream the raw movies and replace them. The CLI/module
name retains `extent` for compatibility.

```bash
python -m deformeot.wormid_extent \
  --calibration "$WHA_DATA/wormid_ey/sub-20190925-04_ses-20190925_ophys.nwb" \
  --held-out \
    "$WHA_DATA/wormid_ey/sub-20190929-02_ses-20190929_ophys.nwb" \
    "$WHA_DATA/wormid_ey/sub-20190929-05_ses-20190929_ophys.nwb" \
    "$WHA_DATA/wormid_ey/sub-20190929-07_ses-20190929_ophys.nwb" \
    "$WHA_DATA/wormid_ey/sub-20191030-07_ses-20191030_ophys.nwb" \
  --feature-cache-dir work/results/wormid_extent/features \
  --output work/results/wormid_extent/benchmark_reproducible.json
```

Fit confidence only on the tuning-animal events and apply the frozen model to the
four other recordings:

```bash
python -m deformeot.wormid_calibration \
  "$WHA_DATA/wormid_ey/sub-20190925-04_ses-20190925_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-02_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-05_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-07_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20191030-07_ses-20191030_ophys.nwb" \
  --feature-caches \
    work/results/wormid_extent/features/sub-20190925-04_ses-20190925_ophys.npz \
    work/results/wormid_extent/features/sub-20190929-02_ses-20190929_ophys.npz \
    work/results/wormid_extent/features/sub-20190929-05_ses-20190929_ophys.npz \
    work/results/wormid_extent/features/sub-20190929-07_ses-20190929_ophys.npz \
    work/results/wormid_extent/features/sub-20191030-07_ses-20191030_ophys.npz \
  --calibration-recording-id sub-20190925-04_ses-20190925_ophys \
  --strides 1 5 10 20 40 \
  --bootstrap-samples 10000 \
  --output work/results/wormid_extent/dual_anchor_extent_calibration_heldout.json
```

Run the leakage-controlled trace analysis. Associations are completed from
coordinates before fluorescence is opened as an outcome:

```bash
python -m deformeot.downstream \
  "$WHA_DATA/wormid_ey/sub-20190925-04_ses-20190925_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-02_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-05_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20190929-07_ses-20190929_ophys.nwb" \
  "$WHA_DATA/wormid_ey/sub-20191030-07_ses-20191030_ophys.nwb" \
  --strides 1 5 10 20 40 \
  --methods previous_hungarian pairwise_affine first_affine dual_anchor \
  --reference-weight 6 \
  --descriptor-weight 0 \
  --bootstrap-samples 10000 \
  --tuning-recording-id sub-20190925-04_ses-20190925_ophys \
  --output work/results/wormid_downstream.json \
  --seed 20260720
```

## Independent-source checks

Targettrack uses one public freely moving movie, identity-stripped ground-truth
instances, a supervised training-frame anchor, and offline bidirectional
tracking. It is cue-isolation sensitivity evidence, not an end-to-end network or
between-animal validation:

```bash
python -m deformeot.targettrack \
  work/data/targettrack_assets/extracted/epfl10_CZANet_Final.h5 \
  --bootstrap 2000 \
  --seed 0 \
  --output work/results/targettrack_extent_benchmark.json
```

The fDNC/NeRVE release is a counterexample showing that strong adjacent linkage
does not guarantee persistent identity:

```bash
python -m deformeot.fdnc \
  work/data/fDNC_assets/extracted/test_tracking/real_jeff_all \
  --method all \
  --stride 1 \
  --max-distance 0.15 \
  --affine-iterations 4 \
  --affine-trim-fraction 0.70 \
  --affine-ridge 0.03 \
  --output work/results/fdnc_sequential_baselines.json
```

## Build and verify the paper artifact

The figure builder validates its input protocols before writing six main and one
supplemental figure in PNG/PDF, tables, and copied source-result JSON files. The
bundle builder then archives the repository sources, canonical compact
`work/results/` JSON, and an exact allowlist of paper outputs. It rejects
symlinks, raw-data-like suffixes, and unexpectedly large inputs.

```bash
python scripts/build_paper_assets.py --output outputs/paper
python scripts/build_manuscript.py \
  --source paper/manuscript.md \
  --output-dir outputs/paper
python scripts/build_reproduction_bundle.py --output-dir outputs/paper
python scripts/verify_paper_artifact.py --output-dir outputs/paper
(cd outputs/paper && sha256sum -c SHA256SUMS.txt)
```

The expected manuscript outputs are `outputs/paper/manuscript.md` and
`outputs/paper/manuscript.pdf`. The verifier also requires the copied Markdown to
match `paper/manuscript.md` byte for byte and checks every archive member against
the reproduction manifest.

Run the code-level checks separately:

```bash
python -m pytest -q
python -m ruff check src tests scripts
```

The exact passing-test count can increase as audits are added; zero failures and
zero Ruff findings are required. The supplied frozen unified-patch artifact has
SHA-256
`96a09f8fb9eb6a0128cf27a385fab3697842fa49445fe2c0d9bd6af2012f8b6a`.
Generated run-time fields can differ if an individual benchmark is regenerated;
the scientific fields and frozen copied artifacts are the reproducibility
targets.

No license has yet been supplied. A rights holder must add the appropriate
software/data/document license before public redistribution; authorship,
affiliations, repository URL/commit, and archival DOI also remain release tasks.
