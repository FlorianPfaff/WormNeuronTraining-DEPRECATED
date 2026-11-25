# Neuron tracking paper feasibility pilot

Decision: **conditional go**.

Practical screen: **passed**. Publication readiness: **blocked**. A failed development screen is not converted into a positive go because other evidence remains unresolved.

This is a bounded exploratory development pilot, not independent confirmation. Three recordings were selected by low, median, and high identity-blind motion; the original tuning recording was excluded. All recordings had been examined in earlier work.

## Native rate comparison

| Method | Accuracy | Switches per 1000 neuron-transitions | Future scans |
| --- | ---: | ---: | ---: |
| cpd_reference | 89.341% | 73.476 | 0 |
| cpd_temporal | 72.463% | 10.872 | 0 |
| dual_anchor | 92.316% | 17.226 | 0 |
| fdnc_reference | 76.495% | 164.142 | 0 |
| first_affine | 90.061% | 66.090 | 0 |
| gnn_kalman | 92.090% | 17.866 | 0 |
| mht_geometry_lag2 | 92.613% | 18.492 | 2 |
| mht_geometry_online | 92.017% | 17.869 | 0 |
| pairwise_affine | 73.099% | 10.672 | 0 |
| previous_hungarian | 71.642% | 10.853 | 0 |

Development-selected strongest new comparator by native-rate accuracy: `cpd_reference`. Dual-anchor accuracy difference: +2.976 percentage points.
Relative switch-rate reduction: +76.6%.
The provisional 25% switch-reduction / at-most-0.25-point accuracy-loss screen passed: `True`.

Method means weight the three recordings equally. Native rate is the primary screen; stride-10 and stride-40 means are stress-test summaries in `per_stride_comparison.csv`. Per-case metrics and failures are retained in `benchmark_details.csv`.

`mht_geometry_online` has lag zero: its result reflects Gaussian filtering/scoring, not persistent competing association histories. `mht_geometry_lag2` uses two future selected observations; it is not a matched-information comparison with an online method.

## Isolated repeated timing

Median-motion recording: `sub-20190928-08_ses-20190928_ophys` at stride 1. Complete repeats: 20 / 20 for available methods (20 planned across all ten methods). Each reported method has both repeats.

| Method | Algorithm-reported mean (s) | Ratio to dual | Whole-worker mean (s) | Ratio to dual |
| --- | ---: | ---: | ---: | ---: |
| cpd_reference | 44.534 | 8.17 | 45.827 | 6.31 |
| cpd_temporal | 27.804 | 5.10 | 29.309 | 4.04 |
| dual_anchor | 5.451 | 1.00 | 7.262 | 1.00 |
| fdnc_reference | 59.815 | 10.97 | 62.484 | 8.60 |
| first_affine | 3.032 | 0.56 | 4.283 | 0.59 |
| gnn_kalman | 44.745 | 8.21 | 46.769 | 6.44 |
| mht_geometry_lag2 | 351.556 | 64.49 | 353.859 | 48.72 |
| mht_geometry_online | 110.962 | 20.36 | 113.454 | 15.62 |
| pairwise_affine | 2.330 | 0.43 | 4.274 | 0.59 |
| previous_hungarian | 0.369 | 0.07 | 1.452 | 0.20 |

These are serial repeated observations with no other pilot jobs running, but unrelated server load remained uncontrolled and is recorded per repeat. Algorithm-reported timer boundaries differ: the fDNC timer starts after model loading and preprocessing and includes inference plus the motion metric; other adapters have different boundaries. Whole-worker wall time comes from the supervisor budget ledger and includes process startup, imports/setup, loading, tracking and scoring as a separate process-level proxy; checkpoint-reported worker intervals, which exclude initial startup/imports, are also retained in the repeat CSV. Neither column is uniformly instrumented steady-state or end-to-end tracker latency. Ratios are descriptive observations, not definitive speedups or the best attainable fDNC GPU performance. No model-load/steady-state decomposition was measured.

`isolated_timing.csv` contains complete-pair means and ratios; `isolated_timing_repeats.csv` retains individual observations and load. Concurrent benchmark runtimes are retained separately and are not used for this timing table.

## Audit and limitations

- No independently annotated external multi-animal cohort has yet been fully verified.
- Not every selected original historical aggregate result was reproduced. Present-day environment agreement does not resolve the original discrepancy; the original July assignment arrays are unavailable.

Reference identity classification: algorithm_assisted_expert_verified_reference_trajectories.
Accuracy means agreement with supplied ROI trajectories. The reviewed acquisition workflow used algorithm-assisted tracking followed by expert verification; independent manual identity annotation at every frame was not established. Stable ROI rows and semantic NeuroPAL names are distinct from independently verified biological identity ground truth.

Complete benchmark tasks: 90 / 90 for available candidates (90 planned with all ten methods).
Conservatively accounted cumulative job time: 2.418 / 24 hours. Accounted additional storage: 1.695 / 10 GiB.

Historical environment reconstruction matches package pins where possible, not the original Python patch version, WSL host, or BLAS dispatch. All recorded mismatches are retained. The current and reconstructed historical environments produced identical assignments in all six diagnostic cases. There were no assignment divergences and therefore no first-divergence captures for that environment contrast. Original July assignment arrays are unavailable, so the original divergence cannot be located and the historical discrepancy remains unexplained. Do not attribute the original discrepancies to ties or dependencies without further evidence.

Identity accuracy includes the initialized frame for compatibility with historical scores; the machine-readable rows also give accuracy excluding it. All methods have complete one-to-one coverage. Wrong-identity episode lengths are measured in selected scans, not seconds. No detector-gap, extent, calibrated-uncertainty, or downstream calcium claim is evaluated.

This implements only the bounded first work package. Matched-lookahead mechanism ablations, new-source confirmation, biological utility, and a comprehensive published-method ranking remain outside this run. The five-frame fDNC inference check validates adapter fidelity, not training-animal independence. Peak memory, warmup-adjusted timing and annotation effort were not measured.

The public-source metadata review was not individually timed; its five short server jobs are conservatively charged at 1,200 seconds. Other numerical worker jobs are wall-clock timed.

## Reproduction

Run `python -m deformeot.paper_pilot --help` for phase controls. The experiment uses `configs/paper_pilot.yaml`, separate per-task checkpoints, a locked resource ledger, and exact data/code/model/environment signatures. Historical frozen files are not edited.

The detailed numerical audit, baseline validation, inventory evidence, task logs, and budget are retained under `work/results/paper_pilot_20261003`. Failed or missing baselines are not silently replaced. No external test identities were scored.

The exact execution source is retained in `execution_source.tar.gz` beside `context.json`. A reporting-only revision adds timing summaries without changing the numerical checkpoints; `final_verification.json` records both renderer hashes, the source archive hash, and final verification. Its enforced two-minute reservation is charged conservatively to the budget. Supervisor wall time is additionally charged as a conservative overhead allowance; the total is not a CPU-time measurement.
