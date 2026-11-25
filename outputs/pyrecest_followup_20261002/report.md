# PyRecEst neuron-tracking follow-up

Completed on gpuserver6000. Settings were selected using the original single tuning recording. All 20 other recordings and all five original strides (1, 5, 10, 20, 40) were evaluated. The comparison uses the same supplied centers, first-frame initialization, anonymous detection permutations, shared affine correction, and immutable reference anchor.

These recordings had already been evaluated in the earlier study. This is a follow-up comparison; its recording-bootstrap intervals are descriptive and do not provide new untouched confirmation or cross-laboratory validation. The three selected families are separate comparisons; intervals are not corrected for multiplicity.

## Comparison with the current dual-anchor tracker

| Method | Accuracy gain (percentage points; 95% interval) | Switch reduction | Runtime ratio | Future selected scans |
| --- | ---: | ---: | ---: | ---: |
| PyRecEst GNN | +0.180 [+0.084, +0.278] | -1.6% | 7.5x | 0 |
| PyRecEst Gaussian filtering, online | +0.171 [+0.085, +0.256] | -1.9% | 17.4x | 0 |
| PyRecEst MHT, lag 2 (geometry) | +0.244 [+0.153, +0.338] | -3.7% | 54.9x | 2 |

These implementations offer small accuracy gains, not an across-the-board improvement: every selected variant produces more identity switches and costs more computation. The current dual-anchor tracker remains the practical default when speed and continuity matter. GNN is the least expensive tested PyRecEst option; lag-2 MHT has the largest accuracy gain but also uses future observations. This does not establish that PyRecEst cannot improve on these implementations with other models or settings.

![Paired recording-bootstrap accuracy comparisons](accuracy_comparison.png)

Execution recovery: the server reboot removed the earlier temporary evaluation directory. This complete evaluation was rerun in persistent server storage, using the saved tuning selection and identical code/package signature. Task parallelism was increased from 12 to 24 workers; scientific parameters, data, and selection were unchanged. The recovery record includes the preserved selection hash.

Accuracy gives equal weight to each recording after averaging its five strides. Intervals resample paired recordings 20,000 times. Switch totals pool overlapping stride analyses and are descriptive. Positive switch reduction means fewer switches. Runtime ratios compare recorded algorithm time in the same server environment; individual tasks ran concurrently.

MHT with lag 2 uses two subsequent selected scans before committing an assignment. Its delay therefore grows with stride, and its accuracy gains must be interpreted with that extra information. The online MHT row provides a comparison with no future scans.

## Implementation and limits

GNN uses native PyRecEst assignment and Gaussian updates, with transported mean/covariance, Mahalanobis association, and an external reference cost. An equivalence control routes the original dual-anchor objective through GNN. MHT uses native Gaussian filter banks, Murty ranked assignments, alternative histories and hypothesis pruning; adapter hooks add reference scores before branching and condition assignments on the exhaustive cohort. Candidate edges are limited by a score window and per-track cap, with a complete Hungarian backbone. The beam is bounded at four global hypotheses and three children per parent. These approximations may discard a correct history. With zero lag, commitment conditions the beam on one complete assignment each scan; any gain there primarily tests recursive state filtering/scoring. The lag-2 variant tests retaining alternative recent assignment histories.

No semantic neuron labels or fluorescence enter association. The coordinate caches have no missing/false detections or segmented extent, so this experiment does not assess birth/death, clutter handling, extended objects, or downstream calcium signal recovery. Correlated deformation and reference cues prevent interpreting the hypothesis weights as calibrated identity probabilities.

Historical baseline outcomes were not exactly reproduced: 121 of 200 recording/stride/method rows have at least one identity-outcome difference. For dual anchor, mean accuracy shifted by -0.021457 percentage points; the largest absolute individual-case shift was 0.742660 points. All comparisons above use freshly rerun baselines in the same environment, not the historical scores. The cause of historical discrepancies has not been isolated. Full mismatch rows and drift summaries for both baselines are in `summary.json`; the canonical results were not overwritten.

GNN equivalence control completed 100 of 100 tasks. Assignments matched the original objective's rerun baseline exactly in 91 tasks. The largest absolute accuracy difference was 0.032737 percentage points. Differences are consistent with numerical sensitivity in the mathematically equivalent cost path; equality of the alternative optima was not verified for every discrepancy. The complete diagnostic rows are saved in `summary.json`.

## Reproduction

Use Python 3.11 or newer and the optional pinned PyRecEst dependency from `pyproject.toml`. The measured environment used Python 3.12.3; its resolved package versions are in [requirements-pyrecest-lock.txt](../../requirements-pyrecest-lock.txt) and execution checks in [verification.json](../../work/results/pyrecest_followup_20261002/verification.json). Clone `work/vendor/PyRecEst` at commit `7eff44c5f4d62c54a8394270fc924cb1302f171b`. Use the original verified coordinate caches (the evaluator extracts a missing tuning cache from the immutable asset through byte-range reads). On a calculation server, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python -m deformeot.wormid_pyrecest_experiment \
  --protocol configs/wormid_pyrecest.yaml --cache-dir /path/to/coordinate-caches \
  --output-dir work/results/pyrecest_followup_20261002 --phase all --workers 12
```

The experiment saves code/package/data hashes, tuning selection, per-task checkpoints, and complete evaluation results. Existing checkpoints are reused only with matching provenance. A changed implementation or environment requires a fresh output directory.

Complete machine-readable results: [evaluation_results.json](../../work/results/pyrecest_followup_20261002/evaluation_results.json). All 500 evaluation tasks and 100 equivalence-control tasks completed; the recovered environment passed all 103 tests.

Evaluation JSON SHA-256: `d7e21a8f9f33516a7c77536284dd17b62e52384606f063e33288e2f5d1759572`.

The earlier paper source results and frozen expansion configuration remain unchanged.
