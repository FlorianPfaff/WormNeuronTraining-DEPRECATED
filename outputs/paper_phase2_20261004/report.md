# Phase-two paper-worthiness assessment

Decision: **conditional followup on robust development lesson**.

This is development evidence on previously examined recordings, not a newly untouched replication or a submission-ready paper.

## Native-rate comparison

| Method | Accuracy | Without seed | Switches / 1000 | Mean wrong episode (scans) | Longest wrong episode (scans) | Coverage | Animals |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pairwise_affine | 75.832% | 75.806% | 9.877 | 67.84 | 949 | 100.0% | 20 |
| first_affine | 90.800% | 90.790% | 63.232 | 3.63 | 945 | 100.0% | 20 |
| dual_anchor | 93.029% | 93.021% | 15.146 | 10.81 | 935 | 100.0% | 20 |
| cpd_reference | 90.045% | 90.034% | 68.110 | 3.74 | 939 | 100.0% | 20 |
| cpd_temporal | 75.779% | 75.754% | 9.829 | 69.70 | 954 | 100.0% | 20 |
| fdnc_reference | 77.230% | 77.206% | 165.448 | 3.88 | 956 | 100.0% | 20 |

Completed unique comparison tasks: 120 / 120. Each animal has equal weight; CPD-reference is the fixed primary comparator.
Dual-anchor accuracy difference: +2.984 percentage points (95% paired-animal bootstrap interval +1.873 to +4.237).
Absolute switch-rate benefit: +52.965 per 1000 transitions (95% interval +43.138 to +63.181).
Relative mean switch-rate reduction: +77.8%.
Prespecified practical screen passed: `True`. All acquisition-date/initialization sensitivity gates passed: `True`.

The screen requires at least 25% relative switch reduction, a positive lower 95% bound on switch benefit, and an accuracy-difference lower bound above −0.25 percentage points. Intervals use 20,000 paired draws with seed 20261004 and are unadjusted descriptive intervals, not confirmatory significance claims.
Whole-acquisition-date bootstrap, every leave-one-date-out analysis, exclusion of the tuning-adjacent acquisition date, and exclusion of the seeded frame are retained in `summaries.json`; date clusters repeat all their animals, with equal-animal means in each bootstrap sample.
Requiring every sensitivity gate for the robust-development label is a conservative follow-up judgment, not an additional prespecified primary-screen requirement.
Coverage refers only to the initialized exhaustive supplied-ROI cohort, not detector recall or coverage of every biological neuron. The mean wrong-episode duration is an equal-animal mean of each animal's episode mean; the longest episode is the maximum across animals. Low switch counts alone can conceal persistent wrong identities.

Expanded CPD tuning uses only the original tuning animal. Boundary winners are flagged in the summary; a finite grid does not establish a globally optimal registration baseline. fDNC is a coordinate-only released-model adapter: model-data overlap and transfer limits remain explicit, and this is not a ranking of complete image-based tracking pipelines.

## Matched-information mechanism

Complete matched-lookahead controls did not meet the prespecified practical screen for the beam-marginal variant versus rolling best path; the recorded pairwise contrasts delimit filtering, lookahead, beam pruning, and commitment effects under these settings.

Complete unique controls: 30 / 30. Practical screen passed: `False`.

| Candidate vs comparator | Stride | Accuracy difference (pp) | Switch benefit / 1000 | Positive animals |
| --- | ---: | ---: | ---: | ---: |
| beam_marginal vs rolling_best_path | 1 | -0.207 | +0.288 | 0 / 3 |
| beam_best_path vs rolling_best_path | 1 | -0.210 | +0.310 | 0 / 3 |
| beam_marginal vs beam_best_path | 1 | +0.002 | -0.023 | 1 / 3 |
| rolling_best_path vs online | 1 | +0.803 | -0.910 | 3 / 3 |
| online vs dual_anchor | 1 | -0.299 | -0.643 | 2 / 3 |
| beam_marginal vs rolling_best_path | 10 | -0.048 | -0.303 | 1 / 3 |
| beam_best_path vs rolling_best_path | 10 | -0.048 | -0.303 | 1 / 3 |
| beam_marginal vs beam_best_path | 10 | +0.000 | +0.000 | 0 / 3 |
| rolling_best_path vs online | 10 | +0.237 | +0.170 | 3 / 3 |
| online vs dual_anchor | 10 | +0.194 | +0.003 | 2 / 3 |

The primary mechanism screen requires at least +0.25 accuracy points at native rate, no mean switch-rate increase, and positive accuracy differences in at least two animals. Rolling-path and beam modes share two future selected observations; final windows are shortened. Rolling optimization is exact only over its admitted candidate tree. These controls do not establish calibrated posterior probabilities or an extended-object benefit.
The selected configuration uses geometry costs, so beam weights are normalized geometry scores rather than validated likelihoods or calibrated identity probabilities. The primary beam-marginal versus rolling-path contrast jointly changes retained/pruned path support and commitment; its gain cannot be attributed solely to marginalization. Beam-best versus rolling is not a pure retention ablation either. Beam-marginal versus beam-best is the narrower commitment contrast, after which their trajectories can diverge.

## Timing and assignment reproducibility

Verified serial repeats: 8 / 8; complete timing/array check: `True`.

| Method | Mean full adapter call (s) | Ratio to dual anchor |
| --- | ---: | ---: |
| dual_anchor | 5.476 | 1.00 |
| rolling_best_path | 1102.435 | 201.33 |
| beam_best_path | 339.918 | 62.08 |
| beam_marginal | 334.442 | 61.08 |

full adapter call including initialization; excludes input loading, independent scoring, and artifact writing. Two serial repeats on one development animal; unrelated server load remains uncontrolled. Not a general hardware speedup or a GPU claim.
Exact saved assignments, truth permutations, frame indices, and timestamps are compared across timing repeats and their same-case scientific reference runs. Independent rescoring checks full-coverage permutations and integer endpoints, not biological label validity.

## Publication decision and unresolved evidence

- The reviewed closest literature does not establish a distinct new-method contribution.
- No credible independently verified, previously unused multi-animal external route is established.

### Public external-data eligibility

| Asset | Status | Verified scope | Main unresolved reasons |
| --- | --- | --- | --- |
| SK1_20230322_185704 | conditional | unverified | bounded_inspection_incomplete; temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| SK1_20230322_201650 | conditional | unverified | bounded_inspection_incomplete; temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| SK1_20230506_150145 | conditional | unverified | bounded_inspection_incomplete; temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| SK1_20230506_153351 | conditional | unverified | bounded_inspection_incomplete; temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| SK2_20220426_w00 | conditional | unverified | bounded_inspection_incomplete; temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| SK2_20220426_w01 | conditional | unverified | bounded_inspection_incomplete; temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_2308918_01 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_2308918_03 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_2308918_04 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_230928_02 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_230928_03 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_230928_04 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_231013_01 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_231013_02 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| KK_231013_03 | conditional | unverified | temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; asset_alias_review_against_project_exposure_not_complete |
| zephir_ZM9624 | conditional | unverified | asset_or_subset_expert_verification_and_release_mapping_not_verified; physical_calibration_not_verified; asset_timing_not_verified; asset_alias_review_against_project_exposure_not_complete |
| three_dee_worm1 | conditional | unverified | bounded_inspection_incomplete; temporal_coordinate_coverage_not_verified_for_declared_scope; persistent_biological_identity_semantics_not_verified; asset_or_subset_expert_verification_and_release_mapping_not_verified; physical_calibration_not_verified; asset_timing_not_verified; asset_alias_review_against_project_exposure_not_complete |
| three_dee_worm4 | excluded | unverified | known_previously_exposed_recording_alias |

`eligibility_inventory.csv` retains alias, calibration/time, expert-verification/release mapping, licensing, and comparator-specific training-overlap checks. Unknown provenance means unverified, not that usable labels do not exist. Sparse documented expert-verified subsets may support scoped validation; they do not establish dense all-frame identity truth. Expert-checked algorithm-assisted references are eligible only with exact release/subset mapping. Unknown fDNC training overlap does not by itself disqualify an untrained geometric comparator.
The ten closest primary papers and their anchoring, deformation, future-frame access, hypothesis retention, annotation requirements, and evaluation scope are tabulated in `closest_papers.csv`.

The equal-animal dual-anchor advantage over the fixed stronger CPD-reference comparator meets the prespecified accuracy/continuity criterion and survives every prespecified acquisition-date and seeded-frame sensitivity. This is a reproducible matched-input development observation, not a new tracking principle or independent biological validation.

The closest-paper claim audit is in `claim_evidence.csv`. A quantitative comparison alone does not establish a novel tracking principle. MHHT already contrasts depth-first ranked-assignment trees with retained K-best hypotheses; the rolling-versus-beam control is therefore prior art. Complete controls alone do not justify a benchmark or mechanism paper. A robust practical development lesson can motivate conditional follow-up, with novelty and external validation still unresolved; otherwise the outcome remains a technical report.

## Historical closure and scope

One bounded search found the July aggregate results, source manifest, and environment record, but no original per-frame assignment arrays or full BLAS/thread record. The discrepancy remains unexplained; the October reconstruction is not the July execution. Original numbers remain preserved and qualified. Exact July recovery is retired as a prerequisite for a newly frozen study; it is not claimed solved.
All current analyses are restricted to the active source signature. Older attempts and failures remain in the checkpoint archive and are listed separately, never silently pooled. Fresh NPZ assignments and an independently implemented scorer permit exact rescoring of this execution.
Accuracy measures agreement with algorithm-assisted, expert-verified supplied ROI trajectories, not independently adjudicated identity at every frame. No external test identities were scored. No neuroscience efficacy, downstream calcium recovery, detection-gap, merge/split, extended-object, or calibrated uncertainty claim is supported by this phase.

Machine-readable effects, sensitivity analyses, completion issues, integrity checks, and resource snapshots are in `summaries.json`; per-case endpoints are in `metrics.csv`; the feasibility decision is in `decision.json`.
