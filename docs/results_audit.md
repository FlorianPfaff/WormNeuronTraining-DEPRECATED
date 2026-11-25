# Final results and claim audit

Audit date: 2026-07-20

## Supported result in one sentence

In a frozen, outcome-untouched cohort of 16 WormID/EY recordings, a
coordinate-only tracker that fused a temporal anchor with the initialized first
frame improved equal-recording, equal-stride seeded identity accuracy over a
locked direct first-frame affine comparator by 1.824 percentage points (95%
paired recording-bootstrap interval, 1.245 to 2.510), meeting the frozen
replication criterion.

This is a narrow within-source replication on device-restricted recordings with
an exhaustive fixed ROI cohort and first-frame identity initialization. It is
not evidence for cross-laboratory transfer, freely moving tracking, detector-gap
recovery, canonical neuron naming, or a Bayesian multi-hypothesis tracker.

## Evidence hierarchy and chronology

1. One recording was used to tune geometry and local-patch parameters.
2. Four convenience recordings formed the legacy evaluation. The
   `first_affine` comparator was selected after comparing implemented geometric
   baselines on these recordings, so their dual-anchor contrast is exploratory
   and conditional on comparator selection.
3. All 21 assets in immutable DANDI version `0.241009.1457` were enumerated.
   The remaining 16 recordings passed structural eligibility checks. Their
   coordinates were inspected for structure, but no method-comparison outcome
   was computed before the expansion protocol, cohort, comparator, parameters,
   seed rule, strides, aggregation, bootstrap, and decision rule were frozen.
4. The 16-recording expansion is therefore outcome-untouched with respect to
   the dual-anchor comparison and comparator choice. The freeze was internal,
   not externally registered or independently timestamped.
5. Local-patch, calibration, downstream, and independent-source analyses do not
   share the untouched-cohort status and remain secondary, post-hoc, or
   dataset-specific as labeled below.

## Primary frozen replication

The primary contrast averaged the five stride-specific accuracy differences
(strides 1, 5, 10, 20, and 40) equally within each recording and then averaged
the 16 recordings equally. A frozen 20,000-sample paired bootstrap resampled
complete recordings with seed `20260720`.

- Dual-anchor minus first-frame-affine seeded identity accuracy: **+1.824
  points** (95% interval, **+1.245 to +2.510**).
- Frozen decision rule, lower 95% endpoint above zero: **met**.
- Recording effects: 15 of 16 positive; the remaining effect was -0.066
  points.
- Stride-specific gains: +2.276, +1.684, +1.706, +1.706, and +1.749 points;
  every recording-bootstrap interval excluded zero.
- Pooled all-transition switch reductions: 74.3%, 61.5%, 57.7%, 57.0%, and
  53.6% across the five strides.
- Exact within-recording top-quartile cloud-motion accuracy: +1.498 points
  (95% interval, +1.075 to +1.963); all 16 recording effects were positive.
- High-motion switch reductions: 65.9%, 52.8%, 47.5%, 46.1%, and 45.4%.

All six acquisition-date mean effects were positive. An equal-date hierarchical
bootstrap gave +2.230 points (95% interval, +1.149 to +3.583). Excluding the
only untouched recording acquired on the tuning date gave +1.884 points (95%
interval, +1.271 to +2.598). These are declared sensitivity analyses, not
replacements for the recording-level primary estimate.

The successful claim is therefore that dual anchoring reduces seeded identity
drift relative to the locked first-frame affine comparator across animals and
dates within this one fixed-cohort source. Switch counts are interpreted jointly
with accuracy because a tracker can preserve a stable but wrong mapping.

## Legacy four-recording geometry result

The original four-recording analysis found a +1.699-point dual-anchor accuracy
gain (95% interval, +0.909 to +2.807); a rerun under the full-inventory seed
convention found +1.671 points (95% interval, +0.917 to +2.714). Exact
high-motion switch reductions in the original audit ranged from 40.2% to 71.6%.

These findings are directionally consistent with the untouched replication,
but they are not confirmatory because the headline comparator was selected on
the same four recordings. Their proper role is method development and
exploratory precedent for the later frozen test.

## Secondary local-patch cue

On the four legacy evaluation recordings, a tuning-frozen 15-dimensional cue
combining intensity-weighted image-centroid offsets and covariance/axis features
improved overall seeded identity accuracy over point-only dual anchoring by
+0.977 points (95% descriptive interval, +0.611 to +1.459). Under the executed
tuning-animal nearest-neighbor Q25 cutoff (1.360 um; 18.58% prevalence), the
gain was +2.260 points (95% descriptive interval, +1.039 to +4.220).

The internal plan instead defined dense scenes as separation below twice the
median nucleus radius. That definition was not implemented. Plausible
radius-proxy sensitivity cutoffs retained positive effects but reduced the gain
below 2 points. The density target is therefore **a protocol deviation with
supportive evidence, not a passed target**.

Additional mechanism analyses were post-hoc:

- Removing the explicit reliability gate, without retuning weight or update,
  gave +1.350 points overall and +2.390 points in the executed Q25 stratum. This
  shows that the benefit did not require that gate; it does not make the cue
  intensity independent because all spatial features remain intensity weighted.
- Image-centroid offsets alone gave +0.712 points overall and +0.562 points in
  the Q25 stratum. Covariance/axis features alone gave +0.729 and +1.463 points.
  Thus both components contribute, but the analysis does not isolate intrinsic
  neuron morphology or ownership of patch voxels.

The local-patch result has not been replicated in the untouched 16-recording
cohort.

## Failed and inconclusive planned endpoints

### Confidence calibration: failed

The joint target required both ECE below 0.05 and at least 99.5% identity
accuracy at 80% selective coverage. Fixed-bin ECE was 0.0167, but accuracy at
80% coverage was 98.459% (recording-macro 98.366%; 95% descriptive interval,
97.544% to 99.188%). The tuning-derived threshold retained 54.60% of events at
99.449% accuracy. Because both conditions were required, the target failed.
These event-level logistic scores are not a calibrated posterior over track
histories.

### Downstream fluorescence: failed

The specified endpoint was the four-recording mean paired gain in the recording
median of per-neuron raw-trace Pearson correlations. The gain was +0.0071 (95%
descriptive interval, +0.0022 to +0.0123), below the +0.03 target. No behavior
decoder was run, so the downstream target failed.

Post-hoc gains in mean raw correlation (+0.0481), median first-difference
correlation (+0.2420), derivative-event F1 (+0.3384), and wrong-source fraction
(-0.0209) are mechanistically consistent with reduced switching but do not
rescue the failed endpoint. The event score is not spike inference.

### Gap recovery: inconclusive

The WormID core contains exhaustive detections for a fixed initialized cohort.
Temporal subsampling increases the interval between complete observations; it
does not create missing detections inside an evaluated frame. The planned gap,
merge, and split claims therefore cannot be tested on this benchmark.

## Independent-source constraints

- **Targettrack:** On one freely moving-worm movie, a cue-isolation analysis
  used a supervised training-frame anchor, offline bidirectional tracking, and
  identity-stripped ground-truth masks. Forced-assignment canonical IDF1 rose
  from 0.0748 to 0.1031, global-mapping IDF1 from 0.2103 to 0.2851, and switches
  fell from 166 to 114. This is one-movie association sensitivity, not an
  end-to-end segmentation result or a generalization estimate. Selective switch
  counts are not directly comparable because coverage differed.
- **fDNC/NeRVE release:** Simple robust-affine Hungarian association reached
  adjacent-link F1 0.8533 while partial temporal IDF1 remained 0.0463. This is a
  counterexample showing that strong local linkage does not certify persistent
  identity; it is not an evaluation of the learned fDNC model itself. Unlabeled
  detections were ignored in the partial-label score.
- **ASCENT audit:** The released track-only table reproduced 10 mismatches and
  modified MOTA 0.999905 under the official evaluator. Current released code on
  released predicted detections gave 259 mismatches, 800 misses, modified MOTA
  0.989971, partial IDF1 0.965950, and median localization error 0.715 um.
  Unmatched extra predictions are ignored, and released-table, current-code,
  and historical-paper artifacts must remain version specific.

These checks delimit metrics and transfer; their incompatible detections,
labels, supervision, and acquisition regimes must not be pooled with the WormID
effect.

## Internal target ledger

| Internally specified target | Final status | Evidence boundary |
|---|---|---|
| Severe/long-interval identity: +3 accuracy points or at least 25% fewer switches | **Met and replicated within source** | Exact high-motion switches fell 45.4% to 65.9% across strides in the frozen 16-recording cohort; high-motion accuracy gained 1.498 points. |
| Local extent: +2 accuracy points below twice median nucleus radius | **Protocol deviation; supportive only** | The combined cue gained 2.260 points under a different Q25 cutoff; plausible radius proxies gave less than 2 points. |
| At 80% coverage: at least 99.5% precision and ECE below 0.05 | **Failed** | Accuracy was 98.459%; ECE was 0.0167. Both were required. |
| Gap recovery: +10 points without localization penalty | **Inconclusive** | Core data contain no missed detections; the external one-movie audit cannot answer the target. |
| Raw-trace correlation +0.03 or behavior decoding | **Failed** | Median raw Pearson gain was +0.0071; no behavior decoder was run. |

The first numeric switch criterion was met and replicated, but the original
broad paper-worthiness gate was not. The validated method is deterministic,
single assignment, and coordinate based. The proposed Bayesian
multi-hypothesis, visibility, missing/merge/split, required-ablation, and
published-baseline program was not completed.

## Claim boundary and paper-worthiness

The evidence is paper worthy for the narrow claim: a frozen,
outcome-untouched 16-recording within-source replication shows that combining
temporal and immutable first-frame anchors reduces seeded identity drift over a
locked geometric comparator. The secondary local-patch result motivates future
work but is not part of that untouched replication.

The study must not claim:

- a general whole-brain tracker across laboratories or acquisition regimes;
- freely moving, online, detector-error, birth/death, merge/split, or gap
  performance;
- de novo identity initialization or semantic neuron-name identification;
- a calibrated Bayesian posterior or validated multi-hypothesis tracker;
- intrinsic neuron morphology from intensity-weighted local patches; or
- a passed confidence, downstream-fluorescence, or radius-defined density
  target.

## Reproducibility anchors

- DANDI dataset: `000541`, immutable version `0.241009.1457`.
- Frozen expansion protocol SHA-256:
  `b99c18526c104993aced172231607e0588c4e84e0b0cce6919d94af7407b56c3`.
- Frozen expansion configuration SHA-256:
  `fdca2e18c61bcb3544106c4cd14e0945be789e6818ecd5e8904d9990c3b42bf0`.
- Canonical prospective result SHA-256:
  `c210e6fe2350fa1e2988d4ad20579b3afc76ae33a7a49e82a10ff27235945b3a`.

The result artifact records all 21 source assets, cohort assignments, source
digests, derived coordinate-cache identities, frozen seeds, zero exclusions,
and the complete 16-recording and four-recording reruns.
