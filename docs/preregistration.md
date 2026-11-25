# DeformEOT preregistration

Date frozen: 2026-07-20

## Research question

Can a deformation-aware, multi-hypothesis extended-object tracker preserve
neuron identity through severe nervous-system deformation and detector failures,
while producing calibrated uncertainty that protects downstream calcium-trace
analyses?

## Novelty boundary

The claim is deliberately narrow:

> In a primary-literature search through July 2026, we found no demonstration in
> freely moving adult *C. elegans* whole-brain microscopy of a deformation-aware
> Bayesian tracker that maintains a calibrated joint temporal posterior over
> canonical identity, visibility, 3D position, and dynamic extent while retaining
> multiple hypotheses through missed, merged, and split detections.

We do **not** claim that neuron extent tracking, probabilistic correspondence,
multiple-hypothesis tracking, deformation-aware tracking, or calibrated link-error
prediction is individually new.  Targettrack tracks 3D volumes; DSMM produces
pairwise correspondence probabilities; MHHT tracks embryonic *C. elegans* cells;
EMC2 combines eMHT with deformation in other neural tissue; CELLECT estimates cell
size and linkage reliability; and other cell-tracking work calibrates link errors.

## Locked evaluation principles

1. Split by recording or animal whenever multiple recordings are available. Never
   split randomly by frame.
2. Tune hyperparameters only on development recordings or the first contiguous
   calibration block declared before inspecting test labels.
3. Keep detection inputs identical across tracking backends.
4. Report full false positives whenever the ground truth exhaustively labels the
   scene. If labels cover only a subset, report a partial-label metric and state that
   unlabeled detections are ignored rather than calling them false positives.
5. Report recording-level bootstrap confidence intervals and per-recording results;
   frames are not independent replicates.
6. Treat calcium intensity as a scientific signal, not a stable identity feature.
7. Preserve failed hypotheses and negative results. The point-only model is the
   accepted result if the extent terms do not improve the prespecified hard strata.

## Datasets and roles

- **ASCENT in-house Opterra**: 1,100 volumes, 96 proofread tracks, predicted
  detections and tracks. Used first to reproduce the published modified-MOTA result
  and to test detector-miss recovery. Its extra detections may be real unlabeled
  neurons, so they are ignored in partial-label scoring.
- **NeRVE**: freely moving head recording with 79 manual tracks. Held out for
  moving-brain generalization once access and storage are validated.
- **Targettrack volumetric example**: public 3D volume/mask data used to develop and
  ablate extent likelihoods. It is a prototype set, not sufficient alone for a paper
  claim.
- **BrainAlignNet/ANTSUN demo and test data**: independent registration pairs used
  for severe-deformation and cross-method checks.
- **WormID-Bench**: independent detection/tracking benchmark used if its full
  identity/segmentation annotations can be executed reproducibly.

## Primary outcomes

- Identity: IDF1, HOTA association accuracy, identity switches, fragmentation.
- Detection/localization: DET/TRA where defined, GOSPA/OSPA2, centroid error.
- Extent: 3D Dice, IoU, panoptic quality, volume and principal-axis error.
- Calibration: negative log likelihood, Brier score, expected calibration error,
  risk-coverage curve, and empirical credible-region coverage.
- Neuroscience: calcium-trace correlation/distortion and held-out-recording behavior
  decoding.

## Prespecified hard strata

- nearest-neighbor separation below twice the median nucleus radius;
- natural or injected gaps of 1, 2, 3, 5, and 10 frames;
- merged and split detections;
- top quartile of frame-to-frame deformation;
- low-SNR or z-cropped observations;
- long-gap or opposite-curvature registration pairs.

## Falsifiable target claims

The method will be described as successful only if all uncertainty intervals are
reported and at least one independent real-data set supports the corresponding claim.

1. In severe-posture/long-gap strata, improve IDF1 by at least 3 percentage points
   or reduce identity switches by at least 25% versus the strongest executable
   baseline.
2. The extent likelihood improves IDF1 by at least 2 points when separation is below
   twice the median radius. No general extent benefit is claimed if this fails.
3. At 80% selective coverage, association precision is at least 99.5% and expected
   calibration error is below 0.05 on held-out recordings.
4. For gaps of 1-10 frames, improve recovery recall by at least 10 points without
   increasing median localization error by more than 5%.
5. Corrected tracking improves median calcium-trace correlation by at least 0.03 or
   measurably reduces held-out-recording behavior-decoder error.

## Required baselines

- nearest-neighbor and constant-velocity Hungarian tracking;
- rigid/affine alignment plus Hungarian assignment;
- coherent point drift or thin-plate-spline registration plus assignment;
- ASCENT, ZephIR, Targettrack, and BrainAlignNet/ANTSUN where their released inputs
  and checkpoints are compatible;
- Ultrack on exhaustive segmentation benchmarks;
- point-only, single-hypothesis, filter-only, and no-shared-deformation internal
  ablations.

## Required ablations

- remove shared deformation;
- remove extent/scatter/mask terms;
- single best assignment versus K-best posterior;
- filter versus fixed-lag smoother;
- remove visibility state;
- remove local residual motion;
- compare extent transport, rotation-only extent, and extent random walk;
- compare raw and calibrated posterior scores.

## Paper-worthiness gate

The work is not paper worthy until:

- published metrics are reproduced from released predictions;
- all baselines run from documented commands on at least two independent real data
  sources;
- the main claim survives recording-level uncertainty intervals and all required
  ablations;
- at least one downstream neural-signal result is demonstrated;
- code, environment, data manifests, seeds, figures, tables, and manuscript are
  reproducible from a clean checkout;
- limitations and counterexamples are stated without novelty inflation.

