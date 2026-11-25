# Frozen protocol for untouched-cohort WormID geometry replication

Status: frozen before any tracker comparison was computed on the 16-recording
expansion cohort.

Freeze date: 2026-07-20 (Europe/Berlin)

This is an internally frozen protocol, not an externally registered or
third-party-timestamped preregistration. Before freezing it, all DANDI assets were
queried for structural eligibility and their coordinate arrays were checked for
finite values and loader compatibility. No association method had been run and no
identity-accuracy or switch outcome had been computed on the 16 recordings below.

## Source and inclusion

- DANDI set `000541`, immutable version `0.241009.1457`.
- Include all 21 published NWB assets. Structural validation found all 21 eligible.
- Retain `sub-20190925-04_ses-20190925_ophys` as the sole legacy tuning animal.
- Treat the four recordings already used in the first split evaluation as legacy
  evaluation data only:
  - `sub-20190929-02_ses-20190929_ophys`
  - `sub-20190929-05_ses-20190929_ophys`
  - `sub-20190929-07_ses-20190929_ophys`
  - `sub-20191030-07_ses-20191030_ophys`
- The primary untouched expansion cohort is every remaining asset, exactly:
  - `sub-20190924-01_ses-20190924_ophys`
  - `sub-20190924-02_ses-20190924_ophys`
  - `sub-20190924-03_ses-20190924_ophys`
  - `sub-20190925-01_ses-20190925_ophys`
  - `sub-20190928-01_ses-20190928_ophys`
  - `sub-20190928-03_ses-20190928_ophys`
  - `sub-20190928-05_ses-20190928_ophys`
  - `sub-20190928-07_ses-20190928_ophys`
  - `sub-20190928-08_ses-20190928_ophys`
  - `sub-20190928-11_ses-20190928_ophys`
  - `sub-20190928-13_ses-20190928_ophys`
  - `sub-20190929-03_ses-20190929_ophys`
  - `sub-20190929-06_ses-20190929_ophys`
  - `sub-20191030-03_ses-20191030_ophys`
  - `sub-20191104-08_ses-20191104_ophys`
  - `sub-20191104-10_ses-20191104_ophys`
- Do not exclude a recording because of its outcome. A digest, network, or
  structural failure triggers retry and explicit reporting, not silent exclusion.
- Blank semantic `NeuronIDs` labels do not cause exclusion. Ground truth is stable
  ROI/neuron-instance row identity, not semantic cell-name identity.

## Frozen association protocol

- Use only the center trajectories represented by the NWB `voxel_mask` tables.
- Use the existing `identity_blind_sequence` permutation and `track_sequence`
  implementations without outcome-driven modification.
- Geometry configuration is frozen at reference weight 6, descriptor weight 0,
  affine iterations 6, ridge 0.2, Huber delta 2.5, scale bounds 0.70 to 1.40,
  and spatial-scale floor 0.10 um.
- Evaluate temporal strides `[1, 5, 10, 20, 40]`; start at source frame 0 and append
  the actual last frame when a stride does not land on it.
- Sort all 21 recording IDs lexicographically. For recording index `i` in this full
  inventory and stride `s`, use permutation seed `20260720 + 1000*i + s`.
- Initialize each track row from the first selected anonymous frame. No semantic
  labels or fluorescence values enter association.
- Run all four already implemented center-only methods: `previous_hungarian`,
  `pairwise_affine`, `first_affine`, and `dual_anchor`.
- The primary comparator is locked as `first_affine`, selected previously on the
  four legacy evaluation recordings. It will not be changed after expansion results.

## Primary outcome and decision rule

- Primary outcome: seeded identity accuracy (`dual_anchor - first_affine`). This is
  persistent initialized ROI/neuron-instance accuracy, not conventional globally
  remapped IDF1.
- Within each untouched recording, calculate the paired accuracy difference at each
  of the five strides and average those five differences equally.
- Average the resulting 16 recording-level differences with equal recording weight.
- Form a 95% percentile paired recording-bootstrap interval by resampling the 16
  complete recordings with replacement 20,000 times using seed `20260720`.
- Replication criterion: the lower endpoint of that interval is greater than zero.
  The point estimate and interval are reported regardless of outcome.

## Frozen secondary analyses

1. Report the paired accuracy effect and all-transition identity-switch reduction at
   each stride. Interpret switches jointly with seeded identity accuracy.
2. For each recording, pool switches and neuron-transition opportunities across
   strides, then compare switch rates. Report an equal-recording paired bootstrap.
3. Apply the existing exact top-quartile identity-blind symmetric-cloud-motion rule
   within recording and stride; report high-motion accuracy and switch effects.
4. Report per-recording effects and the acquisition-date mean effects. Use a
   hierarchical bootstrap that resamples the six acquisition dates, then recordings
   within date, with 20,000 resamples and seed `20260721`.
5. Repeat the primary analysis after excluding
   `sub-20190925-01_ses-20190925_ophys`, the only untouched recording acquired on
   the tuning date. This is a declared sensitivity analysis, not a replacement
   primary analysis.
6. Report a comprehensive 20-nontuning-recording summary only as secondary, with the
   four legacy and 16 untouched cohorts visibly stratified.
7. The other two center-only methods are descriptive comparators. They cannot
   replace the locked primary baseline.

## Interpretation boundary

A successful replication supports transfer across animals and six dates inside one
EY/DANDI acquisition source. It does not establish cross-laboratory, cross-device,
freely moving, detector-error, merge/split, semantic-name, local-patch, confidence,
or downstream-signal performance.
