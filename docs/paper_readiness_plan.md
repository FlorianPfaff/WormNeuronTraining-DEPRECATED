# Neuron tracking paper readiness plan

Date: 3 October 2026. Status: proposed decision plan, not a frozen protocol or a
claim that further experiments have run.

The objective is to determine whether the project supports a useful, defensible
paper, not to obtain a positive PyRecEst result. The recommended main claim is
that combining temporal and immutable reference anchors protects persistent
neuron identity. PyRecEst provides a comparison of filtering, alternative
histories, lookahead, accuracy, continuity, and computational cost.

This is a narrower proposed study than the original Bayesian extended-object
tracking program. It does not retroactively satisfy that program's failed,
deviated, or untested targets. Preserve the original protocols and results.

## Evidence and limitations

The original internally frozen 16-recording comparison found +1.824 percentage
points of seeded identity accuracy and 69% fewer pooled switches against a
locked first-frame affine comparator. It established a within-source effect,
not superiority to published trackers or cross-laboratory generalization.
See the [existing claim audit](results_audit.md).

The [PyRecEst follow-up](../outputs/pyrecest_followup_20261002/report.md) tested
20 previously examined recordings. GNN gained 0.180 accuracy points with 1.6%
more switches and 7.5x runtime; lag-2 MHT gained 0.244 points with 3.7% more
switches, 54.9x runtime, and two future selected scans. These concurrent-run
timings need controlled benchmarking before making efficiency claims. There
were 500 evaluation tasks, not 500 independent biological replicates.

Historical baseline outcomes differed in 121/200 cases. The dual-anchor mean
accuracy shift was -0.021457 points, but the largest case shift was 0.742660
points. Same-environment comparisons remain informative; the discrepancy still
requires diagnosis. A mathematically equivalent GNN control matched assignment
hashes in 91/100 cases, so exact numerical equivalence is not established.

## Sequence and decision gates

| Stage | Question | Required output | Decision |
| --- | --- | --- | --- |
| 1. Trust and scope | Are identities, metrics, numerical behavior, and novelty credible? | Reproduction diagnosis, ground-truth audit, prior-work matrix | Fix material defects before new claims |
| 2. Competitive pilot | Does the anchor result survive stronger, fairly configured baselines? | Small development benchmark with validated adapters | Continue the methods route only for a credible practical advantage |
| 3. Mechanism | Does MHT contribute beyond filtering and future information? | Matched-information ablations and isolated timing | Keep only supported mechanism claims |
| 4. External confirmation | Does the locked result transfer to new animals from another source? | New frozen protocol and independent-source evaluation | Positive, inconclusive, or negative; no retuning on test outcomes |
| 5. Scientific utility | Does tracking improve the intended neural measurement? | Prespecified downstream endpoint on independent data | Required for a broad neuroscience-utility claim |
| 6. Publication decision | What paper, if any, does the evidence support? | Claim ledger, reproducible artifact, revised manuscript | Focused methods paper, benchmark report, or stop |

Stages 1 and external-data feasibility can proceed in parallel. Baseline smoke
tests can also start during that audit, but full comparisons wait for metric and
provenance checks. Stages 2 and 3 use already-seen development data. No external
test outcomes are inspected until the study design is locked.

## Stage 1 Establish trust and a distinct contribution

1. Reproduce selected discrepant and matching cases in the historical and
   PyRecEst environments. Include the worst discrepancy, not only average cases.
   Compare data hashes, permutations, frame selection, units, preprocessing,
   package/BLAS versions, affine estimates, costs, and assignment histories.
   Locate the first divergence and test it under controlled numerical settings.
   Do not assume tied assignments explain the differences without checking.
2. Add regression tests for every diagnosed issue and a test that the scorer
   counts known identity exchanges correctly. Repeat identical runs in the
   pinned environment. If platform-dependent ties are unavoidable, quantify
   their effects and test whether scientific conclusions survive them.
3. Audit the provenance of the released ROI identities. Determine which are
   manually checked versus algorithm-derived. Where images and expertise are
   available, have a reviewer blinded to method identity check a preselected
   mixture of random and ambiguous transitions, including switches and stable
   but wrong tracks. Report disagreement separately; never edit test labels to
   favor a tracker. If this cannot be established, limit the claim to agreement
   with released ROI identities rather than independently verified biological
   identity.
4. Refresh the literature comparison specifically for temporal plus immutable
   anchoring and fixed-cohort identity tracking. MHT and deformation correction
   are already combined in [EMC2](https://doi.org/10.1371/journal.pcbi.1009432).
   A PyRecEst integration alone is not a new tracking principle.

Deliverables: a numerical discrepancy diagnosis, ground-truth provenance table,
metric tests, and a claim-versus-prior-work table. Unresolved defects that can
change the conclusion block a methods submission; they do not justify deleting
unfavorable cases.

## Stage 2 Run a small competitive pilot

Use already-seen recordings for development and troubleshooting. Select pilot
recordings by documented diversity criteria, not where dual anchoring wins.
Begin with two or three recordings and representative strides before running
the complete development set.

For the coordinate-only association track, implement and validate:

- Existing temporal-only, reference-only, and dual-anchor trackers.
- Constant-velocity Kalman/Hungarian association with comparable tuning.
- Nonrigid CPD plus assignment; add TPS-based registration only if it tests a
  distinct mechanism or CPD is unsuitable. TPS must not receive test identity
  correspondences to fit its deformation.
- The released pretrained fDNC correspondence model. The adapter in
  `src/deformeot/wormid_fdnc.py` is a starting point, not a reproduced benchmark.
  Validate its official preprocessing, assignment direction, batching, units,
  native example performance, and training-data overlap. The separate
  `src/deformeot/fdnc.py` geometric experiments are not learned fDNC results.
- The already selected PyRecEst GNN and MHT variants, without further tuning
  against the existing evaluation outcomes for a claimed confirmation.

Keep centers, seeded identities, permitted observations, and scoring identical
inside each matched-input comparison. Use published defaults plus a small,
documented development-only tuning allowance for each applicable method. Do not
force image trackers into a coordinate-only ranking or handicap them by silently
removing their required inputs.

An optional separate image/pipeline comparison should include one well-matched
published tracker, chosen for data compatibility: [ZephIR](https://github.com/venkatachalamlab/ZephIR)
is an image-registration tracker; [WormND](https://github.com/focolab/WormND)
provides tracking workflows for 3DeeCellTracker and Ultrack. Report imagery,
annotations, training, manual corrections, detector access, and future-frame
access explicitly. An end-to-end comparison and an association-only comparison
answer different questions.

Deliverable: a comparison of accuracy, identity switches, fragmentation,
runtime, memory, and annotation burden with per-recording outcomes. Select the
primary external comparator on development data and lock it; retain all other
prespecified comparators in the final results.

Gate: if the anchor method has no credible advantage in continuity, accuracy,
cost, or annotation effort over stronger baselines, stop the superiority-paper
route. A reproducible benchmark or negative-result report remains possible,
but is not guaranteed publishable.

## Stage 3 Identify what actually helps

Run a compact factorial comparison with shared motion models, reference costs,
and candidate gating wherever possible:

- Temporal anchor only, immutable anchor only, and both anchors.
- Deterministic prediction versus recursive Gaussian filtering.
- Online single assignment versus online MHT implementation. The current
  zero-lag adapter commits every scan and therefore does not demonstrate the
  benefit of maintaining competing histories.
- Lag-2 MHT versus an actual three-scan window reassociation or fixed-lag
  single-path optimizer that can revise earlier assignments using the same two
  future scans. Merely delaying the output of a greedy beam-width-one tracker
  is not an information-matched smoother.
- A small prespecified beam-width/gating sensitivity on development data to
  assess whether the current four-hypothesis approximation limits performance.

Inspect identity error duration, stable wrong identities, and correction events
to explain how accuracy can improve while switch counts worsen. Case studies
must include failures and use a documented selection rule.

Measure wall time under isolated, equal-resource conditions, including warmup,
model loading separately from steady-state inference, repeated timings, thread
counts, GPU synchronization where relevant, and actual latency in seconds.

Gate: describe an MHT-specific benefit only if it survives the matched-lookahead
comparison. Otherwise retain PyRecEst as a trade-off result, not the headline.
Do not expand into learned deformation, visibility, calibrated uncertainty, and
extended-object tracking simultaneously. At most one additional mechanism
should be proposed after a diagnosed development failure motivates it; it then
needs a new frozen candidate and an untouched evaluation.

## Stage 4 Lock and run independent confirmation

First make a data-availability and exposure ledger. All 21 EY recordings and
the already inspected Targettrack, fDNC/NeRVE, and ASCENT examples are development
or historical evidence now. They cannot be relabeled untouched. Check for
duplicate animals, repeated recordings, source overlap, and baseline training
overlap, not just different filenames.

Candidate sources are previously unused non-EY WormID recordings, additional
public 3DeeCellTracker or ZephIR animals, and appropriate BrainAlignNet/ANTSUN
data. These are leads, not confirmed cohorts. Audit raw sequence availability,
identity labels, annotation provenance and coverage, frame rate, units, animal
count, licensing, and fluorescence availability before choosing. Registration
pairs alone, such as some [BrainAlignNet](https://github.com/flavell-lab/BrainAlignNet)
assets, cannot establish long-term tracking. A single new movie is a case study,
not a multi-animal generalization estimate.

Use development data to simulate the precision needed for a practically useful
effect, conservatively accounting for acquisition-date clustering. This planning
calculation must run on a calculation server. Choose the cohort size and complete
eligible inventory before test outcomes are inspected. Do not count frames or
five strides as independent animals or collect more animals merely until a
confidence interval becomes positive.

The new protocol must lock data roles, eligibility, preprocessing, every method
and tuning budget, source-code/environment hashes, endpoints, exclusions,
aggregation, analysis order, and practical decision thresholds. Preserve its
date/hash before evaluation; do not describe an internal freeze as external
preregistration. Any genuinely public registration needs a separate user action.

### Proposed primary decision rule

For a continuity-focused methods paper, the proposed endpoint is the paired
animal-level reduction in switches per 1,000 annotated transition opportunities
against the development-selected strong comparator. Use complete animal-level
resampling, with declared date/session sensitivity analyses. Equal-weight
animals rather than pooling all frames. Native acquisition rate is the proposed
primary condition; temporal strides are secondary stress analyses, summarized
within animal. A different primary sampling regime needs a scientific rationale
before the freeze, not selection after observing which stride favors a method.

Proposed practical gate: at least a 25% point-estimate reduction in the mean
switch rate, a paired 95% interval for the absolute reduction above zero, and
accuracy noninferiority with a lower 95% bound above -0.25 percentage points.
These are planning proposals, not established biological tolerances or acceptance
criteria. Justify or revise them before freezing, using downstream cost and
development-only precision estimates. Report the uncertainty of the relative
reduction too; a 25% point estimate does not establish a guaranteed 25% gain.
If baseline switches are near zero, this is not an informative superiority
endpoint; choose a justified endpoint before freezing rather than after results.

Report seeded identity accuracy and conventional identity metrics under their
correct definitions. Prespecify handling of initialization, missing assignments,
partial labels, fragmentation, coverage, and denominator opportunities. A tracker
must not earn a continuity advantage simply by dropping difficult neurons.
Report all planned outcomes; label secondary strata exploratory or apply the
declared multiple-comparison treatment. No selection among favorable endpoints
or methods after opening the test results.

Gate: a useful, precise external effect supports transfer to that source only.
A wide interval is inconclusive, not proof of equivalence. A failed independent
test remains failed even if a later modified model succeeds elsewhere.

## Stage 5 Test scientific utility without outcome switching

For an association-only paper, this is highly valuable; for a claim of improved
neural measurement or a broadly useful pipeline, make it mandatory. Data
feasibility must be established before freezing Stage 4 so the same eligible
independent cohort can provide both endpoints without later cherry-picking.

For each animal and method, compute raw-trace correlations against independently
curated reference identities and take their median across eligible neurons.
The proposed primary endpoint is the mean across animals of the paired difference
between these medians. Prespecify neuron eligibility and constant-trace handling.
A provisional meaningful threshold is +0.03,
retaining the earlier study's target unless a different measurement is justified
prospectively. Select one primary downstream endpoint, not a menu of correlation,
event F1, or decoding scores from which to report whichever improves.

Keep trace extraction, background correction, and normalization fixed across
trackers. Separate reassignment of known ROI signals, which isolates identity
errors, from extraction at predicted locations, which tests the full pipeline.
Do not use the activity being evaluated as an identity feature unless that input
is explicit and circularity controls are provided. Audit original pipeline trace
processing and any gaps it interpolates.

Real detector misses and clutter require a separate, shared predicted-detection
arm and detection/coverage metrics. Synthetic thinning and noise may explain
failure mechanisms but cannot substitute for real detector-error validation.
Extent claims require actual masks and an extent-versus-point ablation; local
patch features alone do not validate extended-object tracking.

## Stage 6 Decide the paper and preserve the record

- Focused methods paper: a distinct contribution, validated stronger comparators,
  reproducible results, and a useful independent-source effect. Keep scope
  association-only unless broader measurements were actually tested.
- Broader neuroscience methods paper: additionally demonstrate practical signal
  recovery and realistic detector/pipeline behavior across the relevant sources.
- Benchmark or negative-result paper: if superiority fails, assess whether the
  controlled accuracy/continuity/runtime findings provide a genuinely new lesson
  beyond existing benchmarks. Negative findings still need sound comparisons.
- Stop or release a technical report: if novelty is absent, reference identities
  are not reliable enough, compatible independent data are unavailable, or the
  effect disappears under fair controls. Do not tune indefinitely to rescue it.

Update the manuscript chronologically: retain the original 16-recording freeze,
label the PyRecEst follow-up as later work, and give new external tests their own
protocol and outcomes. Preserve earlier failed targets. Assemble a reproducible
release with manifests, checkpoints, exact commands, figures, limitations, and
third-party license notices. Authorship, licensing, public upload, and submission
remain decisions for the human authors; this plan does not authorize them.

## Execution scope and first work package

All numerical runs, tests, fitting, power/precision simulations, and scientific
plots belong on `gpuserver6000`, or `gpuserver4090` after connectivity and resource
availability are checked. Use persistent storage, pinned isolated environments,
per-case checkpoints, completion manifests, and regular local result copies.
Do not put the only evaluation copy in `/tmp`. Leave the laptop for inspection,
editing, and orchestration. Do not assume both servers are presently reachable
or that a GPU accelerates the NumPy tracking implementation.

The first work package is deliberately bounded: diagnose representative numerical
differences; audit label provenance; inventory up to three candidate external
cohorts; validate the existing fDNC adapter and one nonrigid baseline on a small
development pilot. Produce a feasibility report before a full benchmark or
large raw-data transfer. Estimate compute/storage from those smoke tests and
agree any material expansion before proceeding. Stop debugging a blocked
baseline after a documented bounded attempt; report missing competitive evidence
rather than silently substituting an easy opponent.

This turn creates the plan only. It starts no server jobs, downloads, training,
external messages, public registration, or submission.
