# Primary-literature source dossier

## Scope and source policy

Search cutoff: **2026-07-20**. All links and repository states below were
accessed on **2026-07-20** unless a different date is stated. This was a
targeted, claim-driven search rather than a registered systematic review. It
used primary papers, publisher records, author-maintained repositories, and
official data records; secondary summaries were used only to locate primary
sources and are not cited as evidence.

The literature separates three ideas that are easy to conflate:

1. deformable registration or correspondence of neuron centers;
2. tracking segmented objects or their spatial extent; and
3. a temporal probability distribution over alternative identity and
   visibility histories.

Finding one of these in a paper is not evidence that the paper supplies the
other two.

## Defensible novelty boundary

The strongest source-supported wording is deliberately conjunctive and must be
reported as a search result, not as proof of absence:

> In a targeted primary-literature search through 20 July 2026, we found no
> demonstration in freely moving adult *C. elegans* whole-brain microscopy of
> a deformation-aware Bayesian tracker that maintains a calibrated joint
> temporal posterior over canonical identity, visibility, 3D position, and
> dynamic extent while retaining alternative hypotheses through missed,
> merged, and split detections.

For the empirically successful, simpler method in this project, an even more
results-facing boundary is appropriate:

> In the primary sources reviewed through 20 July 2026, we found no adult
> *C. elegans* whole-brain tracking demonstration that isolates local 3D object
> extent as an identity cue in an identity-blind held-out test and combines it
> with both a temporal anchor and an immutable reference anchor.

Neither sentence establishes that an individual component is new. Targettrack
tracks 3D neuron volumes; CELLECT estimates coarse cell radius and uses size
differences for linkage; NeRVE, ZephIR, ASCENT, and BrainAlignNet/ANTSUN handle
deformation; DSMM returns pairwise correspondence probabilities; MHHT and eMHT
retain multiple tracking hypotheses; and recent cell trackers quantify link
uncertainty. The first sentence should only describe a method that actually
implements and evaluates every conjunct. It should not be used to imply that a
point estimate from the present dual-anchor method is a Bayesian joint
posterior.

## Claim-separation matrix

| Work | Relevant biological regime | Deformation handling | Temporal alternatives | Object extent | Calibrated association uncertainty | What it rules out as a novelty claim |
|---|---|---|---|---|---|---|
| NeRVE | Freely moving adult worm brain | straightening, non-rigid point-set registration, reference clustering | no explicit temporal posterior | points | no | deformation-aware adult neuron tracking |
| fDNC | Within- and cross-worm neuron correspondence, including NeRVE | learned pairwise point-cloud correspondence | pairwise prediction, not a temporal hypothesis posterior | points | softmax-like scores, not demonstrated calibration | learned probabilistic-looking correspondence scores |
| ZephIR | Freely behaving adult worm whole-brain movie | deformable image registration, spring network, feature detection | deterministic optimization from selected references | fixed keypoints | no | reference-anchored deformation tracking with sparse labels |
| ASCENT | 3D neuron tracking; evaluated on NeRVE and other recordings | deformation-augmented self-supervised embeddings | Hungarian/bipartite links, no maintained hypothesis set | detected candidates/points | no calibration demonstration | annotation-free embeddings plus assignment |
| Targettrack | Freely moving adult worm, keypoints and 3D neuron volumes | targeted deformation augmentation | semantic prediction/proofreading, not a Bayesian temporal posterior | **yes, 3D masks** | no | 3D neuron-volume tracking and extent-aware image prediction |
| BrainAlignNet/ANTSUN | Freely moving adult worm brain | learned non-rigid dense displacement fields plus ROI linking/clustering | similarity matrix and clustering, not a joint posterior | ROIs, not a dynamic extent state | no calibration demonstration | learned image registration and high-accuracy ROI identity linking |
| DSMM | Generic 2D/3D point-set registration | non-rigid Student's-*t* mixture model | no temporal model | points | posterior membership/correspondence probabilities, not temporal calibration | Bayesian pairwise correspondence probabilities |
| MHHT | Late-stage *C. elegans* **embryo** seam cells | relational posture model | **yes, multiple posture hypotheses** | point detections | failure-cost threshold, not calibrated probabilities | multi-hypothesis *C. elegans* cell tracking |
| eMHT + EMC2 | Biological particles; Hydra and mouse neural imaging | eMHT tracklets plus thin-plate-spline field deformation and gap closing | **yes** within eMHT tracklets | spots | probabilistic tracker, no reported calibration of the complete stitched tracks | neural tracking with MHT, deformation, and gap stitching |
| CELLECT | *C. elegans* embryos and other 3D tissues; Drosophila neural imaging | learned embeddings under non-rigid motion | mostly adjacent-frame linking; optional postprocessing | coarse radius/segmentation | linkage reliability scores, not reported calibrated probabilities | size-assisted linkage, reliability scores, and deformation-robust neural tracking |
| Paul et al.; Betjes et al. | General cell tracking and organoids | method-dependent | assignment alternatives or graph marginalization | no dynamic extent posterior | **yes** | calibrated/link-level cell-tracking uncertainty |
| Random-matrix/LRFS extended-target tracking | Radar/sensor tracking | kinematics and extent dynamics | Bayesian multi-target state | **yes, stochastic extent** | posterior uncertainty by model | joint Bayesian kinematics, identity labels, and extent in the tracking literature |

## Adult *C. elegans* whole-brain tracking sources

### NeRVE

**Verified citation.** Jeffrey P. Nguyen, Ashley N. Linder, George S.
Plummer, Joshua W. Shaevitz, and Andrew M. Leifer. “Automatically tracking
neurons in a moving and deforming brain.” *PLOS Computational Biology* 13(5),
e1005517 (2017). [DOI](https://doi.org/10.1371/journal.pcbi.1005517).

- Supports: a full pipeline for freely moving adult-worm brain imaging;
  straightening, segmentation, non-rigid point-set registration to many
  reference volumes, Neuron Registration Vector Encoding, clustering, and
  thin-plate-spline correction. The paper reports tracking approximately 150
  neurons through an eight-minute recording.
- Does not support: a calibrated temporal posterior, retained alternative
  identity histories, or an explicit stochastic extent state.
- Official artifacts: [analysis code](https://github.com/leiferlab/NeRVEclustering)
  and [IEEE DataPort dataset, DOI 10.21227/H2901H](https://doi.org/10.21227/H2901H).
  The author-maintained code repository explicitly identifies this paper and
  data DOI.

### fDNC

**Verified citation.** Xinwei Yu, Matthew S. Creamer, Francesco Randi, Anuj K.
Sharma, Scott W. Linderman, and Andrew M. Leifer. “Fast deep neural
correspondence for tracking and identifying neurons in *C. elegans* using
semi-synthetic training.” *eLife* 10, e66410 (2021).
[DOI](https://doi.org/10.7554/eLife.66410).

- Supports: transformer-based pairwise correspondence from point clouds,
  trained on semi-synthetic worms and used both within an individual over time
  and across individuals; the NeRVE recording is an evaluation source.
- Does not support: a sequence-level posterior over tracks or visibility,
  dynamic extent, or empirical calibration of correspondence probabilities.
- Official artifacts: [code](https://github.com/XinweiYu/fDNC_Neuron_ID) and
  [data plus pretrained model](https://osf.io/t7dzu/), linked by the repository.

### ZephIR

**Verified citation.** James Ryu, Amin Nejatbakhsh, Mahdi Torkashvand, Sahana
Gangadharan, Maedeh Seyedolmohadesin, Jinmahn Kim, Liam Paninski, and Vivek
Venkatachalam. “Versatile multiple object tracking in sparse 2D/3D videos via
deformable image registration.” *PLOS Computational Biology* 20(5), e1012075
(2024). [DOI](https://doi.org/10.1371/journal.pcbi.1012075).

- Supports: deformation-aware tracking of a fixed set of keypoints using local
  image registration, a spring-network loss, feature detection, selected
  reference frames, and optional partial annotations. It includes a freely
  behaving adult-worm whole-brain recording and an evaluation on the 79-neuron
  NeRVE benchmark.
- Does not support: stochastic extent, a Bayesian multi-hypothesis state, or
  calibrated link probabilities. The fixed-keypoint formulation can maintain
  an approximate location for temporarily invisible points through neighboring
  springs, but this is not a visibility posterior.
- Official artifacts: [code](https://github.com/venkatachalamlab/ZephIR) and
  [whole-brain data, DOI 10.5281/zenodo.10818810](https://doi.org/10.5281/zenodo.10818810).

### ASCENT

There are related but non-identical ASCENT records; citations and metrics must
be tied to the precise version used.

**Peer-reviewed conference version.** Haejun Han and Hang Lu. “ASCENT:
Annotation-Free Self-Supervised Contrastive Embeddings for 3D Neuron Tracking
in Fluorescence Microscopy.” *IEEE/CVF ICCV 2025*, 14676–14687.
[DOI](https://doi.org/10.1109/ICCV51701.2025.01362) and
[open paper](https://openaccess.thecvf.com/content/ICCV2025/html/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.html).

**NETr/preprint version used by the current repository.** Haejun Han, Robert
Pritchard, and Hang Lu. “ASCENT: Annotation-free Self-supervised Contrastive
Embeddings for 3D Neuron Tracking in Fluorescence Microscopy.” bioRxiv (2025).
[DOI](https://doi.org/10.1101/2025.07.23.666425). A later manuscript version is
“Annotation-free whole-brain neuron tracking via transformer-based
self-supervised learning,” Research Square v1 (2026),
[DOI](https://doi.org/10.21203/rs.3.rs-8080516/v1).

- Supports: self-supervised image/position embeddings and fast bipartite or
  Hungarian linking without manual track labels. The public implementation
  describes NETr feature extraction followed by cosine-similarity Hungarian
  assignment and provides checkpoints for NeRVE and device-restricted data.
- Does not support: retained K-best identity histories, a calibrated posterior,
  or a dynamic extent state. Its detector and linker are modular.
- Official artifacts: [code](https://github.com/lu-lab/ascent) and
  [device-restricted Opterra data record](https://zenodo.org/records/17561700).
  The latter contains the 1,100-volume movie and ground-truth/predicted
  detection and track tables used in this project. It is device-restricted,
  not a freely moving benchmark.

### Targettrack

**Verified citation.** Core Francisco Park, Mahsa Barzegar-Keshteli, Kseniia
Korchagina, Ariane Delrocq, Vladislav Susoy, Corinne L. Jones, Aravinthan D. T.
Samuel, and Sahand Jamal Rahi. “Automated neuron tracking inside moving and
deforming *C. elegans* using deep learning and targeted augmentation.” *Nature
Methods* 21(1), 142–149 (2024; published online 5 December 2023).
[DOI](https://doi.org/10.1038/s41592-023-02096-3).

- Supports: end-to-end segmentation/tracking in freely moving worms, targeted
  augmentation learned from a few annotated postures, and demonstrations with
  both keypoint and **3D-volume** neuron labels. It is the closest direct prior
  art for tracking neuron extent in this organism.
- Does not support: a random-matrix-like dynamic extent posterior, K-best
  temporal hypotheses, calibrated assignment uncertainty, or explicit
  persistence of merge/split hypotheses.
- Official artifacts: [code](https://github.com/rahi-lab/targettrack) and
  [keypoint/volumetric sample data, DOI 10.5281/zenodo.10008744](https://doi.org/10.5281/zenodo.10008744).

### ANTSUN and BrainAlignNet

**ANTSUN 1.x source.** Adam A. Atanas et al. “Brain-wide representations of
behavior spanning multiple timescales and states in *C. elegans*.” *Cell*
186(19), 4134–4151.e31 (2023).
[DOI](https://doi.org/10.1016/j.cell.2023.07.035).

- Supports: the Automatic Neuron Tracking System for Unconstrained Nematodes
  (ANTSUN), which preprocesses dual-channel freely moving whole-brain movies,
  segments marker-channel neurons, registers volumes, links ROIs, and extracts
  normalized calcium traces.

**BrainAlignNet/ANTSUN 2.0 source.** Adam A. Atanas, Alicia Kun-Yang Lu, Brian
Goodell, Jungsoo Kim, Saba N. Baskoylu, Di Kang, Talya S. Kramer, Eric Bueno,
Flossie K. Wan, Karen L. Cunningham, Brandon Weissbourd, and Steven W. Flavell.
“Deep neural networks to register and annotate cells in moving and deforming
nervous systems.” *eLife* 14, RP108159 (2026).
[DOI](https://doi.org/10.7554/eLife.108159).

- Supports: BrainAlignNet, a semi-supervised learned dense displacement field
  for non-rigid alignment of paired freely moving worm-head volumes. In ANTSUN
  2.0 it feeds ROI similarity and identity clustering; the paper reports 99.6%
  neuron-linking accuracy and also transfers the registration architecture to
  deforming jellyfish.
- Does not support: stochastic dynamic object extent, a calibrated joint
  identity/visibility posterior, or retained temporal assignment hypotheses.
- Official artifacts: [BrainAlignNet](https://github.com/flavell-lab/BrainAlignNet),
  [ANTSUN](https://github.com/flavell-lab/ANTSUN), and
  [data/model checkpoints, DOI 10.7910/DVN/8UE0H9](https://doi.org/10.7910/DVN/8UE0H9).

## WormID-Bench and the EY data used here

### Benchmark paper

**Verified current metadata.** Jason Adhinarta, Jizheng Dong, Tianxiao He,
Junxiang Huang, Daniel Sprague, Jia Wan, Hyun Jee Lee, Zikai Yu, Peng Liu, Hang
Lu, Eviatar Yemini, Saul Kato, Erdem Varol, and Donglai Wei. “WormID-Bench: A
Benchmark for Whole-Brain Activity Extraction in *C. elegans*.” bioRxiv (2025),
version 4. [DOI](https://doi.org/10.1101/2025.01.06.631621).

- Supports: a standardized multi-laboratory benchmark spanning neuron
  detection, identification, and tracking, assembled from 118 worms from five
  groups. It is a benchmark source, not itself a new temporal Bayesian tracker.
- Official implementation: [focolab/WormND](https://github.com/focolab/WormND).

### WormID community data paper

**Verified citation.** Daniel Y. Sprague et al. “Unifying community whole-brain
imaging datasets enables robust neuron identification and reveals determinants
of neuron position in *C. elegans*.” *Cell Reports Methods* 5(1), 100964 (2025).
[DOI](https://doi.org/10.1016/j.crmeth.2024.100964).

- Supports: provenance and harmonization of the multi-laboratory WormID/NWB
  resource and its use for detection, identification, tracking, activity
  extraction, and cross-dataset analyses.

### Exact EY DANDI record

**Verified data citation.** Eviatar Yemini, Amin Nejatbakhsh, Vivek
Venkatachalam, Albert Lin, Daniel Sprague, Aravinthan Samuel, and Oliver Hobert.
“NeuroPAL Microfluidic Chip Images and GCaMP activity,” version
0.241009.1457. DANDI Archive (2024).
[DOI](https://doi.org/10.48324/dandi.000541/0.241009.1457).

- This is the official source of the `sub-20190925-*` and `sub-20191030-*` EY
  NWB files used in the present benchmark. The API record contains 21 NWB
  subjects and identifies the data as NeuroPAL microfluidic-chip images and
  GCaMP activity.
- Important scope constraint: this EY source is device-restricted or
  semi-immobilized, so results on this subset do not alone establish a claim
  about freely moving animals.

## Closest probabilistic, multi-hypothesis, extent, and uncertainty work

### DSMM: pairwise correspondence probabilities

**Verified citation.** Zhiyong Zhou, Jianfei Tu, Chen Geng, Jisu Hu, Baotong
Tong, Jiansong Ji, and Yakang Dai. “Accurate and Robust Non-rigid Point Set
Registration using Student's-*t* Mixture Model with Prior Probability
Modeling.” *Scientific Reports* 8, 8742 (2018).
[DOI](https://doi.org/10.1038/s41598-018-26288-6).

- Supports: non-rigid 2D/3D point-set registration under missing
  correspondences and outliers; an EM formulation estimates posterior mixture
  membership that the paper explicitly interprets as pairwise correspondence
  probabilities.
- Does not support: temporal tracking, neuron visibility, object extent, or
  held-out calibration of those probabilities. DSMM is not a cell tracker.

### MHHT: multiple hypotheses in embryonic *C. elegans*

**Standalone method record.** Andrew Lauziere, Evan Ardiel, Stephen Xu, and
Hari Shroff. “Multiple Hypothesis Hypergraph Tracking for Posture
Identification in Embryonic *Caenorhabditis elegans*.” arXiv:2111.06425 (2021).
[DOI](https://doi.org/10.48550/arXiv.2111.06425).

**Peer-reviewed biological application.** Evan L. Ardiel, Andrew Lauziere,
Stephen Xu, Brandon J. Harvey, Ryan Patrick Christensen, Stephen Nurrish,
Joshua M. Kaplan, and Hari Shroff. “Stereotyped behavioral maturation and
rhythmic quiescence in *C. elegans* embryos.” *eLife* 11, e76836 (2022).
[DOI](https://doi.org/10.7554/eLife.76836).

- Supports: an extension of MHT with hypergraphs that ranks whole-posture
  hypotheses using correlated seam-cell motion amid false positives, false
  negatives, and sporadic embryonic twitching.
- Does not support: freely moving adult whole-brain neurons, 3D neuron extent,
  or calibrated posterior probabilities. Its error-warning cost threshold is
  not a calibration result.
- Official artifact: [MHHT code](https://github.com/lauziere/MHHT), linked in
  the eLife data-availability statement.
- Publication caveat: no separate peer-reviewed paper with the exact MHHT
  title was found; the exact-title source is arXiv, while the algorithm and
  quantitative application are peer-reviewed within the 2022 eLife paper.

### eMHT and EMC2: hypotheses, deformation, and gap stitching

**eMHT source.** Nicolas Chenouard, Isabelle Bloch, and Jean-Christophe
Olivo-Marin. “Multiple Hypothesis Tracking for Cluttered Biological Image
Sequences.” *IEEE Transactions on Pattern Analysis and Machine Intelligence*
35(11), 2736–2750 (2013).
[DOI](https://doi.org/10.1109/TPAMI.2013.97).

**EMC2 source.** Thibault Lagache, Alison Hanson, Jesús E. Pérez-Ortega,
Adrienne Fairhall, and Rafael Yuste. “Tracking calcium dynamics from individual
neurons in behaving animals.” *PLOS Computational Biology* 17(10), e1009432
(2021). [DOI](https://doi.org/10.1371/journal.pcbi.1009432).

- eMHT supports a probabilistic, multiframe MHT for thousands of fluorescent
  biological particles under clutter and missed detections.
- EMC2 first creates eMHT tracklets, estimates an elastic field from those
  tracklets with thin-plate splines, propagates endpoints through the field,
  and globally stitches compatible tracklets across intermittent visibility.
  It was validated on calcium-imaging data from behaving Hydra and multi-day
  mouse visual cortex, not freely moving adult-worm whole-brain movies.
- EMC2 does not jointly propagate a stochastic extent state, and the complete
  stitched trajectories are not accompanied by a reported calibration test.

### CELLECT: size, linkage reliability, and deforming neural tissue

**Verified citation.** Hongyu Zhou, Seonghoon Kim, Zhifeng Zhao, Jiaqi Fan, Wen
Huang, Xinghua Sui, Lizhi Shao, Haoran An, Jing-Ren Zhang, Jiamin Wu, and
Qionghai Dai. “CELLECT: contrastive embedding learning for large-scale
efficient cell tracking.” *Nature Methods* 22(11), 2411–2422 (2025).
[DOI](https://doi.org/10.1038/s41592-025-02886-x).

- Supports: adjacent-frame 3D cell tracking from center confidence, latent
  embeddings, coarse radius/segmentation, size-difference features, division
  scores, a `none` candidate, and human-in-the-loop linkage reliability scores.
  It evaluates *C. elegans* **embryo** datasets and demonstrates calcium-trace
  extraction in strongly deforming Drosophila brain tissue.
- Does not support: adult-worm whole-brain tracking, a long-horizon joint
  posterior, or calibrated uncertainty. The paper contains no calibration or
  uncertainty evaluation; it explicitly notes that its default model uses two
  adjacent frames and has limited long-term temporal context. “Reliability
  score” should therefore not be rewritten as “calibrated probability.”

### Cell-tracking uncertainty

**Verified citation.** Richard D. Paul, Johannes Seiffarth, David Rügamer,
Katharina Nöh, and Hanno Scharr. “How to Make Your Cell Tracker Say ‘I
Dunno!’” *IEEE/CVF ICCV 2025*, 6914–6923.
[DOI](https://doi.org/10.1109/ICCV51701.2025.00650) and
[open paper](https://openaccess.thecvf.com/content/ICCV2025/html/Paul_How_To_Make_Your_Cell_Tracker_Say_I_dunno_ICCV_2025_paper.html).

- Supports: post-hoc uncertainty quantification for linear-assignment cell
  trackers from Bayesian and classification perspectives, with empirical
  calibration evaluations across several trackers.
- Does not support: worm-brain deformation, dynamic object extent, or a joint
  long-horizon identity/visibility posterior.

**Verified citation.** Max A. Betjes, Rutger N. U. Kok, Sander J. Tans, and
Jeroen S. van Zon. “Cell tracking with accurate error prediction.” *Nature
Methods* 22(11), 2400–2410 (2025).
[DOI](https://doi.org/10.1038/s41592-025-02845-6).

- Supports: calibrated local link/division probabilities in OrganoidTracker
  2.0 and context-aware link error probabilities obtained by marginalizing
  alternative graph solutions; the work uses these probabilities for selective
  manual review and high-confidence analyses.
- Does not support: deforming adult-worm whole-brain imaging or dynamic extent.

### Extended-object tracking foundations

**Verified citation.** J. Wolfgang Koch. “Bayesian approach to extended object
and cluster tracking using random matrices.” *IEEE Transactions on Aerospace
and Electronic Systems* 44(3), 1042–1059 (2008).
[DOI](https://doi.org/10.1109/TAES.2008.4655362).

- Supports: the foundational random-matrix formulation in which kinematics and
  an ellipsoidal extent matrix are estimated jointly from multiple measurements
  generated by one extended target.

**Verified citation.** Michael Beard, Stephan Reuter, Karl Granström, Ba-Tuong
Vo, Ba-Ngu Vo, and Alexander Scheel. “Multiple Extended Target Tracking With
Labeled Random Finite Sets.” *IEEE Transactions on Signal Processing* 64(7),
1638–1653 (2016).
[DOI](https://doi.org/10.1109/TSP.2015.2505683).

- Supports: Bayesian multi-target tracking with persistent labels, uncertain
  target number, kinematics, measurement rate, and random-matrix extent.
- Neither paper is a microscopy or neuron-tracking demonstration. They establish
  that joint Bayesian identity/kinematics/extent modeling is not itself new;
  the biological deformation and observation model are the application gap.

## Official software/data ledger

Repository hashes are snapshots observed on the access date, not immutable
paper releases unless the paper explicitly identifies a version.

| Artifact | Official location | Verified snapshot or record |
|---|---|---|
| WormID-Bench code | <https://github.com/focolab/WormND> | `0c74e9ddffee138cfc6330ab65c0e690e8e199dd` |
| EY data | <https://doi.org/10.48324/dandi.000541/0.241009.1457> | DANDI version `0.241009.1457` |
| ASCENT code | <https://github.com/lu-lab/ascent> | local clone `cc885c7352f5913f310196fb99d88a0d17d057d5` |
| ASCENT Opterra data | <https://zenodo.org/records/17561700> | record 17561700, version 1.0 |
| Targettrack code | <https://github.com/rahi-lab/targettrack> | local clone `9310034cc939f5632a10ac82345eb0351a28023c` |
| Targettrack data | <https://doi.org/10.5281/zenodo.10008744> | record 10008744 |
| fDNC code | <https://github.com/XinweiYu/fDNC_Neuron_ID> | local clone `cbd2b1b2e623c22fe1921562afed1967fe58e139` |
| fDNC model/data | <https://osf.io/t7dzu/> | official OSF project linked by repository |
| NeRVE code | <https://github.com/leiferlab/NeRVEclustering> | local clone `1044c47e51ce42133b5b021acd0055fb3c1c6f8d` |
| NeRVE data | <https://doi.org/10.21227/H2901H> | IEEE DataPort record |
| ZephIR code | <https://github.com/venkatachalamlab/ZephIR> | `2842342b039c3aed8dab3b96b3ac0708e950e5b7` |
| ZephIR data | <https://doi.org/10.5281/zenodo.10818810> | record 10818810, version 1.0 |
| BrainAlignNet code | <https://github.com/flavell-lab/BrainAlignNet> | `5c6e8456e913ed637fd2e550238ff3e0ab0bf119` |
| ANTSUN code | <https://github.com/flavell-lab/ANTSUN> | `47c1d667291b8e73960d726686b31c232c2e6ace` |
| BrainAlignNet data/models | <https://doi.org/10.7910/DVN/8UE0H9> | Harvard Dataverse record |
| MHHT code | <https://github.com/lauziere/MHHT> | official repository linked by eLife |

## Metadata and interpretation caveats

1. **ASCENT versioning.** The ICCV paper, bioRxiv preprint, Research Square
   manuscript, live repository, and Opterra record overlap but are not
   interchangeable. The Research Square PDF lists authors as Haejun Han, Robert
   Pritchard, and Hang Lu, whereas Crossref currently returns Lu, Han, and
   Pritchard. Author order above follows the manuscript PDF. Reported metrics
   should cite the exact paper/repository revision that produced them.
2. **ASCENT data DOI.** Zenodo record 17561700 exposes the associated bioRxiv
   DOI (`10.1101/2025.07.23.666425`) rather than a distinct Zenodo DOI. Cite the
   stable record URL for the dataset.
3. **WormID-Bench revisions.** Current Crossref metadata contains 14 authors and
   the “WormID-Bench” title; the bioRxiv API's older entries contain a shorter
   author list and earlier “WormID-Benchmark” title. The dossier uses current
   Crossref metadata and identifies version 4.
4. **Targettrack year.** It was published online in December 2023 and appears in
   the January 2024 volume. Use 2024 for the volume citation, optionally noting
   the online date.
5. **MHHT publication status.** The exact-title method manuscript remains an
   arXiv record in the sources found. Its use and evaluation are peer-reviewed
   in Ardiel et al. (2022).
6. **Probability is not calibration.** DSMM posterior memberships, fDNC output
   scores, CELLECT similarity/reliability scores, and generic neural-network
   confidences should not be called calibrated unless an empirical calibration
   analysis supports that statement.
7. **Absence claims.** Search coverage cannot prove that no paper exists. Keep
   “we found no demonstration,” name the cutoff, preserve the conjunctive
   boundary, and update the search before submission if the manuscript is
   submitted after July 2026.
8. **eMHT pagination.** The IEEE paper and
   [PubMed record](https://pubmed.ncbi.nlm.nih.gov/23689865/) give pages
   2736–2750; Crossref currently returns the apparent typo 2736–3750. The
   citation and BibTeX record use the publisher/PubMed range.

## BibTeX-ready records

```bibtex
@article{adhinarta2025wormidbench,
  author  = {Jason Adhinarta and Jizheng Dong and Tianxiao He and Junxiang Huang and Daniel Sprague and Jia Wan and Hyun Jee Lee and Zikai Yu and Peng Liu and Hang Lu and Eviatar Yemini and Saul Kato and Erdem Varol and Donglai Wei},
  title   = {{WormID-Bench}: A Benchmark for Whole-Brain Activity Extraction in {C. elegans}},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.01.06.631621},
  note    = {Version 4}
}

@article{sprague2025wormid,
  author  = {Daniel Y. Sprague and Kevin Rusch and Raymond L. Dunn and Jackson M. Borchardt and Steven Ban and Greg Bubnis and Grace C. Chiu and Chentao Wen and Ryoga Suzuki and Shivesh Chaudhary and Hyun Jee Lee and Zikai Yu and Benjamin Dichter and Ryan Ly and Shuichi Onami and Hang Lu and Koutarou D. Kimura and Eviatar Yemini and Saul Kato},
  title   = {Unifying community whole-brain imaging datasets enables robust neuron identification and reveals determinants of neuron position in {C. elegans}},
  journal = {Cell Reports Methods},
  year    = {2025},
  volume  = {5},
  number  = {1},
  pages   = {100964},
  doi     = {10.1016/j.crmeth.2024.100964}
}

@misc{yemini2024eydata,
  author       = {Eviatar Yemini and Amin Nejatbakhsh and Vivek Venkatachalam and Albert Lin and Daniel Sprague and Aravinthan Samuel and Oliver Hobert},
  title        = {{NeuroPAL} Microfluidic Chip Images and {GCaMP} Activity},
  year         = {2024},
  howpublished = {DANDI Archive, version 0.241009.1457},
  doi          = {10.48324/dandi.000541/0.241009.1457}
}

@inproceedings{han2025ascenticcv,
  author    = {Haejun Han and Hang Lu},
  title     = {{ASCENT}: Annotation-Free Self-Supervised Contrastive Embeddings for 3D Neuron Tracking in Fluorescence Microscopy},
  booktitle = {2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  pages     = {14676--14687},
  doi       = {10.1109/ICCV51701.2025.01362}
}

@article{han2025ascentpreprint,
  author  = {Haejun Han and Robert Pritchard and Hang Lu},
  title   = {{ASCENT}: Annotation-free Self-supervised Contrastive Embeddings for 3D Neuron Tracking in Fluorescence Microscopy},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.07.23.666425}
}

@article{han2026ascentresearchsquare,
  author  = {Haejun Han and Robert Pritchard and Hang Lu},
  title   = {Annotation-free whole-brain neuron tracking via transformer-based self-supervised learning},
  journal = {Research Square},
  year    = {2026},
  doi     = {10.21203/rs.3.rs-8080516/v1},
  note    = {Author order taken from the manuscript PDF}
}

@misc{pritchard2025ascentdata,
  author       = {Robert Pritchard and Haejun Han and Hang Lu},
  title        = {A device-restricted {C. elegans} whole-brain recording dataset for {ASCENT}: annotation-free neuron tracking benchmark},
  year         = {2025},
  howpublished = {Zenodo record 17561700, version 1.0},
  url          = {https://zenodo.org/records/17561700}
}

@article{park2024targettrack,
  author  = {Core Francisco Park and Mahsa Barzegar-Keshteli and Kseniia Korchagina and Ariane Delrocq and Vladislav Susoy and Corinne L. Jones and Aravinthan D. T. Samuel and Sahand Jamal Rahi},
  title   = {Automated neuron tracking inside moving and deforming {C. elegans} using deep learning and targeted augmentation},
  journal = {Nature Methods},
  year    = {2024},
  volume  = {21},
  number  = {1},
  pages   = {142--149},
  doi     = {10.1038/s41592-023-02096-3},
  note    = {Published online 5 December 2023}
}

@article{nguyen2017nerve,
  author  = {Jeffrey P. Nguyen and Ashley N. Linder and George S. Plummer and Joshua W. Shaevitz and Andrew M. Leifer},
  title   = {Automatically tracking neurons in a moving and deforming brain},
  journal = {PLOS Computational Biology},
  year    = {2017},
  volume  = {13},
  number  = {5},
  pages   = {e1005517},
  doi     = {10.1371/journal.pcbi.1005517}
}

@article{yu2021fdnc,
  author  = {Xinwei Yu and Matthew S. Creamer and Francesco Randi and Anuj K. Sharma and Scott W. Linderman and Andrew M. Leifer},
  title   = {Fast deep neural correspondence for tracking and identifying neurons in {C. elegans} using semi-synthetic training},
  journal = {eLife},
  year    = {2021},
  volume  = {10},
  pages   = {e66410},
  doi     = {10.7554/eLife.66410}
}

@article{ryu2024zephir,
  author  = {James Ryu and Amin Nejatbakhsh and Mahdi Torkashvand and Sahana Gangadharan and Maedeh Seyedolmohadesin and Jinmahn Kim and Liam Paninski and Vivek Venkatachalam},
  title   = {Versatile multiple object tracking in sparse 2D/3D videos via deformable image registration},
  journal = {PLOS Computational Biology},
  year    = {2024},
  volume  = {20},
  number  = {5},
  pages   = {e1012075},
  doi     = {10.1371/journal.pcbi.1012075}
}

@article{atanas2023antsun,
  author  = {Adam A. Atanas and Jungsoo Kim and Ziyu Wang and Eric Bueno and McCoy Becker and Di Kang and Jungyeon Park and Talya S. Kramer and Flossie K. Wan and Saba Baskoylu and Ugur Dag and Elpiniki Kalogeropoulou and Matthew A. Gomes and Cassi Estrem and Netta Cohen and Vikash K. Mansinghka and Steven W. Flavell},
  title   = {Brain-wide representations of behavior spanning multiple timescales and states in {C. elegans}},
  journal = {Cell},
  year    = {2023},
  volume  = {186},
  number  = {19},
  pages   = {4134--4151.e31},
  doi     = {10.1016/j.cell.2023.07.035}
}

@article{atanas2026brainalignnet,
  author  = {Adam A. Atanas and Alicia Kun-Yang Lu and Brian Goodell and Jungsoo Kim and Saba N. Baskoylu and Di Kang and Talya S. Kramer and Eric Bueno and Flossie K. Wan and Karen L. Cunningham and Brandon Weissbourd and Steven W. Flavell},
  title   = {Deep neural networks to register and annotate cells in moving and deforming nervous systems},
  journal = {eLife},
  year    = {2026},
  volume  = {14},
  pages   = {RP108159},
  doi     = {10.7554/eLife.108159}
}

@article{zhou2018dsmm,
  author  = {Zhiyong Zhou and Jianfei Tu and Chen Geng and Jisu Hu and Baotong Tong and Jiansong Ji and Yakang Dai},
  title   = {Accurate and Robust Non-rigid Point Set Registration using Student's-t Mixture Model with Prior Probability Modeling},
  journal = {Scientific Reports},
  year    = {2018},
  volume  = {8},
  pages   = {8742},
  doi     = {10.1038/s41598-018-26288-6}
}

@article{lauziere2021mhht,
  author  = {Andrew Lauziere and Evan Ardiel and Stephen Xu and Hari Shroff},
  title   = {Multiple Hypothesis Hypergraph Tracking for Posture Identification in Embryonic {Caenorhabditis elegans}},
  journal = {arXiv preprint arXiv:2111.06425},
  year    = {2021},
  doi     = {10.48550/arXiv.2111.06425}
}

@article{ardiel2022embryos,
  author  = {Evan L. Ardiel and Andrew Lauziere and Stephen Xu and Brandon J. Harvey and Ryan Patrick Christensen and Stephen Nurrish and Joshua M. Kaplan and Hari Shroff},
  title   = {Stereotyped behavioral maturation and rhythmic quiescence in {C. elegans} embryos},
  journal = {eLife},
  year    = {2022},
  volume  = {11},
  pages   = {e76836},
  doi     = {10.7554/eLife.76836}
}

@article{chenouard2013emht,
  author  = {Nicolas Chenouard and Isabelle Bloch and Jean-Christophe Olivo-Marin},
  title   = {Multiple Hypothesis Tracking for Cluttered Biological Image Sequences},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2013},
  volume  = {35},
  number  = {11},
  pages   = {2736--2750},
  doi     = {10.1109/TPAMI.2013.97}
}

@article{lagache2021emc2,
  author  = {Thibault Lagache and Alison Hanson and Jes{\'u}s E. P{\'e}rez-Ortega and Adrienne Fairhall and Rafael Yuste},
  title   = {Tracking calcium dynamics from individual neurons in behaving animals},
  journal = {PLOS Computational Biology},
  year    = {2021},
  volume  = {17},
  number  = {10},
  pages   = {e1009432},
  doi     = {10.1371/journal.pcbi.1009432}
}

@article{zhou2025cellect,
  author  = {Hongyu Zhou and Seonghoon Kim and Zhifeng Zhao and Jiaqi Fan and Wen Huang and Xinghua Sui and Lizhi Shao and Haoran An and Jing-Ren Zhang and Jiamin Wu and Qionghai Dai},
  title   = {{CELLECT}: contrastive embedding learning for large-scale efficient cell tracking},
  journal = {Nature Methods},
  year    = {2025},
  volume  = {22},
  number  = {11},
  pages   = {2411--2422},
  doi     = {10.1038/s41592-025-02886-x}
}

@inproceedings{paul2025idunno,
  author    = {Richard D. Paul and Johannes Seiffarth and David R{\"u}gamer and Katharina N{\"o}h and Hanno Scharr},
  title     = {How to Make Your Cell Tracker Say ``I Dunno!''},
  booktitle = {2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  pages     = {6914--6923},
  doi       = {10.1109/ICCV51701.2025.00650}
}

@article{betjes2025error,
  author  = {Max A. Betjes and Rutger N. U. Kok and Sander J. Tans and Jeroen S. van Zon},
  title   = {Cell tracking with accurate error prediction},
  journal = {Nature Methods},
  year    = {2025},
  volume  = {22},
  number  = {11},
  pages   = {2400--2410},
  doi     = {10.1038/s41592-025-02845-6}
}

@article{koch2008randommatrix,
  author  = {J. Wolfgang Koch},
  title   = {Bayesian approach to extended object and cluster tracking using random matrices},
  journal = {IEEE Transactions on Aerospace and Electronic Systems},
  year    = {2008},
  volume  = {44},
  number  = {3},
  pages   = {1042--1059},
  doi     = {10.1109/TAES.2008.4655362}
}

@article{beard2016extended,
  author  = {Michael Beard and Stephan Reuter and Karl Granstr{\"o}m and Ba-Tuong Vo and Ba-Ngu Vo and Alexander Scheel},
  title   = {Multiple Extended Target Tracking With Labeled Random Finite Sets},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2016},
  volume  = {64},
  number  = {7},
  pages   = {1638--1653},
  doi     = {10.1109/TSP.2015.2505683}
}
```
