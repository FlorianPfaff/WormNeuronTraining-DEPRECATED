# Worm neuron tracking (deprecated)

This repository archives a study of neuron identity tracking in *C. elegans* whole-brain microscopy. Given a set of 3D neuron centers in each scan and known identities in the first scan, can a tracker keep assigning the same identity as the animal deforms? We tested a dual-anchor method that uses both the previous scan and the first scan, fixed-reference Coherent Point Drift (CPD), and PyRecEst multi-hypothesis tracking.

## Why we stopped

We closed the methods-paper project in November 2026. Dual anchoring improved tracking on the development recordings, but the external comparison yielded no advantage, and the matched comparison gave us no reason to keep developing the multi-hypothesis tracker.

- **The development recordings revealed a trade-off.** Across 20 previously examined EY animals, dual anchoring raised identity accuracy from 90.045% to 93.029% against fixed-reference CPD and reduced the mean switch rate by 77.8%. When it assigned the wrong identity, however, that error lasted 10.81 scans on average, compared with 3.74 for CPD. Counting switches alone would have hidden the slower recovery. [Phase-two report](outputs/paper_phase2_20261004/report.md)
- **The multi-hypothesis approach failed its mechanism test.** With the same access to two future scans, the frozen PyRecEst multi-hypothesis variant scored 0.207 percentage points below rolling best-path search at native rate. It improved accuracy on zero of three animals. Our literature review also found earlier work using reference anchors and multiple hypotheses for neuron tracking. [Matched comparison](outputs/paper_phase2_20261004/report.md#matched-information-mechanism) · [Prior-work comparison](outputs/paper_phase2_20261004/closest_papers.csv)
- **The external recordings reached a ceiling.** On six previously unscored Kato-lab animals (SK1 and SK2), the frozen dual-anchor and CPD trackers made identical assignments. Both matched all 1,429,500 supplied ROI identities and made zero switches in 1,428,808 track transitions. Accuracy remained 100% after excluding the initialized frame. Neither method produced a wrong-identity episode, so these animals could not tell us whether anchoring delays recovery. The reference trajectories were algorithm-assisted and curated. In each animal, 33%–66% of adjacent scans had identical coordinates for every ROI. [Per-animal scores](studies/sk_external/scores/per_animal.csv) · [Input diagnostics](studies/sk_external/input_motion_diagnostics.json)

The finding worth keeping is that accuracy, switch rate and wrong-episode duration answer different questions. A future study of recovery would need several independent animals with human-checked identities through ambiguous crossings and subsequent recovery.

## Code and records

- [Tracking and evaluation code](src/deformeot), [tests](tests), [experiment configurations](configs), and [study protocols](docs).
- [External SK comparison](studies/sk_external) with the original extraction, tracking and independent scoring scripts, [frozen settings](studies/sk_external/frozen_protocol.json), and [versioned input identifiers](studies/sk_external/inputs/input_manifest.json). The scoring summary preserves the historical metric convention of recording zero for animals with no episodes. The conditional mean duration is undefined for those animals. [Full score summary](studies/sk_external/scores/summary.json)
- [Historical project notes](HISTORY.md) document the exploratory work and earlier protocols. The closure decision above includes the later external result.

The historical tracking runs used Linux and Python 3.12.3. Run the source tests from the repository root in the [pinned environment](requirements-paper-pilot-lock.txt).

```sh
PYTHONPATH=src python -m pytest -q
```

The original experiment launchers refer to the study server and its archived runtime. Full experiment reruns require the corresponding inputs. The SK comparison used the original calcium coordinate tables from these two releases. [SK1 on DANDI](https://doi.org/10.48324/dandi.000565/0.241009.1504) · [SK2 on DANDI](https://doi.org/10.48324/dandi.000472/0.241009.1502)
