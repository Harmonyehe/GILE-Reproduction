# GILE Reproduction — Final Results Report

## Scope and provenance

Model training was conducted during the reproduction experiments. All model-training experiments were run in Google Colab using GPU acceleration.

This report was generated during the later evaluation and consolidation stage; that report-generation step did not run additional training epochs.

The final result set combines:
- UCI HAR targets 0–4: corrected 100-epoch experiments, evaluated using fresh CPU evaluation JSONs.
- Opportunity targets S1–S4: completed 100-epoch Run 2 experiments, using Run 2 evaluation JSONs.
- UniMiB-SHAR targets 1, 2, 3 and 5: completed 100-epoch Run 2 experiments, using Run 2 evaluation JSONs.

Run 1 was excluded because it was interrupted and its UCI HAR execution used 150 epochs. The Run 2 UCI HAR execution also used 150 epochs and was superseded by the corrected 100-epoch UCI HAR experiments.

## Target-wise results

| Dataset | Target | Test samples | Accuracy | Macro-F1 | Weighted-F1 | Provenance |
|---|---:|---:|---:|---:|---:|---|
| UCI HAR | 0 | 347 | 85.30% | 85.98% | 85.18% | Fresh UCI HAR evaluation |
| UCI HAR | 1 | 302 | 79.47% | 72.66% | 74.01% | Fresh UCI HAR evaluation |
| UCI HAR | 2 | 341 | 94.13% | 93.90% | 94.12% | Fresh UCI HAR evaluation |
| UCI HAR | 3 | 317 | 91.17% | 91.06% | 91.05% | Fresh UCI HAR evaluation |
| UCI HAR | 4 | 302 | 92.05% | 92.02% | 92.07% | Fresh UCI HAR evaluation |
| Opportunity | S1 | 15643 | 82.66% | 61.84% | 82.19% | Run 2 evaluation evidence |
| Opportunity | S2 | 15011 | 81.24% | 53.51% | 79.85% | Run 2 evaluation evidence |
| Opportunity | S3 | 14456 | 78.20% | 43.82% | 76.37% | Run 2 evaluation evidence |
| Opportunity | S4 | 12843 | 79.59% | 50.14% | 79.13% | Run 2 evaluation evidence |
| UniMiB-SHAR | 1 | 384 | 46.35% | 33.83% | 40.26% | Run 2 evaluation evidence |
| UniMiB-SHAR | 2 | 583 | 53.86% | 47.20% | 52.46% | Run 2 evaluation evidence |
| UniMiB-SHAR | 3 | 304 | 67.11% | 48.81% | 65.79% | Run 2 evaluation evidence |
| UniMiB-SHAR | 5 | 298 | 41.61% | 34.28% | 34.06% | Run 2 evaluation evidence |

## Paper comparison

Paper comparison is provided for accuracy only. No paper F1 baseline is asserted.

- UCI HAR target 0: reproduction 85.30% vs paper 85.15% (+0.15 pp).
- UCI HAR target 1: reproduction 79.47% vs paper 81.56% (-2.09 pp).
- UCI HAR target 2: reproduction 94.13% vs paper 86.97% (+7.16 pp).
- UCI HAR target 3: reproduction 91.17% vs paper 94.37% (-3.20 pp).
- UCI HAR target 4: reproduction 92.05% vs paper 92.81% (-0.76 pp).
- Opportunity target S1: reproduction 82.66% vs paper 83.86% (-1.20 pp).
- Opportunity target S2: reproduction 81.24% vs paper 81.65% (-0.41 pp).
- Opportunity target S3: reproduction 78.20% vs paper 78.66% (-0.46 pp).
- Opportunity target S4: reproduction 79.59% vs paper 81.41% (-1.82 pp).
- UniMiB-SHAR target 1: reproduction 46.35% vs paper 55.72% (-9.37 pp).
- UniMiB-SHAR target 2: reproduction 53.86% vs paper 54.06% (-0.20 pp).
- UniMiB-SHAR target 3: reproduction 67.11% vs paper 70.31% (-3.20 pp).
- UniMiB-SHAR target 5: reproduction 41.61% vs paper 42.81% (-1.20 pp).

## Checkpoint recovery

- Remote branch checkpoint candidates: 5.
- Checkpoint recovery manifest entries: 6.
- Recovered checkpoint files retain their available source-path provenance.
- A recovered checkpoint is not automatically treated as the source of an archived metric JSON; matching and independent evaluation must be established.

## Confusion-matrix evidence

Confusion matrices are generated only from raw matrices contained in authoritative metric JSONs. Fresh UCI HAR matrices are marked unavailable if no fresh raw matrix exists; archived UCI HAR matrices are not substituted.

## Limitations

- Missing fields remain unavailable; no metrics are fabricated.
- Training-history plots show recorded rows and do not infer epoch numbering from ambiguous CSV headers.
- The target-domain accuracy is used for checkpoint selection during training, so target labels influence model selection.
- Historical training accuracy can be underestimated due to the original denominator calculation; see `Analysis/Audit/accuracy_calculation_audit.txt`.
- External data files or dependencies required for re-evaluation are reported as missing rather than guessed.