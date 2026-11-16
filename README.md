# GILE: Cross-Person Human Activity Recognition

This repository contains the original GILE implementation and the completed reproduction of the method described by Qian, Pan and Miao (AAAI 2021).

**Paper:** *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*
**Authors:** Hangwei Qian, Sinno Jialin Pan and Chunyan Miao
**Venue:** AAAI 2021

## Reproduction overview

The reproduction evaluates cross-person human activity recognition on three wearable-sensor datasets:

- UCI Human Activity Recognition (UCI HAR)
- Opportunity
- UniMiB-SHAR

The evaluation covers 13 target participants in total.

| Dataset | Target participants | Number of targets | Final training configuration |
|---|---|---:|---:|
| UCI HAR | 0, 1, 2, 3, 4 | 5 | 100 epochs |
| Opportunity | S1, S2, S3, S4 | 4 | 100 epochs |
| UniMiB-SHAR | 1, 2, 3, 5 | 4 | 100 epochs |

### Final experimental sequence

The first experimental attempt was interrupted before the complete reproduction was obtained. The subsequent Run 2 execution was carried out as the main experiment. During the comparison with the paper, the UCI HAR training duration was checked and aligned to the 100-epoch configuration used for the final reproduction. The UCI HAR target experiments were therefore run with the corrected 100-epoch setting.

The final reproduction consists of the Run 2 Opportunity and UniMiB-SHAR experiments together with the UCI HAR results produced using the corrected 100-epoch configuration. These results form one final 13-target evaluation across the three datasets.

The initial interrupted attempt is retained in the experiment records for traceability. It is not included in the final 13-target result tables.

## Final results

The following values are unweighted means across the target participants within each dataset. Accuracy is compared with the corresponding paper result. Macro-F1 and weighted-F1 are reported for the reproduction.

| Dataset | Paper accuracy (%) | Reproduced accuracy (%) | Difference (percentage points) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|---:|---:|
| UCI HAR | 88.172 | 88.426 | +0.254 | 87.125 | 87.287 |
| Opportunity | 81.395 | 80.421 | -0.974 | 52.326 | 79.385 |
| UniMiB-SHAR | 55.725 | 52.232 | -3.493 | 41.032 | 48.143 |

The target-level results and per-class metrics are available in the final analysis tables. The comparison uses the paper's reported accuracy values; paper F1 values are not supplied in the comparison.

## Evaluation protocol and interpretation

The final UCI HAR metrics were calculated from a fresh CPU evaluation of the selected checkpoints. The Opportunity and UniMiB-SHAR metrics are the results of their Run 2 target experiments. All 13 target results are brought together in the final analysis tables.

The implementation evaluates target-domain performance during training and uses target-domain accuracy for checkpoint selection. Target labels therefore influence checkpoint selection, even though they are not used for gradient-based parameter updates. The reported scores must be interpreted in light of this protocol; they are not an evaluation in which the target domain is entirely unseen during model selection.

The historical training-accuracy calculation also used a denominator based on the configured batch size. Since the final batch can contain fewer samples, the historical training log values may not exactly match sample-level accuracy. The final evaluation metrics are calculated from the actual number of evaluated samples. See `Reproduction/Final Reproduction/Analysis/Audit/accuracy_calculation_audit.txt` for the calculation audit.

## Repository structure

- `Original Code/` — source implementation supplied with the GILE paper.
- `Reproduction/Final Reproduction/` — final reproduction code, results, analysis, logs and documentation.
- `Reproduction/Experiment Archive/Run 1/` — records from the interrupted initial experiment.
- `Reproduction/Experiment Archive/Run 2/` — execution records and analysis associated with the main experimental run.
- `Reproduction/Experiment Archive/Source Archive/` — source snapshots and their supporting documentation.

## Main final-reproduction files

Within `Reproduction/Final Reproduction/`:

- `Results/` — target-level evaluation outputs and final result summaries.
- `Analysis/Final Analysis Tables/` — consolidated target-level, dataset-level and per-class metrics.
- `Analysis/Fresh UCI HAR Confusion Matrices/` — confusion matrices and supporting metrics from the final UCI HAR evaluation.
- `Analysis/Training History Plots/` — activity and domain training-history plots for all 13 targets.
- `Analysis/Audit/` — evaluation and metric-calculation audit notes.
- `Code/` — reproduction and evaluation scripts.

Raw datasets are not included. Follow the original implementation's dataset preparation instructions and place the processed data in the expected `data` directories before running the scripts.

## Citation

If you use this implementation or reproduce the method, cite the original paper:

```bibtex
@inproceedings{DBLP:conf/aaai/QianPM21,
  author    = {Hangwei Qian and Sinno Jialin Pan and Chunyan Miao},
  title     = {Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {11921--11929},
  year      = {2021}
}
```
