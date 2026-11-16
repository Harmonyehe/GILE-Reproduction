# GILE: Cross-Person Human Activity Recognition

This repository contains the original GILE implementation and the completed reproduction of the method introduced by Qian, Pan and Miao (AAAI 2021).

**Paper:** *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*
**Authors:** Hangwei Qian, Sinno Jialin Pan and Chunyan Miao
**Venue:** AAAI 2021

## 1. Project overview

GILE addresses cross-person human activity recognition (HAR) from wearable-sensor time series. The central reproduction question is whether a model trained using data from source participants can recognise activities for a different target participant. The target participant is treated as a separate domain because sensor patterns can vary between people even when they perform the same activity.

The reproduction covers three datasets and 13 target participants:

| Dataset | Target participants | Targets |
|---|---|---:|
| UCI HAR | 0, 1, 2, 3, 4 | 5 |
| Opportunity | S1, S2, S3, S4 | 4 |
| UniMiB-SHAR | 1, 2, 3, 5 | 4 |

## 2. Experimental history and final selection

All model-training experiments were run in Google Colab using GPU acceleration.

**Run 1 — excluded.** The initial attempt was interrupted, and its UCI HAR execution used 150 epochs rather than the intended 100-epoch configuration. Run 1 is retained as a historical record, but its outputs are not included in the final metrics.

**Run 2 — main execution.** Opportunity and UniMiB-SHAR were run for 100 epochs for every target. The Run 2 UCI HAR experiments used 150 epochs, so those UCI HAR results were not retained as the final UCI HAR results.

**Corrected UCI HAR execution.** After comparing the experiment configuration with the paper, UCI HAR was rerun for targets 0–4 using 100 epochs.

**Final 13-target set.** The final results combine the five corrected UCI HAR target experiments with the four Opportunity and four UniMiB-SHAR Run 2 target experiments. This is the basis for the final cross-dataset comparison.

The report-generation and evaluation-consolidation notebooks did not train models. That statement describes those later analysis steps only; it does not mean the reproduction project lacked training.

## 3. Final dataset-level results

The following are unweighted means across target participants within each dataset. The paper comparison is for accuracy only.

| Dataset | Paper accuracy (%) | Reproduced accuracy (%) | Difference (percentage points) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|---:|---:|
| UCI HAR | 88.172 | 88.426 | +0.254 | 87.125 | 87.287 |
| Opportunity | 81.395 | 80.421 | -0.974 | 52.326 | 79.385 |
| UniMiB-SHAR | 55.725 | 52.232 | -3.493 | 41.032 | 48.143 |

The F1 values describe the reproduction. Paper F1 baselines are not asserted.

## 4. How to read the analysis

- **Target-wise tables** show performance for each held-out participant separately. They reveal whether the average is consistent or driven by a subset of targets.
- **Dataset-level tables** are unweighted averages of the target-level values. Each target contributes equally, rather than each test example contributing equally across targets.
- **Accuracy** is the fraction of correctly classified test examples.
- **Macro-F1** computes F1 per class and averages classes equally. It is sensitive to poor performance on less frequent classes.
- **Weighted-F1** averages class F1 using each class's support. It is more influenced by common classes.
- **Confusion matrices** show the relationship between true and predicted labels. Rows and columns should be interpreted using the dataset's verified label mapping.
- **Training-history plots** show the values recorded during training. They are not a substitute for final sample-level evaluation.
- **Paper comparisons** use accuracy values transcribed for the corresponding targets. Differences are reported in percentage points, not relative percent.

A large gap between weighted-F1 and macro-F1 can indicate uneven class-level performance. Class-support imbalance is descriptive; it does not, by itself, establish why a model makes particular errors.

## 5. Evaluation protocol and limitations

The implementation evaluates target-domain performance during training and uses target-domain accuracy for checkpoint selection. Target labels therefore influence checkpoint selection, even though they are not used for gradient-based parameter updates. The results should not be described as a strictly target-unseen model-selection evaluation.

The original training-accuracy calculation uses `len(predictions_y) * batch_size` as its denominator. Because the final batch can be smaller than the configured batch size, historical training accuracy can be underestimated. The final evaluation metrics use the actual number of evaluated samples. See `Reproduction/Final Reproduction/Analysis/Audit/accuracy_calculation_audit.txt`.

The final UCI HAR metrics come from fresh CPU evaluation JSONs. This CPU evaluation is an evaluation step; model training was performed on Colab GPU. Opportunity and UniMiB-SHAR final metrics use the Run 2 evaluation outputs.

## 6. Repository map

- `Original Code/` — source implementation supplied with the paper.
- `Reproduction/Final Reproduction/` — final experiment results, evaluation artefacts, analysis, setup and methodology documentation.
- `Reproduction/Experiment Archive/Run 1/` — interrupted initial attempt, retained for traceability and excluded from final metrics.
- `Reproduction/Experiment Archive/Run 2/` — main execution records and analysis.
- `Reproduction/Experiment Archive/Source Archive/` — source snapshots and supporting records.

Start with:
1. `Reproduction/Final Reproduction/README.md` for the final experiment scope and results.
2. `Reproduction/Final Reproduction/SETUP.md` for environment, data layout and commands.
3. `Reproduction/Final Reproduction/ANALYSIS_GUIDE.md` for interpretation of the evaluation artefacts.

Raw datasets are not included. Follow the setup guide and original dataset preparation instructions before attempting a new run.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/QianPM21,
  author    = {Hangwei Qian and Sinno Jialin Pan and Chunyan Miao},
  title     = {Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {11921--11929},
  year      = {2021}
}
```