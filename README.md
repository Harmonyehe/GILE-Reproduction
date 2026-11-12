# GILE-Reproduction
# GILE: Reproduction of Cross-Person Human Activity Recognition

This repository contains an independent reproduction and evaluation of **Generalizable Independent Latent Excitation (GILE)**, a deep learning approach for cross-person human activity recognition using wearable sensor data.

The implementation is based on the original GILE code by Hangwei Qian, Sinno Jialin Pan and Chunyan Miao, proposed in their AAAI 2021 paper:

**Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition.**

## 1. Project Overview

Human activity recognition models often perform well when training and testing data come from the same individuals. However, differences in movement patterns between people can reduce model performance when it encounters an unseen person.

GILE addresses this problem by learning domain-agnostic and domain-specific representations, aiming to improve generalisation across different participants without requiring target-domain information.

This project reproduces the GILE implementation across three benchmark datasets:

- UCI Human Activity Recognition (UCI HAR)
- Opportunity
- UniMiB SHAR

## 2. Repository Structure

```text
GILE-Reproduction/
│
├── original-code/
│   └── cross-person-HAR/
│       └── Original GILE implementation
│
├── evaluation-code/
│   ├── GILE implementation and evaluation scripts
│   ├── evaluation_results/
│   │   ├── all_13_results.csv
│   │   ├── dataset_averages.csv
│   │   └── Individual participant evaluation JSON files
│   └── README.md
│
├── results-and-report/
│   └── run2_logs/
│       ├── Training logs
│       ├── Evaluation logs
│       └── Experiment execution records
│
├── .gitignore
└── README.md
```

Dataset files and trained model checkpoints are not included in this repository.

## 3. Datasets and Experimental Protocol

The experiments use a cross-person evaluation setting in which one participant is held out as the target domain while the remaining participants form the source domains.

| Dataset | Target participants | Number of experiments |
|---|---|---:|
| UCI HAR | 0, 1, 2, 3, 4 | 5 |
| Opportunity | S1, S2, S3, S4 | 4 |
| UniMiB SHAR | 1, 2, 3, 5 | 4 |
| **Total** | | **13** |

The dataset configurations follow the original paper's experimental setup, including:

- UCI HAR: 128 readings per window, with 50% overlap.
- Opportunity: 77 selected features and one-second sliding windows.
- UniMiB SHAR: Participants 1, 2, 3 and 5.

## 4. Implementation and Modifications

The original GILE architecture and training approach were retained.

The following modifications were made to support execution and evaluation in the reproduction environment:

**Compatibility modifications**
- Replaced `np.product()` with `np.prod()` in `sliding_window.py` to resolve compatibility issues with NumPy 2.x.
- Added explicit device arguments to `torch.eye()` calls in `model_GILE_ucihar.py` to ensure tensors are created on the appropriate device.

**Evaluation modifications**
- Added Macro-F1 score.
- Added Weighted-F1 score.
- Updated CSV logging to record Accuracy, Macro-F1 and Weighted-F1.

These changes were intended to improve compatibility and provide a more detailed evaluation without redesigning the GILE architecture.

## 5. Run 2: Initial Evaluation Results

Run 2 completed all 13 participant-level experiments across the three datasets.

The following table presents the mean metrics across the held-out participants for each dataset.

| Dataset | Mean Accuracy (%) | Mean Macro-F1 (%) | Mean Weighted-F1 (%) |
|---|---:|---:|---:|
| UCI HAR | 89.51 | 88.75 | 88.99 |
| Opportunity | 81.16 | 52.08 | 79.46 |
| UniMiB SHAR | 52.23 | 41.03 | 48.14 |

Individual participant-level metrics, classification reports and confusion matrices are available in `evaluation-code/evaluation_results/`.

Training and evaluation logs are available in `results-and-report/run2_logs/`.

### Important methodological limitation

A subsequent code audit identified an issue in the current training and checkpoint-selection procedure.

The held-out target participant's data is evaluated during every training epoch. The best model checkpoint is selected using target-domain accuracy.

Consequently, although the target participant is excluded from source-domain model training, target-domain labels influence checkpoint selection.

**Therefore, these Run 2 results must be treated as preliminary reproduction results, not as an unbiased measurement of generalisation to a completely unseen participant.**

A corrected evaluation should select the training epoch using source-domain validation data only, then evaluate the held-out target participant after training without using target labels for model or checkpoint selection.

This correction is a necessary next step before making definitive claims about GILE's cross-person generalisation performance or comparing the results directly with the paper.

## 6. Reproducibility and Data

The original implementation and its dataset preparation instructions are retained in `evaluation-code/README.md`.

The datasets are not redistributed in this repository. Users should obtain the required data separately and follow the original preparation instructions.

Trained checkpoints and raw datasets are excluded to keep the repository lightweight.

## 7. Original Paper

Qian, H., Pan, S. J., and Miao, C. (2021).

*Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition.*

Proceedings of the AAAI Conference on Artificial Intelligence, 35(13), 11921–11929.

## 8. Acknowledgement

This reproduction is based on the original GILE implementation by Hangwei Qian, Sinno Jialin Pan and Chunyan Miao.

The original implementation is retained separately under `original-code/`. Compatibility and evaluation changes are documented above.
