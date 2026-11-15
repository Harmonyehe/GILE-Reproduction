# GILE: Cross-Person Human Activity Recognition

This repository contains the original GILE implementation and an independent reproduction of the approach for cross-person human activity recognition using wearable sensor data.

The work is based on:

**Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition**

Qian, H., Pan, S. J., and Miao, C. (2021). AAAI 2021.

## Repository Structure

- `Original Code/` — original GILE implementation.
- `Reproduction/Run 1/` — materials retained from the initial reproduction.
- `Reproduction/Final Reproduction/` — organised reproduction code, results, analysis and logs.
- `Reproduction/Experiment Archive/` — previous experiment outputs retained for reference.

## Datasets

The reproduction considers three benchmark datasets:

- UCI Human Activity Recognition (UCI HAR)
- Opportunity
- UniMiB SHAR

The experiments use a cross-person evaluation setting, with participants treated as target domains.

| Dataset | Target participants |
|---|---|
| UCI HAR | 0, 1, 2, 3, 4 |
| Opportunity | S1, S2, S3, S4 |
| UniMiB SHAR | 1, 2, 3, 5 |

## Implementation

The reproduction code retains the GILE architecture and training approach.

Documented compatibility modifications include:
- Replacing `np.product()` with `np.prod()` in `sliding_window.py`.
- Adding explicit device arguments to `torch.eye()` calls where required.
- Extending evaluation to report Accuracy, Macro-F1 and Weighted-F1.

## Experiment Status and Results

Previous Run 2 outputs are preserved for reference. They include participant-level evaluation results, training logs, classification reports and confusion-matrix analyses.

**Important:** Run 2 results are historical and preliminary. They must not automatically be interpreted as results from the final reproduction.

A methodological limitation was identified in the previous training procedure: target-domain data was evaluated during training, and target-domain accuracy was used for checkpoint selection. This means target-domain information influenced model selection.

A stricter cross-person evaluation should select checkpoints using source-domain validation data only, then evaluate the held-out target participant after training.

The final reproduction summary and report will be updated after the relevant experiments have been completed and verified. Historical results must remain clearly distinguished from new results.

## Reproducibility

- Training, preprocessing and evaluation scripts are organised under `Reproduction/Final Reproduction/Code/`.
- Results, analysis and logs are stored in their respective folders.
- Raw datasets are not included in this repository.
- Model checkpoints are not currently included.
- Consult the relevant experiment documentation before interpreting individual result files.

## Acknowledgements

This reproduction is based on the original GILE implementation by Hangwei Qian, Sinno Jialin Pan and Chunyan Miao.
