# Run 2 - Primary Experimental Results

## Overview

Run 2 is the primary experimental run used in this GILE
reproduction project.

It contains 13 target-participant experiments across three
wearable-sensor human activity recognition datasets:

- UCI Human Activity Recognition (UCI HAR)
- Opportunity
- UniMiB SHAR

All 13 experiments are recorded as completed in the
Run 2 execution log.

## Experiment coverage

| Dataset | Target participants | Experiments | Epochs |
|---|---|---:|---:|
| UCI HAR | 0, 1, 2, 3, 4 | 5 | 150 |
| Opportunity | S1, S2, S3, S4 | 4 | 100 |
| UniMiB SHAR | 1, 2, 3, 5 | 4 | 100 |
| Total | | 13 | |

## Results

| Dataset | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|
| UCI HAR | 89.51 | 88.75 | 88.99 |
| Opportunity | 81.16 | 52.08 | 79.46 |
| UniMiB SHAR | 52.23 | 41.03 | 48.14 |

These are descriptive averages across the target-participant
experiments in Run 2.

## Training configuration

The recorded epoch counts are:

- UCI HAR: 150 epochs per target.
- Opportunity: 100 epochs per target.
- UniMiB SHAR: 100 epochs per target.

The current experiment scripts define dataset-specific default
configurations. The execution manifest does not record every
hyperparameter passed at runtime, so the defaults should not
automatically be treated as a complete record of the original
command-line invocations.

## Analysis outputs

The analysis directory contains:

- Evaluation JSON files.
- Raw and normalised confusion matrices.
- Classification reports.
- Class-distribution tables.
- Weakest-class analysis.
- Top confusion pairs.
- Recurring activity confusion analysis.
- Activity-level error summaries.
- Dataset-level and experiment-level metrics.
- Output inventory and verification audit.

The final output audit verified all 73 expected outputs.

## Important methodological limitation

An audit of the training implementation identified that the
target-domain loader is evaluated after every training epoch.

The best checkpoint is selected using target-domain
classification accuracy.

Consequently, target-domain labels indirectly influence
checkpoint selection, even though they are not shown to
contribute to gradient-based parameter updates.

Therefore, Run 2 should not be interpreted as a strictly
unsupervised domain generalisation evaluation.

The reported results describe performance under the
implemented checkpoint-selection procedure.

## Run 1 distinction

Run 1 is retained as historical experimental work.
Its results are not combined with Run 2 for the final analysis.

## Reproducibility records

Training logs and execution records are stored in:

`../run2_logs/`

Evaluation outputs are stored in:

`../../evaluation-code/evaluation_results/`

The analysis outputs are stored in this directory.
