# GILE Reproduction — Analysis Guide

This guide explains what the saved evaluation and analysis artefacts mean, how the reported metrics are calculated, and what conclusions can and cannot be drawn from them.

## 1. Experimental unit: target participant

Each target experiment evaluates generalisation to one participant/domain. The final set contains five UCI HAR targets (0–4), four Opportunity targets (S1–S4) and four UniMiB-SHAR targets (1, 2, 3 and 5), for 13 target experiments total.

Target-wise reporting matters because a single dataset average can conceal large differences between participants. Compare target-level rows before drawing conclusions from dataset-level means.

## 2. Metrics

### Accuracy
Accuracy is the number of correct predictions divided by the number of evaluated examples. It gives an overall measure but can hide poor performance on minority classes.

### Precision, recall and F1
For a class, precision measures how many predictions of that class are correct. Recall measures how many true examples of that class are recovered. F1 is the harmonic mean of precision and recall.

### Macro-F1
Macro-F1 calculates F1 separately for each class and averages those values equally. A rare class has the same influence on Macro-F1 as a common class. It is useful for detecting uneven class-level performance.

### Weighted-F1
Weighted-F1 averages class F1 values using the number of true examples in each class (support). Common classes therefore contribute more. Weighted-F1 can remain high when performance on less common classes is weak.

A substantial gap between Weighted-F1 and Macro-F1 is a signal to inspect per-class metrics and the confusion matrix. It is not, by itself, proof that class imbalance caused the errors.

## 3. Dataset-level averages

The final dataset-level summary uses an unweighted mean across target participants. Each participant contributes equally to the dataset mean. This is different from pooling all test examples from all participants and computing one global metric.

For example, the dataset-level accuracy is the arithmetic mean of the target-wise accuracies. The sample counts can differ between targets, but they do not act as weights in this reported average.

## 4. Paper comparison

The paper comparison is limited to target-wise accuracy values. Differences are reported in percentage points:

`reproduction accuracy - paper accuracy`

A positive value means the reproduction is higher; a negative value means it is lower. These are percentage-point differences, not relative percentage changes. No paper Macro-F1 or Weighted-F1 baseline is claimed.

## 5. Confusion matrices

A confusion matrix compares true classes (rows) with predicted classes (columns), subject to the plotting convention used by the generating script. The diagonal contains correct predictions. Off-diagonal cells show which classes are confused.

Use the matching dataset's verified label mapping before translating numeric class IDs into activity names. Do not guess class names from numeric IDs.

A normalised matrix expresses proportions rather than raw counts. Always check whether normalisation is by true class, predicted class or the full matrix before interpreting percentages.

## 6. Training-history plots

Training-history plots show values recorded during model fitting, such as activity classification accuracy and domain classification accuracy. They are useful for observing trends, instability and changes over epochs.

They should not be treated as final test metrics. The original training-accuracy implementation uses a denominator based on the configured batch size even when the last batch is smaller. This can underestimate historical training accuracy. See `Analysis/Audit/accuracy_calculation_audit.txt`.

## 7. Checkpoint selection and evaluation protocol

The implementation evaluates target-domain performance during training and uses target-domain accuracy for checkpoint selection. Target labels therefore influence checkpoint selection, even though they are not used for gradient-based parameter updates.

This is an important methodological limitation. The experiment should not be described as a strictly target-unseen model-selection procedure.

## 8. Provenance of final metrics

- UCI HAR: corrected 100-epoch target experiments, followed by fresh CPU evaluation of the selected checkpoints.
- Opportunity: completed 100-epoch Run 2 target experiments; final report uses Run 2 evaluation outputs.
- UniMiB-SHAR: completed 100-epoch Run 2 target experiments; final report uses Run 2 evaluation outputs.

The evaluation/consolidation notebooks assemble and audit these outputs. They do not perform additional training.

## 9. Key files

- `Analysis/Final Analysis Tables/dataset_level_average_comparison.csv` — dataset-level comparison.
- `Analysis/Final Analysis Tables/all_datasets_target_comparison.csv` — target-level comparison.
- `Analysis/Final Analysis Tables/all_datasets_per_class_metrics.csv` — per-class metrics.
- `Analysis/Training History Plots/` — training-history visualisations.
- `Analysis/Fresh UCI HAR Confusion Matrices/` — fresh UCI HAR confusion-matrix evidence.
- `Analysis/Audit/accuracy_calculation_audit.txt` — training-accuracy calculation caveat.
- `FINAL_REPRODUCTION_REPORT.md` — target-wise metrics and paper comparison.
