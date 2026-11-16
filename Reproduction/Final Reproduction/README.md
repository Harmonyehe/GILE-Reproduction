# GILE Reproduction — Final Experimental Results

## Scope

This directory contains the completed reproduction of GILE across three wearable-sensor human activity recognition datasets and 13 target participants:

- **UCI HAR:** targets 0, 1, 2, 3 and 4.
- **Opportunity:** targets S1, S2, S3 and S4.
- **UniMiB-SHAR:** targets 1, 2, 3 and 5.

## Final experimental configuration

The final reproduction uses 100 training epochs for every target experiment.

Run 2 provided the Opportunity and UniMiB-SHAR experiments. During the comparison with the paper, the UCI HAR training duration was checked and adjusted to 100 epochs. The five UCI HAR target experiments were run using this configuration. Together, these experiments form the final 13-target reproduction.

The initial experimental attempt was interrupted and is retained in the experiment records. It is not part of the final result tables.

## Final dataset-level results

The figures below are unweighted means across the target participants within each dataset.

| Dataset | Targets | Paper accuracy (%) | Reproduced accuracy (%) | Difference (percentage points) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| UCI HAR | 5 | 88.172 | 88.426 | +0.254 | 87.125 | 87.287 |
| Opportunity | 4 | 81.395 | 80.421 | -0.974 | 52.326 | 79.385 |
| UniMiB-SHAR | 4 | 55.725 | 52.232 | -3.493 | 41.032 | 48.143 |

The paper comparison is based on the accuracy values reported by Qian et al. (AAAI 2021). F1 scores are reported for the reproduction and are not presented as paper comparisons.

## Evaluation and methodological notes

- UCI HAR results are based on a fresh CPU evaluation of the final 100-epoch configuration.
- Opportunity and UniMiB-SHAR results are from the Run 2 target experiments.
- The final tables bring all 13 target results together using a consistent set of reported metrics.
- Target-domain accuracy is evaluated during training and is used for checkpoint selection. Target labels therefore influence model selection. This limitation should be considered when interpreting cross-person generalisation performance.
- Final sample-level metrics use the number of evaluated examples. The historical training-log accuracy calculation has a batch-size denominator issue when the final batch is incomplete. The details are documented in `Analysis/Audit/accuracy_calculation_audit.txt`.

## Main files

- `Results/` — evaluation metrics, prediction outputs and result summaries.
- `Analysis/Final Analysis Tables/` — consolidated target-level, dataset-level and per-class results.
- `Analysis/Fresh UCI HAR Confusion Matrices/` — final UCI HAR confusion matrices, CSV data and summary metrics.
- `Analysis/Training History Plots/` — activity and domain accuracy plots for all 13 target experiments, plus the training-history summary.
- `Analysis/Analysis Tables/training_history_vs_evaluation.csv` — comparison of training-history and final evaluation metrics.
- `Analysis/Audit/accuracy_calculation_audit.txt` — audit of the historical accuracy calculation and final reporting approach.
- `Code/` — reproduction and evaluation source code.

## Reproducibility

The final result tables identify the dataset, target participant, sample count, accuracy, Macro-F1, Weighted-F1 and evidence source. Prediction-level outputs are retained for the final UCI HAR evaluation. Training histories, plots and experiment records are retained alongside the analysis.

The raw datasets and model checkpoints are not distributed as part of this results directory. Refer to the original implementation documentation for dataset preparation and environment requirements.
