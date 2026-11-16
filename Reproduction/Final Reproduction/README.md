
# GILE Reproduction — Final Results

## Scope

This folder contains the consolidated reproduction results for:

- UCI HAR: target domains 0, 1, 2, 3, 4.
- Opportunity: target domains S1, S2, S3, S4.
- UniMiB-SHAR: target domains 1, 2, 3, 5.

Total: 13 target-domain experiments.

## Authoritative result sources

- UCI HAR uses the newer fresh CPU evaluation results.
- Opportunity uses the existing Run 2 evaluation outputs.
- UniMiB-SHAR uses the existing Run 2 evaluation outputs.

No model training or training epochs were run during final assembly.

## Main files

- `Results/FINAL_RESULTS_SUMMARY.csv`: canonical 13-target summary.
- `Results/Combined/all_13_target_results.csv`: combined target-level results.
- `Results/Combined/dataset_averages.csv`: dataset averages.
- `Analysis/Analysis Tables/paper_comparison.csv`: comparison against AAAI 2021.
- `Analysis/Analysis Tables/per_class_classification_metrics.csv`: per-class precision, recall and F1.
- `Analysis/Confusion Matrices/`: confusion matrix artefacts copied from the source runs.
- `Analysis/Dataset comparisons/`: paper comparison plots.
- `Analysis/Performance analysis/`: overall performance plots.

## Provenance

Each target row in the canonical results table identifies its source.
UCI HAR is labelled as fresh evaluation.
Opportunity and UniMiB-SHAR are labelled as Run 2 archived evaluation.

Archived evidence is retained in the Experiment Archive.
