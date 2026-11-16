
# Evaluation Methodology and Provenance

This final assembly uses existing experiment outputs only.

No training script was invoked and no epochs were run.

UCI HAR results are taken from the newer fresh CPU evaluation JSON files.
Opportunity and UniMiB-SHAR results are taken from the Run 2 metric JSON files and supporting artefacts.

Accuracy, macro-F1 and weighted-F1 are taken directly from the corresponding metric JSON.
Per-class precision, recall, F1 and support are extracted from the saved classification reports.

Paper comparison uses the reported GILE accuracies from Qian et al., AAAI 2021.
Differences are reported in percentage points.

The provenance column in the canonical result table distinguishes fresh UCI HAR metrics from Run 2 archived metrics.
