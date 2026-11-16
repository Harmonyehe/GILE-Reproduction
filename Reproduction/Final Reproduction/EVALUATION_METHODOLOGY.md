# Evaluation Methodology

This finalisation is evaluation and repository maintenance only.
It does not train models or execute training entry points.

## Metric sources

- UCI HAR: fresh CPU evaluation JSON files in
  `Results/UCI HAR/Fresh CPU Evaluation/`.
- Opportunity: Run 2 archived evaluation JSON files.
- UniMiB-SHAR: Run 2 archived evaluation JSON files.

Accuracy, macro-F1, weighted-F1, sample counts, and per-class
precision/recall/F1 are copied from source JSONs when present.
Missing values are not inferred.

Confusion matrices are produced only from raw matrices in the
authoritative source JSONs.

## Backup-branch checkpoints

Fetched remote branches are searched using Git's tracked tree.
Recovered checkpoint files retain available branch and original
path provenance.

Recovery alone does not establish metric provenance.
Independent evaluation requires a target-matched checkpoint,
required dataset inputs, and an evaluation-only script.

## Paper comparison

Accuracy is compared with the target-wise GILE values
transcribed for this project. Paper F1 values are not assumed
or invented.
