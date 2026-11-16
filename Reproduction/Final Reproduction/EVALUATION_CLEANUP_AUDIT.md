# Evaluation and Cleanup Audit

## Scope of this audit

This audit notebook performed repository inspection, metric consolidation, provenance checks and analysis only. It did not invoke model-training entry points or run training epochs. This statement applies to the audit notebook, not to the reproduction project as a whole.

The model-training experiments had already been conducted in Google Colab using GPU acceleration. Run 1 was excluded because it was interrupted and its UCI HAR run used 150 epochs. The final result set uses corrected 100-epoch UCI HAR experiments and completed 100-epoch Run 2 Opportunity and UniMiB-SHAR experiments.

## Metric sources

- UCI HAR: fresh CPU evaluation JSON files in `Results/UCI HAR/Fresh CPU Evaluation/`, associated with the corrected 100-epoch UCI HAR experiments.
- Opportunity: Run 2 evaluation JSON files.
- UniMiB-SHAR: Run 2 evaluation JSON files.

Accuracy, Macro-F1, Weighted-F1, sample counts and per-class precision/recall/F1 are taken from the source evaluation files when present. Missing values are not inferred or fabricated.

## Provenance and checkpoints

The audit inventories metric JSONs, matching checkpoints and logs. A recovered checkpoint is not automatically assumed to be the source of an archived metric JSON. Independent evaluation requires a target-matched checkpoint, the required dataset inputs and an evaluation-only script.

## Interpretation limits

The final report distinguishes fresh UCI HAR evaluation from archived Run 2 Opportunity and UniMiB-SHAR evaluation outputs. The source of a metric must remain visible. A metric JSON alone does not prove that the finalisation notebook freshly evaluated a checkpoint.