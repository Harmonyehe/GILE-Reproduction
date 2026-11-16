# Evaluation Methodology

## Purpose

This document describes how the final evaluation artefacts were assembled and interpreted. The evaluation/finalisation notebooks perform evaluation, repository maintenance and metric consolidation; they do not retrain models. The earlier model-training experiments were run in Google Colab with GPU acceleration.

## Final experiment sources

- **UCI HAR:** corrected 100-epoch training experiments for targets 0–4, followed by fresh CPU evaluation of the selected checkpoints.
- **Opportunity:** completed 100-epoch Run 2 target experiments S1–S4; the final report uses their Run 2 evaluation outputs.
- **UniMiB-SHAR:** completed 100-epoch Run 2 target experiments 1, 2, 3 and 5; the final report uses their Run 2 evaluation outputs.

Run 1 is excluded because it was interrupted and its UCI HAR execution used 150 epochs. The Run 2 UCI HAR execution also used 150 epochs and was superseded by the corrected 100-epoch UCI HAR experiments.

## Metrics

Accuracy is the proportion of correctly classified test examples. Macro-F1 is the unweighted mean of class-level F1 scores. Weighted-F1 weights each class-level F1 by its number of true examples (support). The target-wise results report each participant separately; dataset-level results are unweighted means across target participants.

The paper comparison is restricted to accuracy. No paper F1 baseline is inferred.

## Confusion matrices

Confusion matrices are included only when a raw matrix is available in the authoritative evaluation evidence. A missing fresh matrix is left unavailable; an archived matrix is not silently relabelled as fresh.

## Checkpoints and provenance

Recovered checkpoints retain available branch and source-path provenance. Recovery alone does not establish that a checkpoint generated an archived metric JSON. A fresh evaluation requires a target-matched checkpoint, the corresponding dataset inputs and an evaluation script.

## Methodological caveat

The implementation evaluates target-domain performance during training and uses target-domain accuracy for checkpoint selection. Target labels therefore influence checkpoint selection even though they are not used for gradient-based parameter updates. The results should not be described as a strictly target-unseen model-selection evaluation.

## Training-history accuracy audit

The original training implementation calculates accuracy using `accurate_preds_y / (len(predictions_y) * batch_size)`. Because the final batch may be smaller than the configured batch size, this denominator can overcount samples and underestimate historical training accuracy. Final sample-level evaluation metrics use the actual evaluated sample count. See `Analysis/Audit/accuracy_calculation_audit.txt`.