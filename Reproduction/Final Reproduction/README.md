# GILE Reproduction — Final Experimental Results

## 1. Scope

This directory contains the final 13-target reproduction of GILE across UCI HAR, Opportunity and UniMiB-SHAR.

| Dataset | Final targets | Final training source |
|---|---|---|
| UCI HAR | 0, 1, 2, 3, 4 | Corrected 100-epoch execution |
| Opportunity | S1, S2, S3, S4 | Run 2, 100 epochs per target |
| UniMiB-SHAR | 1, 2, 3, 5 | Run 2, 100 epochs per target |

All model-training experiments were conducted in Google Colab using GPU acceleration. The later UCI HAR CPU evaluation is an evaluation step and should not be confused with the training hardware.

## 2. Experimental history

Run 1 was excluded because the initial execution was interrupted and its UCI HAR experiment used 150 epochs rather than the intended 100-epoch configuration. Run 1 outputs remain archived for traceability and are not included in the final result tables.

Run 2 was the main execution. Opportunity and UniMiB-SHAR completed 100 epochs for each target. The Run 2 UCI HAR experiments used 150 epochs and were superseded.

After comparison with the paper, UCI HAR was rerun for all five target participants using the corrected 100-epoch configuration. The final 13-target result set therefore combines corrected UCI HAR with the completed Run 2 Opportunity and UniMiB-SHAR experiments.

The later evaluation, audit and report-generation notebooks consolidated existing outputs; they did not initiate new model training. This does not mean that the reproduction experiments themselves were untrained.

## 3. Final dataset-level results

Values are unweighted means across target participants within each dataset.

| Dataset | Targets | Paper accuracy (%) | Reproduced accuracy (%) | Difference (percentage points) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| UCI HAR | 5 | 88.172 | 88.426 | +0.254 | 87.125 | 87.287 |
| Opportunity | 4 | 81.395 | 80.421 | -0.974 | 52.326 | 79.385 |
| UniMiB-SHAR | 4 | 55.725 | 52.232 | -3.493 | 41.032 | 48.143 |

Paper comparison is limited to accuracy. Reproduction F1 values are not presented as paper baselines.

## 4. Evaluation sources

- **UCI HAR:** fresh CPU evaluation JSONs in `Results/UCI HAR/Fresh CPU Evaluation/`, using the corrected 100-epoch experiment checkpoints.
- **Opportunity:** Run 2 evaluation outputs for S1–S4.
- **UniMiB-SHAR:** Run 2 evaluation outputs for targets 1, 2, 3 and 5.

The provenance label identifies where each metric was obtained. The presence of an archived evaluation JSON does not by itself mean that the finalisation notebook freshly evaluated a checkpoint.

## 5. Analysis artefacts

- `Analysis/Final Analysis Tables/` — consolidated target-wise, per-class and dataset-level metrics.
- `Analysis/Training History Plots/` — recorded activity/domain training-history plots and summary.
- `Analysis/Fresh UCI HAR Confusion Matrices/` — confusion matrices generated from the fresh UCI HAR evaluation evidence.
- `Analysis/Analysis Tables/training_history_vs_evaluation.csv` — comparison between historical training records and evaluation values.
- `Analysis/Audit/accuracy_calculation_audit.txt` — explanation of the historical training-accuracy denominator issue.

See `ANALYSIS_GUIDE.md` for detailed interpretation of these files.

## 6. Methodological limitations

Target-domain accuracy is evaluated during training and used for checkpoint selection. Target labels therefore influence checkpoint selection, although they are not used for gradient-based parameter updates. The results are not a strictly target-unseen model-selection evaluation.

Historical training accuracy can be underestimated because the original implementation multiplies the number of prediction batches by the configured batch size, even when the last batch is smaller. Final sample-level evaluation metrics use the actual sample count.

## 7. Reproducibility

See `SETUP.md` for the Colab environment, expected data directories, training commands and evaluation commands. The repository does not include the raw datasets. Existing results and logs are preserved as experiment evidence; this documentation update does not retrain models or modify those outputs.