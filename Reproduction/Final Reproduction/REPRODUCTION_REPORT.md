# GILE Reproduction — Historical Archived-Metric Report

> **Status: historical audit output, not the final results report.** This report was generated from archived evaluation JSON files during an earlier repository audit. Its UCI HAR values are from the earlier 150-epoch Run 2 execution and are superseded by the corrected 100-epoch UCI HAR experiments. Use `FINAL_REPRODUCTION_REPORT.md` and the final analysis tables for the final results.

## Experiment coverage

The audit recovered 13 evaluation records from archived JSON files. This report documents those historical files and must not be confused with a fresh evaluation of the corrected UCI HAR checkpoints.

## Dataset-level performance in the archived files

| Dataset | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| Opportunity | 0.804213 | 0.523264 | 0.793854 |
| UCI HAR | 0.893419 | 0.878714 | 0.882422 |
| UniMiB-SHAR | 0.522324 | 0.410315 | 0.481426 |

## Provenance and limitations

- These metrics originate from archived JSON files.
- The archived UCI HAR values correspond to the earlier 150-epoch Run 2 execution and are superseded for final reporting.
- The final UCI HAR metrics are in `FINAL_REPRODUCTION_REPORT.md` and use fresh CPU evaluation of the corrected 100-epoch experiments.
- Opportunity and UniMiB-SHAR final metrics use the completed 100-epoch Run 2 experiments.
- Existing confusion matrices and analysis tables are preserved.
- The evaluation/finalisation notebooks did not train models; the training was performed earlier in the Colab experiments.