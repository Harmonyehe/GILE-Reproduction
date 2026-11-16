# Run 2 — Automated Results Summary (Historical Run 2 Metrics)

This file records the automated summary generated from the 13 Run 2 evaluation JSON files. It is retained as a historical analysis artefact. It is **not** the final 13-target comparison because the Run 2 UCI HAR execution used 150 epochs and was later superseded by the corrected 100-epoch UCI HAR experiments.

The Run 2 Opportunity and UniMiB-SHAR figures below describe their archived evaluation outputs. The final reproduction tables should be used for the consolidated results.

- UCI HAR: mean accuracy 89.51%, mean macro-F1 88.75%, mean weighted-F1 88.99% across 5 Run 2 target evaluation files (150-epoch execution; superseded).
- Opportunity: mean accuracy 81.16%, mean macro-F1 52.08%, mean weighted-F1 79.46% across 4 target experiments.
- UniMiB-SHAR: mean accuracy 52.23%, mean macro-F1 41.03%, mean weighted-F1 48.14% across 4 target experiments.
- UCI HAR: lowest mean class F1 values are class 3 (mean F1 0.780), class 1 (mean F1 0.865), class 4 (mean F1 0.892).
- Opportunity: lowest mean class F1 values are class 5 (mean F1 0.269), class 9 (mean F1 0.282), class 7 (mean F1 0.303).
- UniMiB-SHAR: lowest mean class F1 values are class 15 (mean F1 0.120), class 16 (mean F1 0.130), class 14 (mean F1 0.143).
- UCI HAR: largest observed max/min class-support ratio was 2.02 in target 0.
- Opportunity: largest observed max/min class-support ratio was 133.40 in target S2.
- UniMiB-SHAR: largest observed max/min class-support ratio was 73.00 in target 3.

## Interpretation

Macro-F1 weights classes equally; weighted-F1 weights classes by support. A gap between the two can indicate uneven per-class performance. Class-support imbalance is descriptive and does not prove the cause of errors. Numeric class IDs should not be translated into activity names without checking the dataset's label mapping.

These are post-hoc metrics from saved Run 2 evaluation files. The notebook that generated this summary did not train models; model training occurred during the earlier Colab experiments.