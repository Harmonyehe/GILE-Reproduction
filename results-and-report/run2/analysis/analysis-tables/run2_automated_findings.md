# GILE Run 2 — Automated Results Summary

This summary is generated from the 13 saved evaluation JSON files.

- UCI HAR: mean accuracy 89.51%, mean macro-F1 88.75%, mean weighted-F1 88.99% across 5 target experiments.
- oppor: mean accuracy 81.16%, mean macro-F1 52.08%, mean weighted-F1 79.46% across 4 target experiments.
- shar: mean accuracy 52.23%, mean macro-F1 41.03%, mean weighted-F1 48.14% across 4 target experiments.
- UCI HAR: lowest mean class F1 values are class 3 (mean F1 0.780), class 1 (mean F1 0.865), class 4 (mean F1 0.892).
- oppor: lowest mean class F1 values are class 5 (mean F1 0.269), class 9 (mean F1 0.282), class 7 (mean F1 0.303).
- shar: lowest mean class F1 values are class 15 (mean F1 0.120), class 16 (mean F1 0.130), class 14 (mean F1 0.143).
- UCI HAR: largest observed max/min class-support ratio was 2.02 in target 0.
- oppor: largest observed max/min class-support ratio was 133.40 in target S2.
- shar: largest observed max/min class-support ratio was 73.00 in target 3.

## Interpretation cautions
- Macro-F1 weights classes equally; weighted-F1 weights classes by support. A large gap can indicate uneven per-class performance.
- Class-support imbalance is descriptive and does not establish the cause of errors.
- Numeric class IDs have not been converted to activity names unless verified from dataset label documentation.
- These are post-hoc metrics from saved evaluations; no model training was performed by this notebook.
