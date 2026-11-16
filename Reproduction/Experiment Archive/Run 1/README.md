# Run 1 — Initial Experimental Attempt

## Overview

Run 1 records the first experimental attempt in the GILE reproduction project. The attempt was interrupted before the complete set of experiments required for the final evaluation had been obtained.

The original experiment files, logs and available outputs are retained here so that the development of the reproduction can be traced. These records are not used to calculate the final 13-target results.

## Experimental status

The Run 1 records include incomplete experiments. In particular, the Opportunity S1 and S4 logs document interrupted execution. UCI HAR also includes an experiment that reached epoch 150; this was not the configuration used for the final UCI HAR results.

Run 1 should therefore be read as an account of the initial attempt, not as the completed reproduction.

## Repository organisation

| Folder | Contents |
|---|---|
| `Code/` | Source code associated with the initial attempt |
| `Results/` | Available experiment outputs |
| `Evaluation/` | Evaluation artefacts and summaries |
| `Logs/` | Training logs and execution records |

## File index and provenance

Some files are represented elsewhere in the repository rather than duplicated in this folder. `RUN1_FILE_INDEX.csv` records their original paths, current locations and SHA256 checksums.

The historical Git commit below provides the source snapshot for this experiment:

`46b5ef37b84c9ba82f9bfd5e2be2b45a860067ba`

## Relationship to the final reproduction

The final reproduction was completed through the subsequent Run 2 experiments together with the UCI HAR experiments run using the corrected 100-epoch configuration. Run 1 remains available to document the initial attempt and is not combined with the final metrics.
