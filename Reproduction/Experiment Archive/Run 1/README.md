# Run 1 — Initial Experimental Attempt

## Purpose and status

Run 1 was the initial attempt at reproducing GILE. It was interrupted and did not provide the final, consistent experiment set. It is retained to document the development of the work, but its results are excluded from the final 13-target metrics.

A key reason for discarding Run 1 was the UCI HAR configuration: it used 150 epochs, whereas the final reproduction was standardised to 100 epochs. The interrupted execution and the mismatch in training duration meant Run 1 was not used as the final result source.

All model-training experiments were conducted in Google Colab with GPU acceleration.

## Relationship to the final experiments

The final reproduction uses:
- corrected 100-epoch UCI HAR experiments for targets 0–4;
- Run 2 Opportunity experiments for S1–S4, each trained for 100 epochs;
- Run 2 UniMiB-SHAR experiments for targets 1, 2, 3 and 5, each trained for 100 epochs.

Run 1 logs, outputs and source records remain available for historical inspection. Do not merge its metrics with the final analysis tables.

## Directory map

| Folder | Contents |
|---|---|
| `Code/` | Code snapshot associated with the initial attempt |
| `Results/` | Available Run 1 outputs |
| `Evaluation/` | Evaluation artefacts from the initial attempt |
| `Logs/` | Execution and training logs |

`RUN1_FILE_INDEX.csv` records original paths, current locations and SHA256 checksums for indexed files.

Historical source snapshot: `46b5ef37b84c9ba82f9bfd5e2be2b45a860067ba`.

For the final configuration and results, see `Reproduction/Final Reproduction/README.md`.