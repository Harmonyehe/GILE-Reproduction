# Run 1 — Original Experiment Snapshot

## 1. Overview

Run 1 records the original GILE reproduction experiment from the
historical Git commit:

`46b5ef37b84c9ba82f9bfd5e2be2b45a860067ba`

Commit message: Add GILE reproduction code and experimental results.

This experiment snapshot predates the later Run 2 experiments and
the Final Reproduction organisation.

## 2. Experimental status

The Opportunity experiments for participants S1 and S4 were
interrupted, according to the experiment history.

The original S1 and S4 files are preserved in the repository.
Their historical status must not be confused with results from
later experiments.

The original CSV files and logs should be interpreted using the
historical commit as their source of truth.

## 3. Repository organisation

| Folder | Purpose |
|---|---|
| Code | Original experiment code |
| Results | Original dataset training outputs |
| Evaluation | Metrics, summaries and visualisations |
| Logs | Original experiment training logs |

## 4. Avoiding duplicate files

Files are not copied into Run 1 when identical content already
exists elsewhere in the repository.

Instead, `RUN1_FILE_INDEX.csv` records:

- The original path in the historical Git commit.
- Its historical category.
- Whether the file is present in the current repository.
- Its current physical location.
- Its SHA256 checksum.

This preserves the historical experiment without creating
unnecessary duplicate files.

## 5. Historical source

Original commit: `46b5ef37b84c9ba82f9bfd5e2be2b45a860067ba`

The original Git commit is the authoritative source for this
snapshot.

## 6. Relationship to other experiments

- Original Code contains the original source-code collection.
- Run 1 records the earlier experiment.
- Experiment Archive/Run 2 contains the later experiment.
- Final Reproduction contains the reorganised final implementation.

These collections must not be treated as interchangeable.

## 7. Important files

- `RUN1_FILE_INDEX.csv` — permanent file-location and integrity index.
- `RESTORATION_MANIFEST.csv` — record of the initial restoration attempt.

## 8. Reproducibility note

A file being present in another experiment archive does not,
by itself, establish that it was generated during that experiment.

Historical attribution must be based on the original commit and
the associated experiment records.
