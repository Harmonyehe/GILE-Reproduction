# GILE Reproduction — Final Results

## Experiment coverage

Total evaluation records recovered: 13

The metrics below were recovered from archived evaluation JSON files.
They are not fresh evaluations of the latest UCI HAR checkpoints.

## Dataset-level performance

| Dataset     |   Accuracy |   Macro F1 |   Weighted F1 |
|:------------|-----------:|-----------:|--------------:|
| Opportunity |   0.804213 |   0.523264 |      0.793854 |
| UCI HAR     |   0.893419 |   0.878714 |      0.882422 |
| UniMiB-SHAR |   0.522324 |   0.410315 |      0.481426 |

## Experiment-level results

| Dataset     | Target   |   Samples |   Accuracy |   Macro F1 |   Weighted F1 | Source                   |
|:------------|:---------|----------:|-----------:|-----------:|--------------:|:-------------------------|
| UCI HAR     | 0        |       nan |   0.899135 |   0.893701 |      0.895563 | Archived evaluation JSON |
| UCI HAR     | 1        |       nan |   0.804636 |   0.742113 |      0.754356 | Archived evaluation JSON |
| UCI HAR     | 2        |       nan |   0.97654  |   0.975428 |      0.976609 | Archived evaluation JSON |
| UCI HAR     | 3        |       nan |   0.892744 |   0.888746 |      0.890381 | Archived evaluation JSON |
| UCI HAR     | 4        |       nan |   0.89404  |   0.893581 |      0.895203 | Archived evaluation JSON |
| Opportunity | S1       |     15643 |   0.826568 |   0.618367 |      0.821895 | Archived evaluation JSON |
| Opportunity | S2       |     15011 |   0.812404 |   0.535084 |      0.798548 | Archived evaluation JSON |
| Opportunity | S3       |     14456 |   0.781959 |   0.438229 |      0.763651 | Archived evaluation JSON |
| Opportunity | S4       |     12843 |   0.79592  |   0.501377 |      0.791323 | Archived evaluation JSON |
| UniMiB-SHAR | 1        |       384 |   0.463542 |   0.33828  |      0.402612 | Archived evaluation JSON |
| UniMiB-SHAR | 2        |       583 |   0.538593 |   0.47202  |      0.524619 | Archived evaluation JSON |
| UniMiB-SHAR | 3        |       304 |   0.671053 |   0.488121 |      0.657857 | Archived evaluation JSON |
| UniMiB-SHAR | 5        |       298 |   0.416107 |   0.34284  |      0.340614 | Archived evaluation JSON |

## Provenance and limitations

- All evaluation metrics originate from archived JSON files.
- The latest UCI HAR 100-epoch training histories are stored separately.
- Archived metrics must not be represented as fresh evaluations of those latest checkpoints.
- Opportunity S1 and S4 training completion should be checked against their logs.
- Existing confusion matrices and analysis tables have been preserved.
