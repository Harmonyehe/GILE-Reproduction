# Evaluation and cleanup audit

Generated: 2026-11-16T13:02:06.143720+00:00

## Scope
This notebook performed repository inspection, metric consolidation and analysis only. It did not invoke model training or run any training epochs.

## Provenance
- Fresh UCI HAR metrics are selected from the Fresh CPU Evaluation folder when present.
- Opportunity and UniMiB-SHAR metrics are labelled archived unless a fresh evaluation source is explicitly found.
- Missing values remain blank; no metrics are fabricated.
- `experiment_evidence_audit.csv` lists metric JSONs, matching checkpoints and logs found.

## Limitations
The repository audit must be reviewed before describing all 13 experiments as independently evaluated. A metric JSON alone is not proof that the corresponding model checkpoint was freshly evaluated.
