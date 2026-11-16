# Run 2 — Main Experimental Execution

## Overview

Run 2 was conducted as the main experimental execution for the GILE reproduction. It covers the three wearable-sensor datasets and their target participants:

- UCI HAR: targets 0, 1, 2, 3 and 4.
- Opportunity: targets S1, S2, S3 and S4.
- UniMiB-SHAR: targets 1, 2, 3 and 5.

The Run 2 execution records document 13 target experiments. During comparison with the paper, the UCI HAR training duration was checked and aligned to 100 epochs for the final reproduction. The final UCI HAR results are therefore taken from the five target experiments conducted with the corrected 100-epoch configuration. Opportunity and UniMiB-SHAR results are taken from their Run 2 experiments.

## Role in the final reproduction

The final reproduction consists of the Run 2 Opportunity and UniMiB-SHAR results together with the UCI HAR results produced using the corrected 100-epoch configuration. These results are presented together as the final 13-target evaluation.

The Run 2 execution records also retain the UCI HAR 150-epoch attempt so that the experiment sequence remains traceable. That attempt is not the source of the final UCI HAR metrics.

## Run 2 execution configuration recorded in the logs

| Dataset | Targets | Recorded Run 2 duration |
|---|---|---:|
| UCI HAR | 0, 1, 2, 3, 4 | 150 epochs |
| Opportunity | S1, S2, S3, S4 | 100 epochs |
| UniMiB-SHAR | 1, 2, 3, 5 | 100 epochs |

The UCI HAR row above describes the Run 2 execution record. The final reproduction uses the separate UCI HAR execution with the corrected 100-epoch configuration.

## Analysis records

This directory contains the Run 2 execution and analysis records, including evaluation metrics, confusion matrices, classification reports, class-distribution summaries and experiment-level analysis. Consult the final reproduction tables for the metrics used in the completed cross-dataset comparison.

## Evaluation protocol

The implementation evaluates target-domain performance during training and uses target-domain accuracy for checkpoint selection. Target-domain labels therefore influence model selection, although they are not used for gradient-based parameter updates. Results should be interpreted in light of this protocol and should not be described as a strictly target-unseen model-selection procedure.

## Final result location

The consolidated final results, including the corrected UCI HAR results, are documented in:

`Reproduction/Final Reproduction/README.md`
