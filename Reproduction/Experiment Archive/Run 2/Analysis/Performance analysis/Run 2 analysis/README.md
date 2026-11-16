# Run 2 — Main Experimental Execution

## Overview

Run 2 was the main experimental execution for the GILE reproduction. It includes target experiments for UCI HAR, Opportunity and UniMiB-SHAR. All model-training experiments were conducted in Google Colab using GPU acceleration.

| Dataset | Targets | Run 2 training duration | Role in final results |
|---|---|---:|---|
| UCI HAR | 0, 1, 2, 3, 4 | 150 epochs | Superseded by corrected 100-epoch UCI HAR execution |
| Opportunity | S1, S2, S3, S4 | 100 epochs each | Included |
| UniMiB-SHAR | 1, 2, 3, 5 | 100 epochs each | Included |

## Role in the final reproduction

During comparison with the paper, the UCI HAR training duration was reviewed. The Run 2 UCI HAR execution used 150 epochs, so it was not used for the final UCI HAR results. The five UCI HAR targets were subsequently run using the corrected 100-epoch configuration.

The final 13-target result set combines the corrected UCI HAR results with the Run 2 Opportunity and UniMiB-SHAR results. The Run 2 Opportunity S1–S4 and UniMiB-SHAR target experiments completed their 100-epoch runs.

## Analysis records

The Run 2 analysis includes target-level evaluation metrics, confusion matrices, classification reports, class-support summaries and automated findings. These artefacts describe different aspects of model performance:

- target-wise accuracy measures overall correct classification for each participant;
- per-class precision and recall help identify which activities are missed or over-predicted;
- per-class F1 balances precision and recall;
- confusion matrices show the distribution of true labels against predicted labels;
- class-support summaries show how many test examples contribute to each class metric.

A large gap between Macro-F1 and Weighted-F1 can indicate that performance is less consistent across classes than the overall support-weighted score suggests. Class imbalance is a possible context for interpretation, not proof of the cause of a model's errors.

## Evaluation protocol

The implementation evaluates target-domain performance during training and uses target-domain accuracy for checkpoint selection. Target labels therefore influence checkpoint selection, although they are not used for gradient-based parameter updates. This should be disclosed when discussing generalisation to unseen people.

## Final result location

The consolidated final results and analysis guide are in `Reproduction/Final Reproduction/README.md` and `Reproduction/Final Reproduction/ANALYSIS_GUIDE.md`.