# GILE: Cross-Person Human Activity Recognition

This repository contains the original GILE implementation, the completed reproduction of the method introduced by Qian, Pan and Miao (AAAI 2021), and an evaluated improvement to the method.

**Paper:** *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*
**Authors:** Hangwei Qian, Sinno Jialin Pan and Chunyan Miao
**Venue:** AAAI 2021

## 1. Project overview

GILE addresses cross-person human activity recognition (HAR) from wearable-sensor time series. The central reproduction question is whether a model trained using data from source participants can recognise activities for a different target participant. The target participant is treated as a separate domain because sensor patterns can vary between people even when they perform the same activity.

The reproduction covers three datasets and 13 target participants:

| Dataset | Target participants | Targets |
|---|---|---:|
| UCI HAR | 0, 1, 2, 3, 4 | 5 |
| Opportunity | S1, S2, S3, S4 | 4 |
| UniMiB-SHAR | 1, 2, 3, 5 | 4 |

## 2. Experimental history and final selection

All model-training experiments were run in Google Colab using GPU acceleration.

**Run 1 — excluded.** The initial attempt was interrupted, and its UCI HAR execution used 150 epochs rather than the intended 100-epoch configuration. Run 1 is retained as a historical record, but its outputs are not included in the final metrics.

**Run 2 — main execution.** Opportunity and UniMiB-SHAR were run for 100 epochs for every target. The Run 2 UCI HAR experiments used 150 epochs, so those UCI HAR results were not retained as the final UCI HAR results.

**Corrected UCI HAR execution.** After comparing the experiment configuration with the paper, UCI HAR was rerun for targets 0–4 using 100 epochs.

**Final 13-target set.** The final results combine the five corrected UCI HAR target experiments with the four Opportunity and four UniMiB-SHAR Run 2 target experiments. This is the basis for the final cross-dataset comparison.

The report-generation and evaluation-consolidation notebooks did not train models. That statement describes those later analysis steps only; it does not mean the reproduction project lacked training.

## 3. Final dataset-level results

The following are unweighted means across target participants within each dataset. The paper comparison is for accuracy only.

| Dataset | Paper accuracy (%) | Reproduced accuracy (%) | Difference (percentage points) | Macro-F1 (%) | Weighted-F1 (%) |
|---|---:|---:|---:|---:|---:|
| UCI HAR | 88.172 | 88.426 | +0.254 | 87.125 | 87.287 |
| Opportunity | 81.395 | 80.421 | -0.974 | 52.326 | 79.385 |
| UniMiB-SHAR | 55.725 | 52.232 | -3.493 | 41.032 | 48.143 |

The F1 values describe the reproduction. Paper F1 baselines are not asserted.

## 4. How to read the analysis

- **Target-wise tables** show performance for each held-out participant separately. They reveal whether the average is consistent or driven by a subset of targets.
- **Dataset-level tables** are unweighted averages of the target-level values. Each target contributes equally, rather than each test example contributing equally across targets.
- **Accuracy** is the fraction of correctly classified test examples.
- **Macro-F1** computes F1 per class and averages classes equally. It is sensitive to poor performance on less frequent classes.
- **Weighted-F1** averages class F1 using each class's support. It is more influenced by common classes.
- **Confusion matrices** show the relationship between true and predicted labels. Rows and columns should be interpreted using the dataset's verified label mapping.
- **Training-history plots** show the values recorded during training. They are not a substitute for final sample-level evaluation.
- **Paper comparisons** use accuracy values transcribed for the corresponding targets. Differences are reported in percentage points, not relative percent.

A large gap between weighted-F1 and macro-F1 can indicate uneven class-level performance. Class-support imbalance is descriptive; it does not, by itself, establish why a model makes particular errors.

## 5. Evaluation protocol and limitations

The implementation evaluates target-domain performance during training and uses target-domain accuracy for checkpoint selection. Target labels therefore influence checkpoint selection, even though they are not used for gradient-based parameter updates. The results should not be described as a strictly target-unseen model-selection evaluation.

The original training-accuracy calculation uses `len(predictions_y) * batch_size` as its denominator. Because the final batch can be smaller than the configured batch size, historical training accuracy can be underestimated. The final evaluation metrics use the actual number of evaluated samples. See `Reproduction/Final Reproduction/Analysis/Audit/accuracy_calculation_audit.txt`.

The final UCI HAR metrics come from fresh CPU evaluation JSONs. This CPU evaluation is an evaluation step; model training was performed on Colab GPU. Opportunity and UniMiB-SHAR final metrics use the Run 2 evaluation outputs.

## 6. Improvement work (UCI HAR)

The `improvement/` folder contains the proposed improvements and their evaluation. All improvement experiments use **source-only model selection**:
- each source participant is split into training and validation data, with overlapping windows removed;
- the epoch is chosen on source validation accuracy;
- the held-out participant is never used for training, selection or tuning.

**Findings from auditing the released code:**
- **Independent Excitation (IE) has no effect in the released code.** Its loss is computed from detached predictions, so it sends no gradient to any weight. The released training is therefore effectively GILE without IE.
- **The encoders scramble the input.** They reshape each `(128 time × 9 sensor)` window with `.view()` instead of a transpose, so every convolution window mixes five different sensors instead of following one sensor over time. The decoder shows that the intended layout is channels = sensors, width = time.
- **Target-based checkpoint selection (section 5) inflates results.** In our runs it raised UCI HAR weighted-F1 by about 10 points compared with source-only selection.

**Improvements tested.** All runs on a Colab Tesla T4, 150 epochs, UCI HAR targets 0–4.

| Improvement | Runs | Result vs control |
|---|---:|---|
| Repaired, differentiable IE | 75 | −0.99 accuracy; lowered non-linear person information in the activity code, but no accuracy gain |
| XPC-B: cross-person contrastive loss with person-balanced batches | 50 | +1.39 vs its control; inconclusive under the predeclared rule |
| **Input-layout fix + person-style sensor augmentation** | 45 | **79.17% → 91.61% accuracy** |

The success rule for the final experiment was written before any results. Results below are for seeds 0–2, accuracy at the source-selected epoch:

| Arm | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Δ accuracy vs control [95% CI] | Targets improved |
|---|---:|---:|---:|---|---:|
| Control (released GILE behaviour) | 79.17 | 77.86 | 77.97 | — | — |
| Layout fix | 87.63 | 86.63 | 86.49 | +8.46 [+4.99, +11.93] | 5/5 |
| Augmentation only | 83.16 | 81.81 | 81.97 | +3.99 [−1.50, +9.48] | 4/5 |
| **Layout fix + augmentation** | **91.61** | **91.30** | **91.25** | **+12.44 [+6.08, +18.80]** | **5/5** |

**Not directly comparable with section 3.** The improvement numbers use source-only selection and 150 epochs; the reproduction numbers in section 3 use target-based selection and 100 epochs. Under target-based selection, our control reaches about 87% accuracy, comparable to the reproduced 88.4%.

**Limitations:**
- UCI HAR only;
- 3 seeds: seed 3 was partial and excluded, and seed 4 was not run because the Colab GPU quota was exhausted;
- 5 participants, so the confidence intervals are wide.

Full details, per-target tables, statistics and limitations are in `improvement/docs/FINAL_REPORT.md`.

## 7. Repository map

- `Original Code/` — source implementation supplied with the paper.
- `Reproduction/Final Reproduction/` — final experiment results, evaluation artefacts, analysis, setup and methodology documentation.
- `Reproduction/Experiment Archive/Run 1/` — interrupted initial attempt, retained for traceability and excluded from final metrics.
- `Reproduction/Experiment Archive/Run 2/` — main execution records and analysis.
- `Reproduction/Experiment Archive/Source Archive/` — source snapshots and supporting records.
- `improvement/` — improvement work. It contains:
  - `docs/`: reports;
  - `experiment/`: code, tests, per-run results, analysis and the Colab notebook;
  - `earlier_stages_code/`: the IE and XPC-B experiments.

Start with:
1. `Reproduction/Final Reproduction/README.md` for the final experiment scope and results.
2. `Reproduction/Final Reproduction/SETUP.md` for environment, data layout and commands.
3. `Reproduction/Final Reproduction/ANALYSIS_GUIDE.md` for interpretation of the evaluation artefacts.
4. `improvement/README.md` for the improvement results, setup and reproduction commands.
5. `improvement/docs/FINAL_REPORT.md` for the full improvement report.

Raw datasets are not included. Follow the setup guide and original dataset preparation instructions before attempting a new run.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/QianPM21,
  author    = {Hangwei Qian and Sinno Jialin Pan and Chunyan Miao},
  title     = {Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {11921--11929},
  year      = {2021}
}
```
