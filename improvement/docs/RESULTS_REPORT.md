# Full experiment results — repaired Independent Excitation on UCI HAR (Person 3)

Date: 15 Nov 2026. 75 runs = 5 target persons × 3 variants × 5 seeds, 150 epochs each, Colab Tesla T4, torch 2.14.1+cu130.
Raw data: `runs_full/ucihar/<variant>/target<t>/seed<k>/` (config.json, history.csv, log.txt, metrics.json, selected_model.pt).
Tables: `runs_full/ucihar/{all_runs,summary,per_target,paired_differences,paired_target_level_CI}.csv`, `completeness.txt`.
Reproduce tables: `python aggregate.py --root runs_full` then `python checks/analyse_full.py`.
Design and the paper-vs-repair differences: `IE_IMPLEMENTATION_NOTES.md`.

**Research question.** Does restoring a differentiable Independent Excitation mechanism improve the separation of person-specific and activity-specific information, and does that translate into better recognition of unseen people?

**Short answer.** Partly on separation, not on recognition.
- The repaired IE gives a small, consistent reduction in *non-linearly* decodable person information in the activity code (MLP probe −3.8 points, all 5 targets). It also reduces activity information in the person code.
- *Linearly* decodable person information is unchanged.
- Recognition of unseen people does not detectably change (weighted-F1 −1.2, 95% CI −6.9 to +4.5). The effect is person-dependent: +3 to +4 on two targets and −3 to −7 on three.

## 1. Completion and integrity
- 75/75 runs completed; 0 missing, 0 duplicates, 0 runs with non-finite losses; "FAILED RUNS: none" in both Colab blocks.
- Code verified on Colab by SHA-256 against local files; UCI archive SHA-256 verified; `checks/test_variants.py` ALL PASSED on Colab.
- Per-run guards held in every run: train/val split overlap = 0; probe train/test shared windows = 0; target windows used in probes = 0.
- Reproducibility: target 0 / seed 0 in the full run reproduced the separate pilot session **exactly** (same selected epoch, metrics and probes for all 3 variants).

## 2. Primary results — target performance at the epoch selected on source-person validation
Mean over the 5 targets within each seed, then mean ± std over the 5 seeds.

| Variant | Accuracy | Macro-F1 | Weighted-F1 | Selected epoch |
|---|---|---|---|---|
| released (original code) | 78.15 ± 1.28 | 76.62 ± 1.37 | 76.77 ± 1.30 | 121 ± 15 |
| ie_off (single Adam, lr 1.1e-3, no IE) | 78.35 ± 1.65 | 76.93 ± 1.77 | 77.02 ± 1.87 | 122 ± 9 |
| ie_on (repaired IE, γ = 1000) | 77.35 ± 1.19 | 75.98 ± 1.17 | 75.80 ± 1.41 | 125 ± 6 |

Paired differences are computed per (target, seed), averaged within each target, then a 95% t-interval is taken over the 5 targets (df = 4). The target is the honest unit, because seeds of one person are not independent.

| Comparison | Accuracy | Macro-F1 | Weighted-F1 | Per-target weighted-F1 |
|---|---|---|---|---|
| ie_on − ie_off | −0.99 [−6.28, +4.29] | −0.96 [−6.64, +4.72] | −1.22 [−6.92, +4.48] | 0: −3.0, 1: −6.7, 2: +3.2, 3: −3.5, 4: +3.9 |
| ie_on − released | −0.80 [−6.27, +4.67] | −0.65 [−5.91, +4.62] | −0.97 [−6.49, +4.55] | 0: −3.5, 1: −7.2, 2: +3.0, 3: −0.4, 4: +3.2 |
| ie_off − released | +0.20 [−2.12, +2.51] | +0.31 [−1.65, +2.27] | +0.25 [−1.78, +2.28] | 0: −0.5, 1: −0.5, 2: −0.2, 3: +3.2, 4: −0.7 |

**Interpretation.**
- There is no detectable effect of the repaired IE on unseen-person recognition.
- `ie_off` ≈ `released` empirically, as predicted by the maths: released training is one Adam step at lr 1.1e-3.
- Per-run paired SD between these two mathematically identical procedures is about 4.8 points. That is the noise floor any real effect must exceed.

Per target (weighted-F1, mean ± std over seeds):

| Target | released | ie_off | ie_on |
|---|---|---|---|
| 0 | 77.98 ± 4.55 | 77.51 ± 3.62 | 74.49 ± 4.33 |
| 1 | 66.76 ± 3.31 | 66.23 ± 2.77 | 59.55 ± 3.08 |
| 2 | 92.10 ± 1.13 | 91.87 ± 1.35 | 95.10 ± 0.56 |
| 3 | 72.34 ± 7.95 | 75.51 ± 6.31 | 71.97 ± 5.34 |
| 4 | 74.65 ± 3.55 | 73.98 ± 2.21 | 77.87 ± 1.28 |

## 3. Separation of information (probes on the selected-epoch model)
- **Probe data.** Probes are fitted on source-train codes and scored on source-validation codes of the 4 source persons, using posterior means.
- **Chance level.** The person majority baseline is ≈ 27%.
- **Reading the numbers.** Lower is better (less leakage). The interval is the target-level 95% CI.

| Probe | released | ie_off | ie_on | ie_on − ie_off | ie_on − released |
|---|---|---|---|---|---|
| person ← activity code z, linear | 52.35 ± 1.80 | 52.31 ± 3.08 | 53.38 ± 1.75 | +1.07 [−3.06, +5.20] | +1.03 [−0.74, +2.80] |
| person ← activity code z, MLP | 64.88 ± 2.36 | 64.83 ± 2.34 | 61.07 ± 1.49 | **−3.76 [−6.36, −1.16]**, 5/5 targets lower | **−3.81 [−6.23, −1.39]**, 5/5 targets lower |
| activity ← person code z_d, linear | 69.91 ± 1.49 | 68.58 ± 2.56 | 66.79 ± 1.25 | −1.79 [−4.94, +1.35] | −3.12 [−6.70, +0.46] |
| activity ← person code z_d, MLP | 73.33 ± 2.06 | 72.20 ± 1.56 | 69.66 ± 1.21 | **−2.55 [−3.95, −1.15]**, 5/5 lower | **−3.67 [−5.86, −1.49]**, 5/5 lower |

**Interpretation.**
- IE reduced *non-linearly* decodable leakage in both directions, consistently across persons. It did not reduce *linearly* decodable person information in z.
- This is consistent with the predicted limitation of a frozen, linear cross-classifier: the encoder only has to make person information unreadable along the head's few directions (`IE_IMPLEMENTATION_NOTES.md` §4).
- Even with IE, the activity code still lets a probe recover the person far above chance: 53–61% against 27%.
- Caution on multiple comparisons. 27 interval comparisons were made, and one control comparison (`ie_off − released`, activity←z_d MLP: −1.13 [−2.24, −0.02]) only just excludes 0, even though the two procedures are mathematically identical. So effects of about 1 point should not be over-read. The ~3.7-point MLP-probe reductions are larger and consistent in 5/5 targets against both controls.

## 4. Secondary (descriptive stability only) and diagnostic numbers
These were not used for selection, tuning, or choosing a variant.

| Variant | SECONDARY: mean target weighted-F1 over final 10 epochs | Last-epoch weighted-F1 | DIAGNOSTIC: best-on-target ("leaky") weighted-F1 |
|---|---|---|---|
| released | 78.70 ± 0.99 | 78.79 ± 1.61 | 86.69 ± 1.41 |
| ie_off | 76.69 ± 1.11 | 77.03 ± 1.50 | 86.64 ± 0.77 |
| ie_on | 78.64 ± 0.75 | 78.93 ± 2.73 | 87.48 ± 0.64 |

- **Secondary.** ie_on − ie_off = +1.94 [−0.12, +4.01], but ie_on − released = −0.06 [−3.01, +2.88]. The two equivalent controls disagree, so this is not evidence for IE. The seed-to-seed SD of ie_on is somewhat lower, which is a weak descriptive hint of extra stability.
- **Diagnostic: target leakage.** Selecting the epoch on the held-out person's own accuracy (the released protocol) inflates weighted-F1 by about **10 points** in the same runs (≈ 86.7–87.5 vs ≈ 75.8–77.0). The leaky numbers are close to the paper's 88.17 and the team's Run 2 (88.99). Most of the gap between the paper and an honest protocol is therefore model selection on target data.

## 5. Training behaviour and runtime
- All losses were finite and decreased smoothly. The γ·IE term stayed a minor share of the loss (pilot: 5% → 15%), and IE values fell quickly toward the uniform optimum.
- Selected epochs ranged from 64 to 150; 19/75 runs selected epoch ≥ 140. Some runs may still have been improving at 150 epochs; this is a limitation, kept identical across variants.
- Runtime: about 85–95 s training, ≈ 95–110 s wall per run; 75 runs ≈ 2 h 10 min on one T4, plus ≈ 4 min queue for the GPU.

## 6. Limitations
1. Only 5 persons (UCI HAR subjects 1–5, as in the paper), so the intervals are wide. One dataset; Opportunity and UniMiB were not run.
2. The released channel-mixing `.view()` bug and the person-agnostic prior (BatchNorm on single-person batches) are kept unchanged in all variants.
3. The repaired IE is a modification (KL-to-uniform, detached heads, source-only person classes, one optimiser), not the paper's literal two-optimiser −CE scheme. γ = 1000 was fixed a priori and not tuned.
4. Validation windows are the temporally last 20% of each activity, so the probes and selection measure generalisation to later segments of the same people.
5. The Wilcoxon p-values in `paired_differences.csv` treat (target, seed) pairs as independent and are descriptive only. Use the target-level CIs above.

## 7. Errors encountered (none affected training or results)
- Colab browser downloads: the first 352 MB download failed while training was still running; later downloads were blocked by Chrome's multiple-automatic-downloads protection until the user allowed it. Results were then retrieved intact (SHA-256 recorded, zip integrity tested).
- Helper cells added to the Colab copy of the notebook (re-download, per-seed zip; one failed with an auto-indent error, then fixed as a one-liner). No local code changed during the runs.
