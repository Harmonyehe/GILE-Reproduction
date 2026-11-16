# Proposed improvements and their evaluation (GILE, cross-person HAR on UCI HAR)

**Scope:** proposing, implementing and evaluating improvements to GILE.
**Status labels:** [MEASURED] = from our runs; [CODE] = verified by reading code or by a test; [REPORTED] = claimed in a
paper; [HYPOTHESIS] = our interpretation, not tested.

**Evidence base, all on Colab Tesla T4 with torch 2.14.1+cu130, UCI HAR persons 0–4, leave-one-person-out:**
| Stage | Runs | Arms |
|---|---|---|
| Stage 1 | 75 | `released`, `ie_off`, `ie_on` |
| Stage 2 | 50 + 1 anchor | `ie_off_bal`, `xpc_bal` |
| Stage 3 (this report's main experiment) | 75 + 1 anchor | `fix`, `aug`, `fix_aug` |

**Shared protocol:**
- the epoch is selected on source-person validation only;
- the target person is never used for training, selection or tuning;
- 5 seeds;
- inference is at target level (n = 5 persons, paired t, df = 4).

## 1. Problem identified: an audit of GILE and of our own earlier attempts
| # | Question | Finding | Evidence |
|---|---|---|---|
| 1 | What does the released Independent Excitation (IE) do? | **Nothing.** `loss_function_false` builds its loss from classifier outputs computed under `torch.no_grad()` and then sets `requires_grad = True`. 0 of 53 parameter groups receive a gradient. The second optimiser only re-applies the main loss's stale gradients, so released training = one Adam at an effective lr of 1.1e-3. | [CODE] `Person3_checks/ie_gradient_check.py`; [MEASURED] `ie_off` ≈ `released` (+0.20 accuracy, CI −2.12 to +2.51) |
| 2 | What did the repaired IE change? | It cut **non-linearly** decodable person information in z: MLP probe −3.8, lower for 5/5 persons. **Linear** person information was unchanged (+1.1). Unseen-person accuracy −1.0 (CI −6.3 to +4.3): +3 to +4 for two persons, −3 to −7 for three. | [MEASURED] `RESULTS_REPORT.md` |
| 3 | Why did less person information not give higher accuracy? | IE only makes person identity unreadable along the frozen heads' directions; the person information stays linearly decodable (53% against 27% chance). Also, accuracy on a new person depends on whether that person's activity patterns fall inside the source data, which removing person information from z does not change. | [MEASURED] probes; second sentence [HYPOTHESIS] |
| 4 | Why did XPC-B fail its success rule? | `xpc_bal − ie_off_bal` accuracy +1.39 (CI −2.24 to +5.01), up for only 3/5 persons. The person probes did not fall (linear −0.33, MLP +0.19). | [MEASURED] `XPCB_FINAL_REPORT.md` |
| 5 | Did the projection head absorb the contrastive objective? | Consistent with that, but not proven. The loss acted on g(μ). The XPC term reached 78% of the loss, yet person information in z did not change. Head outputs were not saved, so g(z) and z could not be compared directly. | [MEASURED] + [HYPOTHESIS] |
| 6 | Did balanced batching help? | No. `ie_off_bal − ie_off` −1.81 (CI −8.48 to +4.85); person 1 −10.1. | [MEASURED] |
| 7 | Is the encoder `.view()` a real bug? | **Yes.** `x.view(-1, 9, 1, 128)` on a contiguous `(B,128,9,1)` batch reinterprets memory. Each "channel" holds about 14 time steps × all 9 sensors, interleaved, and every width-5 convolution window spans 5 *different sensors* at nearly the same instant. The decoder outputs `(B,9,1,128)` and permutes back, which shows that the intended layout is channel = sensor, width = time. | [CODE] tests C1–C4, C6 (`experiment/code/stage3/checks/test_stage3.py`) |
| 8 | Does the person-specific prior work? | No, during training. Batches contain one person, and the BatchNorm in `pzd` normalises a one-hot input that is identical across the batch, so every person gets the same prior. Mixed batches restore person-specific priors, but mixing (row 6) did not help accuracy. | [CODE] `checks/test_variants.py` test 6; [MEASURED] row 6 |
| 9 | Does the protocol bias the results? | **Yes, strongly, in the released code:** the best epoch is chosen on the held-out person's own accuracy. In the same runs this inflates weighted-F1 by about 10 points (≈ 86.7 vs ≈ 77.0), which explains most of the gap to the paper's 88.17. Also, training accuracy divides by n_batches·64. All our results use source-only selection. | [MEASURED] `RESULTS_REPORT.md` §4; [CODE] `train.py` |

**Supported by measurement:** rows 1, 2, 4, 6 and 9 (effects) and 7 (bug). **Hypotheses:** the mechanisms in rows 3
and 5.

**Conclusion of the audit.**
- Both earlier improvements were *representation-regularisation* losses: they reshape z with an extra objective.
- Neither moved unseen-person accuracy beyond the noise floor. The per-run paired SD between two mathematically
  identical procedures is about 4.8 points.
- Meanwhile the **input representation itself was broken** (row 7), and the model only ever sees four people's styles
  (row 3).
- Stage 3 therefore tests two changes aimed directly at what the encoder sees, rather than at z:
  - (C) fix the broken input layout so the convolutions work over time, per sensor;
  - (B) widen the range of person styles seen in training with physically plausible augmentation.

## 2. Candidates considered and ranking
| Criterion | A: direct cross-person SupCon on z | B: person-style augmentation | C: input-layout correction |
|---|---|---|---|
| Expected unseen-person accuracy gain | low: XPC-B, the closest relative, gave +1.39 with 3/5 up | medium: augmentation alone added +9.21 on DSADS in DDLearn's ablation [REPORTED, P13]; Um et al. 77.5 → 86.9 on a Parkinson task [REPORTED, P15] | unknown but potentially large: the convolutions currently do not see temporal patterns per sensor at all [CODE] |
| Scientific justification | removes the head-absorption hypothesis (row 5) | models between-person variation in intensity and placement | correctness: restores the authors' intended architecture |
| Published HAR evidence | SupCon for DG HAR: DDLearn [P13] (contrastive term not isolated) | strong: P13, P14 (DIVERSIFY: Mixup strong), P15 | none needed: it is a bug fix |
| Compatibility with GILE | needs mixed-person batches, which hurt by −1.81 (row 6): confounded | full; no loss change | full; one line, same parameters |
| Expected effect on person leakage | intended to lower it | unclear; it could lower person information in z | unknown |
| Implementation complexity | moderate (λ re-calibration, batching) | low | very low |
| GPU runtime | as XPC-B | about +0–10% (CPU augmentation) | unchanged |
| Risk of destabilising training | high (XPC share grew to 78%) | low (small, bounded transforms) | low |
| Non-trivial contribution | incremental on XPC-B | yes (method + physical design) | yes: a genuine defect in the published code, with a measured effect |

**Decision.**
- Run **C and B together as a 2 × 2 factorial** against the existing, bit-reproducible `ie_off` control. The factorial
  isolates each change's contribution and their interaction.
- **A was not run.** The 50-run XPC-B evidence makes another contrastive variant the weakest bet for accuracy. It would
  need the mixed batches that hurt, and GPU time on the deadline day was limited.
- A remains the logical follow-up for the separation question (section 8).

## 3. Proposed method
**C: layout correction.**
- Replace `x.view(-1, x.shape[2], 1, x.shape[1])` with `x.permute(0, 2, 3, 1)` in the encoders (`qzd`, `qzy`).
- Each of the 9 input channels is then one sensor axis over 128 time steps. The (1×5) kernels become temporal filters
  shared across time, as in a standard 1-D temporal CNN for HAR.

**B: person-style augmentation** (training windows only; p = 0.5 per window; raw units). It applies:
- movement-intensity scaling s ~ N(1, 0.1²), clipped to [0.7, 1.3], on body_acc and body_gyro, with total_acc adjusted
  consistently so that gravity is unchanged;
- one small 3-D rotation (at most 15°) applied identically to all three sensor triads, which models phone placement and
  preserves norms and the gravity magnitude;
- Gaussian noise σ = 0.01.

All values were fixed a priori from Um et al. [P15], never tuned. Time-warping and large rotations were excluded
because they could change the activity label (cadence; the gravity orientation that separates the postures).

## 4. Technical implementation (details: `IMPLEMENTATION_NOTES.md` §3)
- **Isolation.** Everything is in `improvement-code/experiment/`. The shared modules are byte-identical copies.
- **The layout fix** replaces `forward` on the encoder *instances* with an exact copy of the released forward, except the
  reshaping line. Test C6 proves that the copy is otherwise exact (bit-identical outputs).
- **The augmentation** has its own RNG stream. Initial parameters, batch order and VAE noise are identical to the
  control (tests A5, A6, R2).
- **The runner** keeps only the control's training path of the tested stage-2 runner. Test R1: its `ie_off` arm
  reproduces the stage-2 runner exactly.
- **Resume:** per-epoch atomic checkpoint (test R4).
- **Tests:** 20 of 20 passed on the laptop CPU and on the Colab T4.
- **Reproducibility anchor on the T4:** `ie_off` target 0 seed 0 reproduced stage 1 exactly (selected epoch 99,
  accuracy 74.0634, weighted-F1 73.9257), so the existing 25 control runs are valid.

## 5. Experimental methodology (predeclared in `experiment/PREDECLARED_PLAN.md`, written before any result)
- **Arms:**
  - `ie_off` (control: 25 existing runs);
  - `fix`, `aug`, `fix_aug` (25 new runs each);
  - all on 5 targets × seeds 0–4, 150 epochs.
  - Same split, sampler, optimiser, initialisation, selection and probes.
- **Primary outcome:** target accuracy at the epoch selected on source validation.
- **Success rule, per arm vs `ie_off`:**
  - IMPROVEMENT SUPPORTED if the mean Δaccuracy > 0, accuracy is higher for at least 4 of 5 persons, and the mean
    Δmacro-F1 > 0;
  - STRONG if, in addition, the 95% target-level CI excludes 0 and the Holm-adjusted p < 0.05;
  - HARMFUL if the mean Δ < 0 and accuracy is lower for at least 4 of 5 persons;
  - otherwise INCONCLUSIVE.
- **Secondary:**
  - macro- and weighted-F1;
  - per-person tables;
  - person ← z linear and MLP probes (a lower probe is *not* counted as an accuracy gain);
  - stability;
  - runtime;
  - factorial main effects and interaction.

**What was run (status of every seed).**
| Seed | Status | Used in |
|---|---|---|
| 0, 1, 2 | **Complete:** 15 runs each, all files downloaded and verified | **the main analysis** (45 new runs vs the matching 15 `ie_off` runs: same seeds, same initial weights) |
| 3 | **Partial, excluded.** The free Colab GPU quota of the first account ended after 10 of 15 runs; their printed metrics are recovered, but no files exist. A second attempt on another account was started on 16 Nov ~20:50 and **stopped on the author's instruction** before any seed-3 output was retrieved or used. | descriptive table §6.6 only |
| 4 | **Not run** (GPU quota exhausted) | — |

- Seeds 0–2 were fixed in advance as the minimum design (`PREDECLARED_PLAN.md` §5).
- Deviations are logged in `PREDECLARED_PLAN.md` §8.
- **Integrity:** all 46 returned runs (45 + anchor) pass `tools/verify_runs_p3.py` V1–V10:
  - arm flags correct;
  - target never among the sources;
  - split and probe leak guards zero;
  - environment and code hashes match;
  - identical initial parameters across arms;
  - selection rule correctly applied;
  - no resumed runs;
  - checkpoints present;
  - 230 manifest files re-hashed.
- **Stability:** no failed runs and no non-finite losses.

## 6. Results [MEASURED]
### 6.1 Arm means (target person at the source-val-selected epoch; seeds 0–2, 15 runs per arm)
| Arm | Accuracy | Macro-F1 | Weighted-F1 | Last-epoch accuracy | Person ← z probe, linear | Person ← z probe, MLP | Mean SD over seeds | *Leaky best-target accuracy (diagnostic)* |
|---|---|---|---|---|---|---|---|---|
| `released` (original code) | 78.75 | 77.17 | 77.28 | 80.21 | 52.94 | 65.93 | 4.21 | *87.01* |
| `ie_off` (**control**) | 79.17 | 77.86 | 77.97 | 77.67 | 52.93 | 65.91 | 2.27 | *87.06* |
| `ie_on` (stage 1, repaired IE) | 76.58 | 75.18 | 74.81 | 80.14 | 54.52 | 61.37 | 2.69 | *88.31* |
| `ie_off_bal` (stage 2) | 76.37 | 74.91 | 74.63 | 77.24 | 54.83 | 65.18 | 2.86 | *86.97* |
| `xpc_bal` (stage 2, XPC-B) | 77.93 | 76.78 | 76.39 | 77.04 | 54.08 | 65.66 | 3.06 | *85.64* |
| **`fix`** (layout corrected) | **87.63** | **86.63** | **86.49** | 91.44 | 53.58 | 63.10 | 5.52 | *95.84* |
| `aug` (augmentation) | 83.16 | 81.81 | 81.97 | 84.28 | 52.15 | 62.06 | 4.19 | *90.74* |
| **`fix_aug`** (both) | **91.61** | **91.30** | **91.25** | 91.89 | **47.39** | **59.31** | 2.17 | *96.20* |

- Chance for the person probe is ≈ 27%.
- The earlier arms are restricted to the same seeds 0–2. Their 5-seed values are in the earlier reports.

### 6.2 Per person (accuracy, mean over seeds 0–2)
| Person | `ie_off` | `fix` | `aug` | `fix_aug` | `fix_aug` per seed |
|---|---|---|---|---|---|
| 0 | 78.58 | 88.28 | 81.36 | 90.39 | `analysis_p3/per_target.csv` |
| 1 | 69.54 | 82.34 | 80.91 | 89.74 | ″ |
| 2 | 91.59 | 98.34 | 91.20 | 97.85 | ″ |
| 3 | 80.76 | 86.86 | 82.97 | 94.43 | ″ |
| 4 | 75.39 | 82.34 | 79.36 | 85.65 | ″ |

### 6.3 Predeclared contrasts against `ie_off` (paired, target level, n = 5, df = 4)
| Arm | ΔAccuracy [95% CI] | Persons up | p (Holm) | ΔMacro-F1 [95% CI] | ΔWeighted-F1 [95% CI] | **Predeclared verdict** |
|---|---|---|---|---|---|---|
| `fix` | **+8.46 [+4.99, +11.93]** | 5/5 | 0.002 (0.007) | +8.76 [+6.39, +11.13] | +8.52 [+5.85, +11.19] | **STRONG IMPROVEMENT** |
| `aug` | +3.99 [−1.50, +9.48] | 4/5 | 0.114 (0.114) | +3.95 [−1.82, +9.72] | +3.99 [−1.78, +9.77] | **IMPROVEMENT SUPPORTED** (CI includes 0, so not strong) |
| `fix_aug` | **+12.44 [+6.08, +18.80]** | 5/5 | 0.006 (0.011) | +13.44 [+5.87, +21.00] | +13.28 [+5.75, +20.81] | **STRONG IMPROVEMENT** |

**Per-person ΔAccuracy:**
- `fix`: +9.70, +12.80, +6.74, +6.10, +6.95.
- `aug`: +2.79, +11.37, −0.39, +2.21, +3.97.
- `fix_aug`: +11.82, +20.20, +6.26, +13.67, +10.26.

**Largest gain:** person 1, the hardest person for every earlier method (69.5 → 89.7 with `fix_aug`).

### 6.4 Ablation: factorial effects (descriptive)
| Effect | Accuracy [95% CI] | Positive for | Macro-F1 | MLP person probe |
|---|---|---|---|---|
| Layout correction (main effect) | **+8.46 [+6.22, +10.69]** | 5/5 | +9.13 [+6.66, +11.60] | −2.78 [−5.45, −0.11] |
| Augmentation (main effect) | +3.98 [−0.49, +8.46] | 4/5 | +4.31 [−1.31, +9.93] | −3.82 [−8.82, +1.17] |
| Interaction | −0.01 [−4.19, +4.18] | 1/5 | +0.73 [−2.96, +4.41] | +0.06 [−4.29, +4.41] |

**Reading.**
- The two changes are **additive**: the interaction is about 0. The layout fix gives about two-thirds of the combined
  gain, and augmentation about one-third.
- The layout effect is large and consistent. The augmentation effect is positive for 4 of 5 persons, but its interval
  includes 0.

### 6.5 Person information in the activity code (secondary; not part of the success rule)
| Arm − `ie_off` | Linear probe [95% CI] | MLP probe [95% CI] | Persons lower (MLP) |
|---|---|---|---|
| `fix` | +0.65 [−7.46, +8.75] | −2.81 [−6.48, +0.86] | 4/5 |
| `aug` | −0.78 [−6.21, +4.65] | −3.85 [−8.34, +0.63] | 4/5 |
| `fix_aug` | −5.55 [−11.45, +0.36] | **−6.60 [−10.30, −2.90]** | **5/5** |

- `fix_aug` gives the largest reduction of both probes in any stage (linear −5.6, lower for 4/5 persons).
  `aug` alone lowers both only slightly (linear −0.8, MLP −3.9), with both intervals including 0.
- Its MLP-probe reduction (−6.6, all 5 persons) is larger than that of the repaired IE (−3.8), and it comes **with** a
  large accuracy gain, whereas IE's came with none.
- Person information is still far above chance (47% against 27%), so the representation is **not** person-invariant.

### 6.6 Partial seed 3 (descriptive only; printed metrics recovered from the notebook, no files)
| Arm | Person 0 | Person 1 | Person 2 | Person 3 |
|---|---|---|---|---|
| `fix` − `ie_off` | +7.21 | +4.31 | +8.50 | +11.04 |
| `aug` − `ie_off` | +4.90 | +8.28 | −3.23 | — |
| `fix_aug` − `ie_off` | −4.61 | +15.23 | +6.45 | — |

- `fix` is ahead in all 4 completed comparisons, as in seeds 0–2.
- `fix_aug` has one clear loss (person 0, −4.6). This is a reminder that single runs vary by about ±5 points.

## 7. Computational cost and training behaviour
- **Same model size and FLOPs.** The layout fix changes only how the input tensor is indexed. Augmentation acts on the
  CPU input batch.
- **Measured cost** (laptop CPU, 4 threads, ms per training step at batch 64, same code):

  | Arm | ms per step | Overhead |
  |---|---|---|
  | `ie_off` | 661 | — |
  | `fix` | 691 | +4.5%, from copying the permuted input |
  | `aug` | 686 | the augmentation itself is 4.2 ms (0.6%); the rest is noise |
  | `fix_aug` | 715 | +8% |

- **The T4 wall times are not comparable.** About 250–300 s per run in this stage, against about 85–110 s before,
  because 3 runs shared one T4 (`PREDECLARED_PLAN.md` §8).
- **Selection and stability.**
  - Selected epochs are similar: 110–133 on average.
  - `fix` has a larger spread over seeds (mean SD 5.5), mostly from one run: person 3, seed 1 scored 71.61 at
    selected epoch 88, against 94.32 and 94.64 for seeds 0 and 2.
  - `aug` has a similar outlier: person 3, seed 1 scored 70.98 at selected epoch 33.
  - `fix_aug` is the most stable arm (SD 2.17, against 2.27 for the control).
- **Source-only selection is noisy.** For `fix`, the last-epoch accuracy (91.4) is higher than at the selected epoch
  (87.6). The protocol was not changed after seeing this.
- **Leaky diagnostic.** The target-peeking protocol inflates every arm, for example 96.2 for `fix_aug` and 87.1 for
  `ie_off`. These numbers are never used for conclusions. They show that the paper-style 88.17 for GILE was reached in
  our runs only with target peeking, while the corrected model exceeds it **without** peeking (91.6).

## 8. Limitations
1. **One dataset, 5 persons, 3 seeds.** The target-level intervals use n = 5. The layout result is very consistent
   (5/5, Holm p = 0.007), but its exact size has wide uncertainty. Opportunity and UniMiB-SHAR were not run. The layout
   bug exists in their encoders too, but its effect there is untested.
2. **The control is reused from stage 1.** This is valid because the T4 anchor reproduced it bit for bit and the
   initial weights are identical (V6). Still, the new arms were run 3-at-a-time, which does not change the arithmetic of
   a deterministic run.
3. **Augmentation keeps the person label** of the transformed window. This asks z_d to recognise the person despite the
   style change [design choice; not compared with the alternative].
4. **The augmentation parameters were fixed a priori** and not tuned. A better setting may exist, but finding it
   without target data would need a source-only validation scheme.
5. **The mechanism is a hypothesis.** We attribute `fix`'s gain to the convolutions finally seeing per-sensor temporal
   patterns. We have not, for example, measured which filters change.
6. **Why the published GILE still reached 88%:** with target peeking, the broken model reaches about 87 in our runs too.
   The bug was masked by the leaky protocol (§1 row 9).
7. **Candidate A** (direct contrastive loss on z) was not tested.

## 9. Conclusion
- **The strongest improvement is a correction to the published model, not a new loss.**
  - Fixing GILE's encoder input layout, a one-line bug that makes every convolution mix nine different sensors instead
    of filtering each sensor over time, raised unseen-person accuracy by **+8.5 points** (95% CI +5.0 to +11.9; all 5
    persons; Holm p = 0.007).
  - Adding physically constrained person-style augmentation raised it by **+12.4** in total (CI +6.1 to +18.8; all 5
    persons), from 79.2% to 91.6%.
  - Both arms meet the predeclared STRONG criterion. Augmentation alone meets the weaker SUPPORTED criterion only.
- **Contrast with the earlier work.** Our two earlier attempts reshaped the latent code with extra losses:
  - repaired IE: −1.0 against the control;
  - XPC-B: +1.4 against its balanced-batch control.
  Neither improved accuracy beyond noise. The corrected input representation did, and with augmentation it also reduced
  person information in the activity code (MLP probe −6.6, 5/5 persons), more than IE did.
- **Main lesson.** In this reproduction, the largest gains came from auditing the data path and the evaluation protocol:
  - the layout bug;
  - the target-peeking selection, which inflates results by about 10 points.
  New regularisers did not deliver comparable gains.
- **What would strengthen the claim:**
  - seeds 3–4 (blocked only by the GPU quota);
  - the same correction on Opportunity and UniMiB-SHAR;
  - testing candidate A on top of `fix_aug`, now that the encoder sees the signal correctly.

## 10. Files
| Item | Path |
|---|---|
| Plan (written before the results) and deviations log | `experiment/PREDECLARED_PLAN.md` |
| Code | `experiment/code/stage3/` (`run_stage3.py`, `p3_variants.py`, `checks/test_stage3.py`) |
| Runs: config, history, log, metrics, `selected_model.pt` | `experiment/code/stage3/runs_p3/ucihar/<arm>/target<t>/seed<s>/` |
| Returned zips, per-seed manifests, partial seed 3 log | `experiment/returned_results/`, `experiment/code/stage3/manifests/` |
| Analysis | `experiment/analysis_p3/` (`summary.txt`, `arm_summary.csv`, `per_target.csv`, `decision.json`, `verify_runs_output.txt`) |
| Notebook (as run) | `experiment/colab_notebook.ipynb` |
| Verified local backup (284 files incl. 46 checkpoints, internal SHA-256 manifest) | `BACKUPS/P3_STAGE3_final_backup_20261116_2106.zip` (+ `.sha256`) |
| Number check: report vs analysis outputs (100/100 values match) | `experiment/tools/check_report_numbers.py` |
| GitHub-ready package (code, per-run results without checkpoints, docs, README with reproduction commands) | `improvement/` (this folder) |

## References (marked as in `LITERATURE_REVIEW_AND_PROPOSALS.md`)
- [GILE] Qian, H., Pan, S. J., Miao, C. Latent Independent Excitation for Generalizable Sensor-based Cross-Person
  Activity Recognition. AAAI 2021. [paper in `Project Context/`]
- [P2] Ilse, M., Tomczak, J. M., Louizos, C., Welling, M. DIVA: Domain Invariant Variational Autoencoders. MIDL,
  PMLR 121:322–348, 2020. [read: abstract]
- [P9] Khosla, P. et al. Supervised Contrastive Learning. NeurIPS 2020. arXiv:2004.11362. [read: abstract]
- [P13] Qin, X. et al. Generalizable Low-Resource Activity Recognition with Diverse and Discriminative Representation
  Learning (DDLearn). KDD 2023. doi:10.1145/3580305.3599360. [read: full]
- [P14] Lu, W., Wang, J., Sun, X., Chen, Y., Xie, X. Out-of-Distribution Representation Learning for Time Series
  Classification (DIVERSIFY). ICLR 2023. arXiv:2209.07027. [read: partial]
- [P15] Um, T. T. et al. Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using
  Convolutional Neural Networks. ICMI 2017, pp. 216–220. doi:10.1145/3136755.3136817. [read: abstract]
