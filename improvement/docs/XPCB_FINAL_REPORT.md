# XPC-B final report: person-balanced batches + cross-person supervised contrastive learning (stage 2)

Date: 16 Nov 2026 (final, full 50-run design). The previous version, which relied on recovered log lines and had no probes, is preserved at `XPCB_TRANSFER_PACKAGE/snapshot/reports/XPCB_FINAL_REPORT.md`.
- Implementation: `XPCB_IMPLEMENTATION_NOTES.md`.
- Design and literature: `LITERATURE_REVIEW_AND_PROPOSALS.md`.
- All numbers come from `XPCB_TRANSFER_PACKAGE/analysis_t4/` (`python tools/analyse_results.py code/stage2_xpc/runs_xpc_t4`). The run files are in `XPCB_TRANSFER_PACKAGE/code/stage2_xpc/runs_xpc_t4/`.

## 0. Bottom line
- **Verdict (predeclared rule, full 50-run design): INCONCLUSIVE.** XPC-B improved neither objective reliably.
  - **Recognition:** `xpc_bal − ie_off_bal` accuracy **+1.39** (95% CI −2.24 to +5.01). It was higher for only **3 of 5** held-out people (the rule required at least 4). Macro-F1 +1.64, weighted-F1 +1.61, each 3 of 5.
  - **Separation:** person ← z probes did **not** fall. Linear −0.33 [−2.87, +2.21] and MLP +0.19 [−2.38, +2.77], each lower for only 3 of 5 people.
- **Balanced batches alone did not help:** `ie_off_bal − ie_off` accuracy −1.81 [−8.48, +4.85], person 1 −10.1. The full XPC-B package is not better than the existing baselines: vs `ie_off` −0.43, vs `released` −0.23 accuracy.
- **Repaired IE remains the only variant that reduced nonlinearly decodable person information.** Its MLP person probe is 4.49 points *lower* than `xpc_bal` (CI 2.65 to 6.32, all 5 people). However, IE did not improve recognition either.
- **Completeness and integrity:**
  - **All 50 runs plus the reproducibility anchor completed** on a Colab T4 (torch 2.14.1+cu130), the same environment as the 75-run baseline.
  - All five target people and all five seeds are covered.
  - All files were downloaded and verified locally: 255 SHA-256 checks, the config and selection-rule checks, and the probe leak guards.
  - The 46 runs that also existed in the interrupted first T4 session reproduced their recorded results **exactly**, which shows the experiment is deterministic and reproducible.

## 1. What was changed and why
**What stage 1 showed (75 runs):**
- The repaired IE did not improve unseen-person recognition.
- Person information stayed decodable from the activity code z (linear ≈ 52%, MLP ≈ 61–65%, chance 27%).

**What the literature review found:**
- **(a)** GILE trains on **single-person minibatches**. This collapses the person prior and rules out any cross-person loss.
- **(b)** Class-conditional, cross-person alignment (supervised contrastive learning with positives = same activity, different person) was the best-justified way to remove *within-activity* person variation.

**What stage 2 changed:**
1. **Person-balanced batches.** Each step takes 16 windows from each of the 4 source people. The number of steps per epoch and the class-balanced sampler are unchanged.
2. **The XPC loss** on the posterior mean of z, through a training-only projection head g (50 → 64 → 32).
   - Positives: same activity, different person.
   - Same-person, same-activity windows are excluded from the denominator.
   - τ = 0.1. λ = 6012, fixed in advance by a 10%-of-initial-loss rule on source data only.

**Arms:** `ie_off_bal` (balanced batches only, the control) and `xpc_bal` (balanced batches + XPC). Everything else matches stage 1.

## 2. Protocol
- **Data and folds:** UCI HAR persons 0–4 (subjects 1–5), leave-one-person-out over 5 targets.
- **Source split:** 80/20 per activity (the last 20% goes to validation), overlap-purged, with zero leakage asserted.
- **Training:** 150 epochs, batch 64, one Adam optimiser at lr 1.1e-3.
- **Model selection:** epoch chosen by mean **source-validation** accuracy (ties broken by CE). Target data is never used for training, λ, selection or probes.
- **Probes:** logistic regression and an MLP(64), fit on source-train posterior means and scored on source-validation, with a leak guard asserted in every run.
- **Seeds:** 0–4. **Runs:** 5 targets × 5 seeds × 2 arms = **50**, plus the anchor.
- **Statistics:** paired differences per (target, seed), averaged within each target, then a 95% t-interval over the 5 targets (df = 4).
  - With 5 people the power is low. Stage 1 showed about 4.8 weighted-F1 points of per-run noise even between mathematically identical procedures.
  - **No significance is claimed.**
- **Predeclared rule:**
  - IMPROVEMENT only if accuracy improves for at least 4 of 5 targets **and** both person ← z probes fall for at least 4 of 5.
  - FAILURE if the probes fall but recognition gets worse.
  - HARMFUL if accuracy is worse for at least 4 of 5.
  - Otherwise INCONCLUSIVE.
  - The full 50-run design is primary because it completed (the planned preference). Seeds 0–2, the pre-planned reduced design, are shown alongside.
- **Reproducibility anchor:** stage-1 `ie_off` (target 0, seed 0) re-run through the stage-2 runner on the T4 matched stage 1 **exactly**: epoch 99, accuracy 74.0634, weighted-F1 73.9257.

## 3. Results: unseen-person recognition (T4, epoch selected on source validation)
### 3.1 Full design: mean over 5 targets within each seed, then mean ± sd over 5 seeds
| Variant | Accuracy | Macro-F1 | Weighted-F1 | Selected epoch |
|---|---|---|---|---|
| released (stage 1) | 78.15 ± 1.28 | 76.62 ± 1.37 | 76.77 ± 1.30 | 121 |
| ie_off (stage 1) | 78.35 ± 1.65 | 76.93 ± 1.77 | 77.02 ± 1.87 | 122 |
| ie_on, repaired IE (stage 1) | 77.35 ± 1.19 | 75.98 ± 1.17 | 75.80 ± 1.41 | 125 |
| **ie_off_bal** | 76.53 ± 1.37 | 75.11 ± 1.39 | 74.86 ± 1.41 | 112 |
| **xpc_bal** | 77.92 ± 1.96 | 76.75 ± 2.03 | 76.47 ± 2.29 | 109 |

### 3.2 Paired comparisons (full design: 25 pairs, 5 targets)
| Comparison | Accuracy | Weighted-F1 | Accuracy, targets up / down | Per-target accuracy |
|---|---|---|---|---|
| **xpc_bal − ie_off_bal** (primary) | **+1.39 [−2.24, +5.01]** | +1.61 [−2.89, +6.11] | **3 / 2** | 0: −1.33, 1: +2.12, 2: −0.70, 3: +0.82, 4: +6.03 |
| ie_off_bal − ie_off (batching effect) | −1.81 [−8.48, +4.85] | −2.16 [−9.38, +5.06] | 2 / 3 | 0: −3.40, 1: −10.13, 2: −0.65, 3: +1.14, 4: +3.97 |
| xpc_bal − ie_off | −0.43 [−9.01, +8.16] | −0.55 [−10.10, +9.00] | 2 / 3 | 0: −4.73, 1: −8.01, 2: −1.35, 3: +1.96, 4: +10.00 |
| xpc_bal − released | −0.23 [−9.59, +9.13] | −0.30 [−10.51, +9.91] | 2 / 3 | 0: −5.13, 1: −9.14, 2: −1.58, 3: +5.43, 4: +9.27 |
| xpc_bal − ie_on | +0.57 [−5.33, +6.47] | +0.67 [−6.23, +7.57] | 2 / 3 | 0: −2.07, 1: −1.79, 2: −4.52, 3: +4.86, 4: +6.36 |

Macro-F1 follows accuracy throughout (`xpc_bal − ie_off_bal` +1.64 [−2.38, +5.67]). Full tables: `analysis_t4/paired_FULL_seeds01234.csv`.

### 3.3 Per person (mean over 5 seeds)
| Target | released | ie_off | ie_on | ie_off_bal | xpc_bal |
|---|---|---|---|---|---|
| Accuracy 0 | 77.98 | 77.58 | 74.93 | 74.18 | 72.85 |
| Accuracy 1 | 71.19 | 70.07 | 63.84 | 59.93 | 62.05 |
| Accuracy 2 | 92.20 | 91.96 | 95.13 | 91.32 | 90.62 |
| Accuracy 3 | 73.88 | 77.35 | 74.45 | 78.49 | 79.31 |
| Accuracy 4 | 75.50 | 74.77 | 78.41 | 78.74 | 84.77 |
| Weighted-F1 0 | 77.98 | 77.51 | 74.49 | 73.67 | 71.50 |
| Weighted-F1 1 | 66.76 | 66.23 | 59.55 | 54.88 | 57.35 |
| Weighted-F1 2 | 92.10 | 91.87 | 95.10 | 91.03 | 90.30 |
| Weighted-F1 3 | 72.34 | 75.51 | 71.97 | 77.46 | 78.67 |
| Weighted-F1 4 | 74.65 | 73.98 | 77.87 | 77.24 | 84.51 |

Per (target, seed) values: `analysis_t4/per_target_seed_FULL_seeds01234.csv`.

### 3.4 Pre-planned reduced design (seeds 0–2), shown alongside
- **xpc_bal − ie_off_bal:** accuracy +1.56 [−3.95, +7.07], 3 / 2 targets. This is identical to the earlier analysis of the recovered log.
- **Probes:** linear −0.74 (1 / 4 targets up / down), MLP +0.48 (2 / 3).
- **Verdict:** also INCONCLUSIVE.

## 4. Results: representation separation (probes, selected-epoch model, all 50 runs)
Lower person ← z means less person information in the activity code. Chance (person majority) is 27.1%.

| Variant | person ← z linear | person ← z MLP | activity ← z_d linear | activity ← z_d MLP |
|---|---|---|---|---|
| released | 52.35 ± 1.80 | 64.88 ± 2.36 | 69.91 ± 1.49 | 73.33 ± 2.06 |
| ie_off | 52.31 ± 3.08 | 64.83 ± 2.34 | 68.58 ± 2.56 | 72.20 ± 1.56 |
| ie_on | 53.38 ± 1.75 | **61.07 ± 1.49** | 66.79 ± 1.25 | 69.66 ± 1.21 |
| ie_off_bal | 54.89 ± 2.12 | 65.37 ± 1.66 | 70.24 ± 1.22 | 74.72 ± 1.91 |
| xpc_bal | 54.56 ± 1.05 | 65.56 ± 1.69 | 70.24 ± 2.87 | 74.55 ± 1.38 |

| Comparison | person ← z linear | person ← z MLP |
|---|---|---|
| **xpc_bal − ie_off_bal** | −0.33 [−2.87, +2.21]; down for 3 of 5 | +0.19 [−2.38, +2.77]; down for 3 of 5 |
| ie_off_bal − ie_off | +2.58 [−3.53, +8.69]; up for 4 of 5 | +0.53 [−2.97, +4.04] |
| xpc_bal − ie_on | +1.18 [−3.06, +5.42] | **+4.49 [+2.65, +6.32]; up for all 5** |

- **XPC did not reduce person information in z** under either probe.
- Balanced batching slightly *raised* linearly decodable person information (+2.6, 4 of 5 people).
- The repaired IE (stage 1) is still the only variant with a consistent reduction in the nonlinear probe.

## 5. Training behaviour and runtime
- **Stability:** 50/50 runs finished with finite losses and exit code 0. No failed runs, and no runs needed resuming.
- **XPC's share of the total loss:**
  - 9.5% at epoch 1, as designed (range 8.9–9.7%).
  - **77.8% at epoch 150** (range 54–92%; per-run maximum 80–92%).
  - Reason: the GILE main loss falls about 25-fold during training, while L_XPC falls only from 3.81 to 2.05.
  - So late in training the contrastive term dominated the objective. This follows from the predeclared λ rule; λ was not changed.
- **Selected epochs:** XPC runs selected earlier epochs (109 vs 112 for the control, vs 121–125 in stage 1).
- **Seed spread:** the seed-to-seed spread of accuracy was larger for `xpc_bal` (sd 1.96 vs 1.37).
- **Runtime:**
  - About 101 s per `ie_off_bal` run and 122 s per `xpc_bal` run on the T4. That includes the per-epoch resume checkpoints of the transfer runner.
  - About 1 h 50 min of GPU time in total.
- **Secondary and diagnostic numbers (not used for any decision):**
  - Last-epoch weighted-F1: `ie_off_bal` 76.42, `xpc_bal` 76.25.
  - Target-peeking ("leaky") weighted-F1: `ie_off_bal` 86.56, `xpc_bal` 85.67.
  - The ~10-point inflation from selecting on the target person is the same as in stage 1.

## 6. Interpretation (my interpretation, labelled as such)
1. **Why XPC did not remove person information from z (hypothesis, consistent with the measurements).**
   - The contrastive loss acts on the *projection-head output* g(μ), not on z itself.
   - A non-linear head can map person-specific variation in z onto the same point. That satisfies the loss without removing the variation from z.
   - This matches a known effect in contrastive learning: information discarded at the projection output often stays in the representation before the head (Chen et al., SimCLR, ICML 2020; not re-read for this report).
   - It also explains why L_XPC fell only to about 2.0 while the probes on z did not move.
   - **The logical next variant:** apply the loss to z directly (identity head). It was not run: no unplanned variants.
2. **Why accuracy moved a little (+1.4) but inconsistently.**
   - The largest gain was for person 4 (+6.0). Persons 0 and 2 got slightly worse.
   - With only 4 source people, aligning activity clusters across *those* people does not guarantee alignment for a fifth.
   - The large and growing weight of XPC (up to 92% of the loss) may also have pulled training away from the ELBO and the classifier terms, so runs selected earlier epochs.
3. **Why balanced batches did not help.**
   - Mixed-person batches are standard practice in domain generalisation, but here they lowered accuracy on average, most strongly for person 1 (−10).
   - They also slightly raised linear person information in z.
   - A plausible but untested explanation: once the person prior and person classifier see several people per batch, the ELBO organises z partly by person.
4. **Overall.** Across 125 GILE runs (75 + 50), none of the four modifications (repaired IE, balanced batching, XPC, and released vs ie_off) reliably improved recognition of unseen people under an honest, source-only protocol.
   - The one intervention that reduced person information (IE) did not translate that into accuracy.
   - This agrees with large domain-generalisation benchmarks (DomainBed), where invariance-based methods rarely beat plain training by more than about 1 point.

## 7. Limitations and threats to validity
1. **Only 5 people and one dataset**, so the intervals are wide and only effects of roughly 4 points or more are detectable.
2. **λ was fixed in advance by the 10% rule and not tuned.** Its effective weight grew to dominate late training. A different λ, or a schedule that keeps XPC's share constant, might behave differently. Testing that now would be post-hoc and is left as future work.
3. **XPC was applied through a projection head** (standard practice). The probes measure z, before the head.
4. **The batching change and XPC were tested as a package against stage-1 baselines.** The `ie_off_bal` control separates the two effects.
5. **Known stage-1 issues are kept in all arms:** channel mixing in `.view()`, and validation windows taken from the end of each activity.
6. **Environment:** all stage-2 numbers come from a different Colab account than stage 1. The anchor and the 46 exact reproductions show that the computational environment was identical.

## 8. Execution history (full transparency)
- **First T4 session (account 1):**
  - 45 of 50 runs plus the anchor completed.
  - Then Colab's GPU usage limit ended the session. The run folders were lost before download, and only the printed metrics were recovered.
- **CPU fallback:** a fallback on the laptop CPU was started for probes. It was too slow (about 2 min per epoch), had no checkpoints, and was superseded.
- **Transfer package:** built with a resumable runner, per-seed downloads and integrity tools (`XPCB_TRANSFER_PACKAGE/`).
  - A second account had no GPU available.
  - A third account (university Google Workspace) obtained a T4, and the full design ran there in one session.
- **Bugs found and fixed before any result was used:**
  1. Building the projection head reseeded the GPU's random stream (caught by the GPU test).
  2. Three bugs in the packaging and restore tooling (caught by the self-tests).
- **Not changed after seeing results:** no setting, rule or analysis choice.

## 9. Where everything is
| Item | Location |
|---|---|
| This report | `improvement-code/XPCB_FINAL_REPORT.md` |
| Implementation notes | `improvement-code/XPCB_IMPLEMENTATION_NOTES.md` |
| Final run files (50 + anchor: config, history, log, metrics, selected_model.pt) | `XPCB_TRANSFER_PACKAGE/code/stage2_xpc/runs_xpc_t4/ucihar/` |
| Per-seed SHA-256 manifests | `XPCB_TRANSFER_PACKAGE/code/stage2_xpc/manifests/` |
| Downloaded zips, as received | `XPCB_TRANSFER_PACKAGE/returned_results/` |
| Analysis (summaries, paired CIs, per-target, decision.json) | `XPCB_TRANSFER_PACKAGE/analysis_t4/` |
| Run status | `XPCB_TRANSFER_PACKAGE/returned_results/run_status_final.csv` (51/51 COMPLETE) |
| Colab notebook used (account a Colab account (account 1)) | (private Colab copy) |
| First-session evidence (recovered log) | `improvement-code/stage2_xpc/gpu_log_recovered/` |
| Unchanged | stage-1 code, `runs_full/` (75 runs), pilot results, `stage2_xpc/` originals, team repository |
