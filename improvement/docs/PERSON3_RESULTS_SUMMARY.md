# Person 3: results summary (all improvements tested)

**Setup for every row:**
- UCI HAR, unseen-person (leave-one-person-out) accuracy;
- persons 0–4;
- epoch selected on source-person validation only;
- Colab T4;
- Δ is paired against the stated control with a target-level 95% CI (n = 5 persons).

Full report: `PERSON3_IMPROVEMENT_FINAL_REPORT.md`.

| Stage | Improvement | Runs | Control | ΔAccuracy [95% CI] | Persons up | Person ← z probe Δ (linear / MLP) | Predeclared criterion | Outcome |
|---|---|---|---|---|---|---|---|---|
| 1 | Repaired Independent Excitation (`ie_on`) | 75 (5 seeds) | `ie_off` | −0.99 [−6.28, +4.29] | 2/5 (weighted-F1) | +1.07 / **−3.76** (5/5 lower) | separation and accuracy | separation partly, accuracy **no** |
| 2 | XPC-B: cross-person contrastive loss + balanced batches (`xpc_bal`) | 50 + anchor (5 seeds) | `ie_off_bal` | +1.39 [−2.24, +5.01] | 3/5 | −0.33 / +0.19 | at least 4/5 up AND both probes down | **INCONCLUSIVE** |
| 2 | Person-balanced batches alone (`ie_off_bal`) | (as above) | `ie_off` | −1.81 [−8.48, +4.85] | 2/5 | — | — | not helpful |
| 3 | **Input-layout correction (`fix`)** | 15 (seeds 0–2) | `ie_off` | **+8.46 [+4.99, +11.93]** | **5/5** | +0.65 / −2.81 | STRONG = SUPPORTED + CI > 0 + Holm p < 0.05 | **STRONG IMPROVEMENT** (Holm p 0.007) |
| 3 | Person-style augmentation (`aug`) | 15 (seeds 0–2) | `ie_off` | +3.99 [−1.50, +9.48] | 4/5 | −0.78 / −3.85 | SUPPORTED = mean > 0, at least 4/5 up, macro-F1 > 0 | **IMPROVEMENT SUPPORTED** (not strong) |
| 3 | **Layout correction + augmentation (`fix_aug`)** | 15 (seeds 0–2) | `ie_off` | **+12.44 [+6.08, +18.80]** | **5/5** | **−5.55 / −6.60** (MLP lower for 5/5) | as above | **STRONG IMPROVEMENT** (Holm p 0.011) |

**Absolute accuracy, seeds 0–2** (macro-F1 / weighted-F1 in the report):
| Arm | Accuracy |
|---|---|
| `released` | 78.75 |
| `ie_off` | 79.17 |
| `ie_on` | 76.58 |
| `xpc_bal` | 77.93 |
| `aug` | 83.16 |
| `fix` | 87.63 |
| **`fix_aug`** | **91.61** |

**Which method performed best?**
- **`fix_aug`**: 91.61% accuracy and 91.30 macro-F1, against 79.17 / 77.86 for the control.
- It has the lowest seed-to-seed spread (SD 2.17).
- It gives the largest reduction of both person probes in any stage: linear −5.6, MLP −6.6.

**Did accuracy genuinely improve?**
- **Yes, for the layout correction, alone and with augmentation:**
  - higher for every one of the 5 unseen persons;
  - the intervals exclude 0;
  - the result survives a Holm correction for 3 comparisons.
- Augmentation alone is positive (4/5 persons) but not conclusive on its own.
- The factorial shows the two effects are additive: layout +8.5, augmentation +4.0, interaction ≈ 0.

**Did person information decrease?**
- Only with `fix_aug`: MLP probe −6.6 [−10.3, −2.9], lower for 5/5 persons; linear probe −5.6 (CI includes 0).
- It is still well above chance (47% against 27%), so z is not person-invariant.
- A probe reduction is never counted as an accuracy gain.

**Did each result meet its predefined criteria?**
- Stage 1 IE: no accuracy gain.
- Stage 2 XPC-B: inconclusive.
- Stage 3 `fix` and `fix_aug`: **STRONG**.
- Stage 3 `aug`: **SUPPORTED**.

**Caveats:**
- 3 seeds instead of the planned 5: the GPU quota ended. This was the predeclared minimum.
- One dataset.
- See the report's §8.
