# Stage 3: predeclared experimental plan (written BEFORE any stage-3 result exists)

Written 16 Nov 2026, about 18:20 (laptop clock). No stage-3 run had been started when this file was written. This plan, the
success rule and the analysis are fixed here and will not be changed after results are seen. Any later deviation is
recorded in section 8 with its reason.

## 1. Question
Can a **non-loss** change to GILE raise activity-recognition accuracy on an **unseen person** (UCI HAR,
leave-one-person-out over the 5 persons used by the team)? Two candidates are tested, alone and together.

- **C: input-layout correction (`fix`).**
  - All three released encoders (`qzd`, `qzx`, `qzy` in `model_GILE_ucihar.py`) do
    `x.view(-1, x.shape[2], 1, x.shape[1])` on a contiguous `(B, 128 time, 9 sensor, 1)` tensor. This is a
    *reinterpretation* of memory, not a transpose.
  - The resulting "channel" c holds flat elements c·128 … c·128+127, that is about 14 consecutive time steps × all
    9 sensors interleaved.
  - Along the convolution axis, neighbouring positions are **different sensors at (nearly) the same time**. The
    (1×5) kernel therefore convolves across sensors, not over time.
  - The decoder `px` produces `(B, 9, 1, 128)` and permutes it back with `.permute(0, 3, 1, 2)`. So the authors'
    intended layout was channel = sensor and width = time.
  - The correction is `x.permute(0, 2, 3, 1)`: `(B,128,9,1) → (B,9,1,128)`. Parameter shapes, initialisation and
    every other line stay unchanged.
- **B: person-style sensor augmentation (`aug`)**, training windows only, fixed a priori (no tuning).
  - Each window is transformed with probability 0.5. The transform is applied in raw physical units (the released
    loader does not normalise).
  - **Movement-intensity scaling.** s ~ N(1, 0.1²), clipped to [0.7, 1.3]:
    - multiplies `body_acc` and `body_gyro`;
    - `total_acc` gets the same change, `total_acc + (s−1)·body_acc`, so gravity is untouched.
    - Rationale: people move with different vigour (Um et al. 2017 scaling σ = 0.1).
  - **Small device-orientation change.** One rotation R about a uniformly random axis, angle ~ U(−15°, +15°):
    - R is applied identically to the `body_acc`, `body_gyro` and `total_acc` triads, since all are in the same
      phone frame;
    - it preserves vector norms and the gravity magnitude;
    - rationale: the waist-worn phone sits slightly differently on each person (Um et al. 2017 rotation;
      Mathur et al. 2018).
  - **Small sensor noise.** Gaussian with σ = 0.01 (g for acc, rad/s for gyro); the same noise is added to `body_acc`
    and `total_acc`.
  - **Not used:**
    - time-warping or window slicing: they can change cadence, which separates walking from walking-up/down stairs;
    - large rotations: they destroy the gravity-orientation cue that separates sitting, standing and lying;
    - channel permutation: it destroys sensor meaning.
  - The augmentation uses its **own RNG stream** (`torch.Generator`, seed·7919 + 17). It draws the same number of
    random numbers whether or not a window is transformed. So model initialisation, batch order and the VAE noise are
    identical to the control.
- **A: direct cross-person contrastive loss on z.** Ranked third and **not run** in this stage (reasons in the final
  report: the 50-run XPC-B evidence, the balanced-batching confound, and GPU time on the deadline day).

## 2. Arms (2 × 2 factorial; everything else identical to the stage-1 control)
| Arm | Layout | Augmentation | Source of results |
|---|---|---|---|
| `ie_off` (control) | released `.view()` | none | **existing** 25 stage-1 runs (`runs_full/ucihar/ie_off`, copied read-only to `baseline_ie_off_metrics/`). They were reproduced bit-exactly on the T4 in stage 2; re-checked again by a new anchor run (section 5). |
| `fix` | corrected permute | none | new |
| `aug` | released `.view()` | B | new |
| `fix_aug` | corrected permute | B | new |

**Shared settings:**
- the stage-1 control's training step: person-sequential batches, class-balanced `WeightedRandomSampler`
  (seed·100+k), single Adam at lr 1.1e-3, main GILE loss only (IE off), 150 epochs, batch 64,
  β_d = β_y = 1, aux weights 1000;
- the same source train/val split (per activity, contiguous tail of 20%, overlap-purged);
- targets 0–4; seeds 0–4.

## 3. Protocol (no target information is used for training or selection)
- Epoch selection: maximum mean **source-validation** accuracy (tie → lower source-val CE_y). This is the
  **primary** metric.
- Also reported: last-epoch target scores, and the leaky best-target epoch (labelled diagnostic, never used for any
  conclusion).
- Probes (secondary): logistic regression and MLP(64) predicting person from the posterior mean of z, and activity
  from z_d. They are fit on source-train and scored on source-val, with the leak guard.
- **No hyper-parameter of B or C is tuned on any data.** All values above are fixed in this file.

## 4. Primary analysis and success rule (fixed now)
**Unit of analysis.** For each arm, per target: the mean over seeds; differences are paired by (target, seed) against
`ie_off`. Inference is at target level: n = 5 paired target means, t-interval with df = 4.

Primary outcome: **target accuracy at the source-val-selected epoch.** For each new arm vs `ie_off`:

- **IMPROVEMENT SUPPORTED**, if all of:
  1. mean Δacc > 0;
  2. accuracy is higher for at least 4 of 5 target persons;
  3. mean Δ macro-F1 > 0.
- **STRONG**, additionally: the 95% target-level CI of Δacc excludes 0.
- **HARMFUL**: mean Δacc < 0 **and** accuracy lower for at least 4 of 5 targets.
- **INCONCLUSIVE**: anything else.
- **Multiplicity.** Three arms are compared with one control. A STRONG claim also requires the Holm-adjusted
  (3 comparisons) paired t-test p < 0.05; unadjusted CIs are reported as well.

**Factorial effects** (descriptive, with the same t-intervals):
- layout main effect = mean of (`fix` − `ie_off`, `fix_aug` − `aug`);
- augmentation main effect = mean of (`aug` − `ie_off`, `fix_aug` − `fix`);
- interaction = (`fix_aug` − `aug`) − (`fix` − `ie_off`).

**Secondary** (reported, never used to redefine success):
- macro-F1 and weighted-F1;
- per-person tables;
- the person ← z linear and MLP probes;
- training stability: non-finite losses, failed runs, standard deviation over seeds, selected epoch;
- runtime: seconds per run and per epoch.

A lower probe value is **not** treated as an accuracy improvement.

## 5. Execution and integrity
- **GPU:** Colab T4 (the `a Colab account (account 1)` account used in stage 2), with the same transfer workflow.
  - One embedded notebook; UCI HAR is downloaded from the UCI archive with its SHA-256 checked.
  - Tests run before any experiment.
  - **Reproducibility anchor:** `ie_off` target 0 seed 0 must reproduce the stage-1 value exactly. If it does
    not, the environment changed: this is recorded, and the control is **re-run** in the new environment rather
    than mixing environments.
- **Order:** seed by seed, all 15 runs of a seed (3 arms × 5 targets) before the next seed.
  - Each seed is zipped with a SHA-256 manifest and downloaded before the next seed starts.
  - Each seed's zip is verified locally.
- **Minimum design** if GPU time or quota runs out: seeds 0–2 (45 runs). The analysis then uses the seeds completed
  for **all** arms and targets; partial seeds are reported but excluded from the primary test. The number of seeds is
  decided by time and quota only, never by results. Analysis starts after the last seed cell.
- No result is discarded. Failed runs are kept with their logs and reported.

## 6. Isolation
- All stage-3 work is in `improvement-code/experiment/`.
- Shared modules are **copies** (byte-identical to stage 1). The layout fix is applied by replacing the encoders'
  `forward` on the model *instance* at run time; `model_GILE_ucihar.py` is not edited.
- Nothing in `runs_full/`, `stage2_xpc/`, `XPCB_TRANSFER_PACKAGE/`, `original-code/` or `evaluation-code/` is
  modified.

## 7. References for the design (verified earlier in `LITERATURE_REVIEW_AND_PROPOSALS.md` unless marked)
- Um, T. T. et al. (2017). Data augmentation of wearable sensor data for Parkinson's disease monitoring using
  convolutional neural networks. *ICMI 2017*. Source of the scaling, rotation and jitter augmentations.
- Qian, H., Pan, S. J., Miao, C. (2021). Latent independent excitation for generalizable sensor-based cross-person
  activity recognition. *AAAI 2021*. GILE; the decoder layout that shows the intended encoder input.
- Further HAR augmentation and generalisation references are given in the final report, each marked as verified or
  not.

## 8. Deviations log (filled in during execution)
- **Execution: 3 runs in parallel on the one T4.** Each run is still deterministic on its own: the anchor reproduced
  exactly, and the test suite passed on the T4. Consequence: per-run wall times (~250–300 s) are **not** comparable
  with the ~85–110 s of the earlier, sequential runs. Computational cost is therefore measured separately, on the same
  hardware, per training step (final report §7).
- **GPU quota.** About 20:00 (laptop clock), during seed 3, the Colab runtime was disconnected. Reconnecting gave
  "Cannot connect to GPU backend" (free-tier usage limit). No purchase was made.
  - Seeds 0–2 were complete and verified locally. They form the predeclared **minimum design** (45 new runs + the
    matching 15 control runs).
  - Seed 3: 10 of 15 runs had printed their final metrics. These are recovered verbatim in
    `returned_results/seed3_PARTIAL_printed_log_recovered.txt` and reported descriptively only.
  - Seed 4 was never run.
  - This decision follows section 5 (time and quota only, not results). No stage-3 aggregate result had been computed
    when the quota ended; only the individual runs' printed lines were visible in the notebook.
- **Second seed-3 attempt** (16 Nov ~20:50, account a second Colab account, same notebook).
  - Tests passed and the anchor matched exactly, so seed 3 was restarted.
  - About 20 minutes later, the author instructed that no further seeds be run. The cell was interrupted and the
    training processes stopped (GPU idle).
  - No output of this attempt was downloaded, retrieved or used.
  - **Final status:** main analysis = seeds 0–2; seed 3 = partial (10 recovered printed results, descriptive only);
    seed 4 = not run.
- **Citation.** Mathur et al. (2018), cited in section 1 as a motivation for rotation, was **not verified** in this
  project. It is not relied on in the final report; Um et al. (2017) is the verified source.
