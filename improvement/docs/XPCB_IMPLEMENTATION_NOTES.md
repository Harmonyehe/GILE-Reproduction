# XPC-B implementation notes (stage 2, Person 3)

Date: 16 Nov 2026. Design and rationale: `LITERATURE_REVIEW_AND_PROPOSALS.md` §4–§6.

**Isolation.** All stage-2 code and outputs live in `improvement-code/stage2_xpc/`.
- No stage-1 file was edited: `run_experiment.py`, `model_GILE_ucihar.py`, `train.py`, the data files, `runs_full/`, `runs_pilot_colab/` and the stage-1 notebooks are all untouched.
- The shared stage-1 modules (model, data loader) are **imported read-only**.
- The team repository was not touched.

## 1. Files
| File | Role |
|---|---|
| `stage2_xpc/run_stage2.py` | Stage-2 runner. Started as a byte copy of `run_experiment.py`, then edited for stage 2. Variants: `ie_off` (reproducibility anchor only), `ie_off_bal` (control), `xpc_bal` (XPC-B). Also has `--verify_batches_only` and `--calibrate_only` |
| `stage2_xpc/xpc.py` | `BalancedBatches` sampler, `make_proj_head`, `xpc_masks`, `xpc_loss` |
| `stage2_xpc/lambda.json` | λ fixed in advance by the 10% rule (see §4) |
| `stage2_xpc/checks/test_stage2.py` | 18 tests (17 on CPU; plus 1 CUDA test that runs only on a GPU) |
| `stage2_xpc/checks/test_stage2_local_output.txt` | Local CPU test output |
| `stage2_xpc/runs_xpc/ucihar/batch_verification/target*/seed0/batch_verification.json` | Real-data sampler checks, all 5 targets |
| `stage2_xpc/runs_xpc/ucihar/calibration/` | λ calibration log |
| `stage2_xpc/colab/build_stage2_notebook.py` → `GILE_XPCB_stage2.ipynb` | Self-contained Colab notebook (code embedded with SHA-256 checks; no Drive) |
| `stage2_xpc/analyse_stage2.py` | Manifest, summaries, paired target-level CIs, predeclared decision → `stage2_xpc/analysis/` |
| `stage2_xpc/runs_xpc/ucihar/<variant>/target<t>/seed<k>/` | Per-run config.json, history.csv, log.txt, metrics.json, selected_model.pt |

## 2. Person-balanced sampler (`BalancedBatches`)
**The stage-1 defect** [CODE]: `train.py:12` and `run_experiment.py:228` loop over the source people one after another. Every minibatch therefore contains a single person, with about 13–14 steps per epoch in 4 single-person blocks.

**How the stage-2 sampler builds a batch:**
- Every step concatenates **16 windows from each of the 4 source people**, making a batch of 64.
- For each person it uses the **same** sampler as stage 1: `WeightedRandomSampler` with replacement, weights 100/class count (class-balanced), and generator seed `seed*100 + k`.
- It draws 16 windows at a time, and restarts a person's stream when it is exhausted.
- **Steps per epoch** = Σ_p ⌊n_p/64⌋, exactly the stage-1 number. This is asserted in every run.

**Real-data verification** (`--verify_batches_only`, all 5 targets, seed 0):
- Every batch contains all 4 source people, 16 each.
- The **target person appears in 0 training windows**.
- 208 or 224 windows per person per epoch (13 or 14 steps).
- Class shares are 14.1–19.0%, the same as the stage-1 sampler (14.3–18.9%).
- 63.8–64.0 of 64 anchors per batch have a cross-person positive, with about 8.0 positives per window.

**The five people and five folds** [CODE + verified]:
- Persons 0–4 = UCI HAR subjects 1–5 (train and test files combined, `data_preprocess_ucihar.py`), with 347 / 302 / 341 / 317 / 302 windows.
- Target t is held out, and the other 4 are the sources.
- Each source person is split 80/20, with the last 20% of each activity going to validation. The 6 overlapping windows are purged, and zero overlap is asserted.

## 3. Cross-person supervised contrastive loss (`xpc_loss`)
Notation:
- μ_i = posterior mean of `qzy(x_i)`, the activity code (50-dimensional).
- g = Linear(50, 64) → ReLU → Linear(64, 32), used only in training.
- u_i = g(μ_i)/‖g(μ_i)‖.

```
P(i) = { j ≠ i : y_j = y_i, d_j ≠ d_i }            positives: same activity, different person
A(i) = { j : not (y_j = y_i and d_j = d_i) }        denominator: excludes self and same-person-same-activity
L_XPC = mean_{i : |P(i)| ≥ 1}  −1/|P(i)| Σ_{p∈P(i)} log( exp(u_i·u_p/τ) / Σ_{a∈A(i)} exp(u_i·u_a/τ) )
Total (xpc_bal) = loss_function(d, x, y)  [released GILE main loss, unchanged]  + λ · L_XPC
```

**Design choices:**
- **Why exclude same-person, same-activity windows from the denominator:** UCI windows overlap by 50%, so such windows are often near-duplicates. Treating them as negatives would push near-identical inputs apart. Treating them as positives would reward *within-person* similarity, which is not the goal.
- **Batches with no valid positives:** anchors without a positive are skipped. If no anchor has one, the loss returns an exact 0 that is still differentiable (tested).
- **Why the posterior mean:** it is the same quantity the probes and the classifier use at test time. `qzy`'s mean is deterministic, so recomputing it adds no randomness.
- **Optimiser and scope:** a single Adam optimiser at lr 1.1e-3 over the model plus g (Adam is per-parameter, so g does not change the model updates). The gradient of L_XPC reaches **only `qzy` and g** (tested).
- **Random-number isolation:** g is built with **all** random states (CPU and every CUDA device) saved and restored. So `ie_off_bal` and `xpc_bal` get identical model initialisation, sampling noise and batches, and differ only by λ·L_XPC.

**Correction made during execution:**
- The first Colab test run failed the "stage-2 `ie_off` reproduces stage-1 exactly" check on GPU, although it passed on CPU.
- **Cause:** `torch.manual_seed` also reseeds CUDA, and the original `fork_rng(devices=[])` protected only the CPU state.
- **Fix:** fork the CUDA states as well, and add a CUDA test.
- The faulty notebook run was interrupted during its first anchor run, before any run completed. Its runtime was deleted and the corrected notebook was uploaded fresh.
- Superseded notebook copy in the Colab account: (private Colab copy). Used notebook: (private Colab copy).

## 4. Hyperparameters (all fixed before any stage-2 result was seen)
- τ = 0.1.
- **λ = 6012.38** (`lambda.json`), computed as 0.1 × (mean main loss per batch) / (mean L_XPC per batch). This was measured at initialisation on CPU, on the **source** data of target 0 with seed 0: one epoch of balanced batches, no parameter updates, no target data. The two means were 246,202 and 4.095.
- λ is used unchanged for all targets and seeds.
- **Expected consequence (noted before the runs):** the main GILE loss falls about 20-fold over 150 epochs (stage 1: ~4,250 → ~150 per window), while L_XPC falls much less. So XPC's share of the loss rises from 10% to an estimated 30–50% late in training. The share is logged every epoch. The rule was not changed, because changing it now would be tuning.
- Everything else is identical to stage 1: 150 epochs, batch 64, lr 1.1e-3, β_d = β_y = 1, auxiliary multipliers 1000, the data split and purge, epoch selection by mean source-validation accuracy (ties broken by CE), and the probe protocol.

## 5. Tests (`checks/test_stage2.py`, ALL PASSED locally on CPU and on the Colab T4)
1–3. The positive mask and the denominator mask match their definitions element by element.
4. `xpc_loss` equals a naive per-anchor loop (difference < 1e-4).
5. A single-person batch gives no anchors, a loss of exactly 0, and backward still works.
6. The XPC gradient reaches only `qzy` (and g).
7. Building the head leaves the global CPU random stream unchanged. (7b, GPU only: the CUDA stream is also unchanged.)
8–11. Balanced batches: the step count equals stage 1; every batch has 16 windows from each person; a second epoch works; the batches are seed-reproducible.
12. A real-sized balanced batch gives positives to 64 of 64 anchors.
13. Mechanism check: the person prior p(z_d|d) becomes person-specific in mixed batches.
14. The stage-2 runner's `ie_off` reproduces the stage-1 `run_experiment.py` `ie_off` **exactly** (fake data).
15. `xpc_bal` with λ = 0 is identical to `ie_off_bal`.
16. `xpc_bal` runs with a finite loss and an active XPC term, and the probe leak guard reports 0 shared and 0 target windows.
17. `ie_off_bal` differs from the person-sequential run, confirming the batching really changed.

On Colab, a **reproducibility anchor** re-runs stage-1 `ie_off` (target 0, seed 0) through the stage-2 runner and compares it with the stored `runs_full` result. See `stage2_xpc/analysis/anchor_check.json`.

## 6. Leakage safeguards (no target data used)
- The target person is excluded from training batches (asserted on real data) and from the λ calibration.
- Epoch selection uses source-person validation only.
- Probes are fit on source-train codes and scored on source-validation codes, with a per-run assertion: 0 shared windows and 0 target windows.
- Target metrics are computed each epoch **for logging only**. The "leaky" best-on-target epoch is stored as a labelled diagnostic and is never used.
