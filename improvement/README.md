# Improving GILE for unseen-person activity recognition (UCI HAR)

**Context.** COMP41850 group project: reproduction and improvement of GILE (Qian, Pan & Miao, *Latent Independent
Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*, AAAI 2021). This folder holds the
improvement work: proposing, implementing and evaluating changes to GILE.
- Full write-up: [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).
- The team's reproduction code is not modified by anything here.

## Main result
- GILE's encoders reshape their input with `x.view(-1, 9, 1, 128)`. This **scrambles sensors and time**, so every
  convolution window mixes 5 different sensors instead of following one sensor over time.
- Correcting this layout (`fix`), and adding physically constrained person-style augmentation (`aug`), raised
  unseen-person accuracy from **79.2% to 91.6%**.

**Test setup:**
- leave-one-person-out over UCI HAR persons 0–4;
- epoch selected on **source-person validation only** (no target peeking);
- Colab Tesla T4;
- **seeds 0–2**.

| Arm | Accuracy | Macro-F1 | Weighted-F1 | Δ accuracy vs control [95% CI] | Persons improved | Predeclared verdict |
|---|---|---|---|---|---|---|
| `ie_off` (control = released GILE behaviour) | 79.17 | 77.86 | 77.97 | — | — | — |
| `fix` (layout corrected) | 87.63 | 86.63 | 86.49 | +8.46 [+4.99, +11.93] | 5/5 | STRONG (Holm p 0.007) |
| `aug` (augmentation only) | 83.16 | 81.81 | 81.97 | +3.99 [−1.50, +9.48] | 4/5 | SUPPORTED |
| **`fix_aug`** (both) | **91.61** | **91.30** | **91.25** | **+12.44 [+6.08, +18.80]** | **5/5** | **STRONG** (Holm p 0.011) |

**How the comparison is done.** Differences are paired by (person, seed) and averaged per person. The confidence
interval is a t-interval over the 5 persons (df = 4). The success rule was written before any result:
[`experiment/PREDECLARED_PLAN.md`](experiment/PREDECLARED_PLAN.md).

**Earlier attempts, for comparison** (`docs/`):
| Attempt | Δ accuracy | Runs | Outcome |
|---|---|---|---|
| Repaired Independent Excitation | −0.99 | 75 | no accuracy gain |
| XPC-B, cross-person contrastive loss | +1.39 vs its control | 50 | inconclusive |

**Seed status:**
| Seeds | Status |
|---|---|
| 0–2 | complete: the main analysis |
| 3 | partial and **excluded** (10 of 15 runs printed metrics before the free Colab GPU quota ran out; recovered log only) |
| 4 | not run (GPU quota exhausted) |

## Repository layout
| Path | Content |
|---|---|
| `docs/` | Final report, results summary, implementation notes, contributions, literature review, stage-1 (IE) and stage-2 (XPC-B) reports |
| `experiment/PREDECLARED_PLAN.md` | Experimental plan and success rule (written before the results) + deviations log |
| `experiment/code/` | Shared GILE modules (copies of the team's `evaluation-code`; see below), `stage3/` (this work), `stage2_ref/` (stage-2 runner, used by a test) |
| `experiment/code/stage3/p3_variants.py` | The two changes: `apply_layout_fix(model)` and `PersonStyleAug` |
| `experiment/code/stage3/run_stage3.py` | Runner: `--arm {ie_off, fix, aug, fix_aug} --target {0..4} --seed k` |
| `experiment/code/stage3/checks/test_stage3.py` | 20 correctness checks (layout, augmentation physics, control equivalence, resume) |
| `experiment/code/stage3/runs_p3/ucihar/` | All 46 returned runs (45 + anchor): `config.json`, `history.csv` (150 epochs), `log.txt`, `metrics.json` |
| `experiment/code/stage3/manifests/` | SHA-256 manifests of every returned file (including the checkpoints not stored here) |
| `experiment/baseline_ie_off_metrics/` | Control runs (stage 1, `ie_off`, 5 targets × 5 seeds) |
| `experiment/reference_metrics_earlier_stages/` | Metrics of `released`, `ie_on`, `ie_off_bal`, `xpc_bal` (context rows) |
| `experiment/analysis_p3/` | `summary.txt`, `arm_summary.csv`, `per_target.csv`, `decision.json`, `verify_runs_output.txt` |
| `experiment/returned_results/` | Recovered printed metrics of the partial seed 3 |
| `experiment/colab_notebook.ipynb` | The Colab notebook used for the GPU runs (code embedded and SHA-256 checked) |
| `experiment/tools/` | Data preparation, runner loop, analysis, verification, report-number check, notebook builder |
| `earlier_stages_code/` | Stage-1 and stage-2 runners and tests, as run (reference) |

**Paths in the documents.** The reports were written in the author's local workspace, so some paths refer to it:
| Path in `docs/` | Where it is in this repository |
|---|---|
| `improvement-code/` | this repository's root |
| `improvement-code/experiment/…` | `experiment/…` (same layout) |
| `runs_full/ucihar/<arm>/…` (stage 1) | metrics only: `experiment/baseline_ie_off_metrics/` (`ie_off`) and `experiment/reference_metrics_earlier_stages/{released,ie_on}/` |
| `XPCB_TRANSFER_PACKAGE/…/runs_xpc_t4/…` (stage 2) | metrics only: `experiment/reference_metrics_earlier_stages/{ie_off_bal,xpc_bal}/` |
| `run_experiment.py`, `checks/…` (stage 1) | `earlier_stages_code/stage1_ie/` |
| `stage2_xpc/run_stage2.py`, `xpc.py` (stage 2) | `earlier_stages_code/stage2_xpcb/` (also `experiment/code/stage2_ref/`) |
| `Person3_checks/ie_gradient_check.py` | `earlier_stages_code/audit/ie_gradient_check.py` |
| `original-code/`, `evaluation-code/` | the team's reproduction code, elsewhere in the team repository (not modified) |

The full stage-1 and stage-2 run folders (histories, logs, checkpoints) are kept in the author's local workspace and
backup.

**What the shared modules contain.**
- `model_GILE_ucihar.py`: the released model plus an appended `ie_term` method used only by stage 1.
- `data_preprocess_ucihar.py`: the released loader with two numpy-compatibility fixes (`np.float` → `float`,
  `np.int` → `int`).
- `train.py`, `utils.py`, `sliding_window.py`: identical to the team's `evaluation-code`.
- **The layout fix is applied at run time** on the model instance. The model file itself is not edited.

**Not included:**
- the model checkpoints (`selected_model.pt`, 8.4 MB each; their SHA-256 are in the manifests);
- the dataset.

## Setup
```bash
pip install -r requirements.txt
```
- **Tested versions:** see `requirements.txt`.
- **For bit-exact reproduction of the reported numbers:** Tesla T4 + torch 2.14.1+cu130 (Colab). Other GPUs or
  versions give slightly different numbers, because training is deterministic only within one environment.

## Dataset
**Source:** UCI HAR, *Human Activity Recognition Using Smartphones*.
- **Archive:** https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
- **SHA-256:** `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`

**Expected layout** (the released loader's relative paths):
`experiment/code/data/ucihar/{train,test}/InertialSignals/*.txt` plus `y_{train,test}.txt` and
`subject_{train,test}.txt`, 22 files in total.

```bash
python experiment/tools/prepare_ucihar.py                      # downloads, verifies SHA-256, arranges the files
python experiment/tools/prepare_ucihar.py --zip <downloaded.zip>  # or use an archive you already have (the inner
                                                                 # "UCI HAR Dataset.zip" also works, without the SHA check)
```

**What the experiment uses:**
- the 9 raw inertial channels (`body_acc`, `body_gyro`, `total_acc` × xyz) in g and rad/s;
- 128-sample windows with 50% overlap;
- persons 0–4 as defined by the team loader.

## Reproducing the experiment
Run all commands from the repository root (this folder).

1. **Tests** (CPU is enough, about 7 min; synthetic data, no dataset needed):
   ```bash
   python experiment/code/stage3/checks/test_stage3.py
   ```
2. **One run** (GPU strongly recommended; about 1.5–2 min on a T4, much slower on a CPU):
   ```bash
   python experiment/code/stage3/run_stage3.py --arm fix_aug --target 1 --seed 0 --out_root runs_p3_rerun --no_last_ckpt
   ```
3. **The full main design**: the anchor + 3 arms × 5 persons × seeds 0–2. A new `--out_root` keeps the shipped results
   untouched; completed runs are skipped and interrupted runs resume.
   ```bash
   python experiment/tools/run_all_p3.py --anchor --seeds 0 1 2 --out_root runs_p3_rerun
   ```
   The anchor must print `EXACT MATCH ... True` on a T4 with torch 2.14.1+cu130.
   - *Colab alternative:* upload `experiment/colab_notebook.ipynb`, choose a T4 runtime, run the cells one at a time.
4. **Analysis**, exactly as predeclared:
   ```bash
   python experiment/tools/analyse_p3.py experiment/code/stage3/runs_p3_rerun/ucihar --out experiment/analysis_rerun
   ```
   To re-analyse the shipped runs instead:
   ```bash
   python experiment/tools/analyse_p3.py experiment/code/stage3/runs_p3/ucihar --out experiment/analysis_check
   ```
5. **Integrity and report checks:**
   ```bash
   python experiment/tools/verify_runs_p3.py experiment/code/stage3/runs_p3/ucihar --no_ckpt
   ```
   ```bash
   python experiment/tools/check_report_numbers.py docs/FINAL_REPORT.md
   ```

## Protocol (all arms)
- **Splits.** For each source person, the last 20% of each activity (in file order) is held out for validation.
  Training windows that overlap a validation window by half a window are removed; zero overlap is asserted.
- **Training.** 150 epochs, batch 64, class-balanced sampling, single Adam at lr 1.1e-3, cuDNN deterministic.
- **Selection.** The epoch is selected by the mean source-validation accuracy. The target person is used only for
  reporting. The released "best epoch on the target person" score is logged as a labelled diagnostic and never used.
- **Probes.** Logistic and MLP probes predict the person from the activity code z (posterior mean). They are fit on
  source-train and scored on source-validation, with a leak guard.

## Limitations
- 3 seeds, 5 persons, one dataset (UCI HAR). Opportunity and UniMiB-SHAR were not run with the fix.
- The mechanism behind the gain is a hypothesis: the convolutions now see per-sensor temporal patterns.
- See the final report, §8.
