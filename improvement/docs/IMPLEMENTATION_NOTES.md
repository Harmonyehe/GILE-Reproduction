# Implementation notes (all improvement stages)

This file covers the exact code for each improvement tested, the model architecture, loss functions, data
transformations, training configuration and tests. Detailed earlier notes: `IE_IMPLEMENTATION_NOTES.md` (stage 1) and
`XPCB_IMPLEMENTATION_NOTES.md` (stage 2). Stage 3 is documented in full here.

## 0. Shared base (unchanged in every stage)
**Model: GILE as released** (`model_GILE_ucihar.py`, byte-identical copies everywhere)
- **Encoders.**
  - `qzd` (person code z_d, 50-d) and `qzy` (activity code z, 50-d).
  - Each is 4 × [Conv2d (1×5 then 1×1) → ReLU → MaxPool (1×2)] with channels 9→1024→512→128→64, then linear heads
    for the posterior mean and scale.
- **Decoder.** `px` (MaxUnpool + ConvTranspose), reconstructing `(B,128,9,1)`.
- **Learned conditional priors.** `pzd(d)` and `pzy(y)`.
- **Classifiers.** `qd(z_d)` and `qy(z)`.
- **Main loss** (`loss_function`):
  `MSE(x̂, x) − β_d·Σ[log p(z_d|d) − log q(z_d|x)] − β_y·Σ[log p(z|y) − log q(z|x)] + 1000·CE(qd(z_d), d) + 1000·CE(qy(z), y)`.
- **Data:** UCI HAR raw inertial signals; persons 0–4 as in the team code. Channel order
  `[body_acc xyz, body_gyro xyz, total_acc xyz]` in g and rad/s. No normalisation.
- **Protocol** (all stages):
  - leave-one-person-out;
  - source persons split train/val per activity (contiguous last 20%), with half-window overlaps purged;
  - epoch selected on source-val accuracy;
  - target metrics at the selected epoch are primary, last epoch secondary;
  - the leaky best-target epoch is diagnostic only;
  - probes: logistic + MLP(64), person ← z and activity ← z_d, fit on source-train, scored on source-val, with a leak
    guard.
- **Training** (control `ie_off`):
  - person-sequential batches of 64 with a class-balanced `WeightedRandomSampler` (seed·100+k);
  - a single Adam at lr 1.1e-3 (the released effective rate);
  - 150 epochs, β_d = β_y = 1;
  - cuDNN deterministic.

## 1. Stage 1: repaired Independent Excitation (`ie_on`), 75 runs
See `IE_IMPLEMENTATION_NOTES.md`.
- The released `loss_function_false` computes the classifiers under `torch.no_grad()` and sets
  `loss.requires_grad = True`. So it gives **no gradient to any weight**: 0 of 53 parameter groups get one (verified
  by `Person3_checks/ie_gradient_check.py`).
- The repaired version uses real logits with a gradient path, and masks the absent target person.

## 2. Stage 2: XPC-B (`ie_off_bal`, `xpc_bal`), 50 runs + anchor
See `XPCB_IMPLEMENTATION_NOTES.md`.
- Person-balanced batches (16 windows × 4 source persons).
- SupCon-style cross-person loss on the posterior mean of z, through a projection head g (50 → 64 → 32), τ = 0.1.
- λ is set by the 10%-of-initial-loss rule.

## 3. Stage 3: input-layout correction (C) and person-style augmentation (B)
Folder: `improvement-code/experiment/` (isolated; the plan was written before any result: `experiment/PREDECLARED_PLAN.md`).

### 3.1 Files
| File | Role |
|---|---|
| `code/{model_GILE_ucihar,data_preprocess_ucihar,train,utils,sliding_window}.py` | byte-identical copies of the shared modules (not edited) |
| `code/stage3/p3_variants.py` | the two changes: `apply_layout_fix`, `PersonStyleAug` |
| `code/stage3/run_stage3.py` | runner, `--arm {ie_off, fix, aug, fix_aug}`; derived from the resumable stage-2 runner, keeping only the control's training path |
| `code/stage3/checks/test_stage3.py` | 20 checks (layout, augmentation, runner equivalence, resume) |
| `code/stage2_ref/` | read-only copy of the stage-2 runner, used only by test R1 |
| `tools/build_p3_notebook.py`, `colab_notebook.ipynb` | Colab notebook with embedded files, each verified by SHA-256 |
| `tools/analyse_p3.py` | analysis exactly as predeclared |
| `baseline_ie_off_metrics/` | read-only copies of the 25 stage-1 `ie_off` metrics (the control) |

### 3.2 Change C: encoder input layout
**Released** (identical line in `qzd`, `qzx` and `qzy`):
```python
x_img = x.view(-1, x.shape[2], 1, x.shape[1])      # x: (B, 128, 9, 1) contiguous
```
- `view` keeps the memory order (time-major: t0s0, t0s1, …, t0s8, t1s0, …). It only relabels the shape as
  `(B, 9, 1, 128)`.
- So "channel" c holds flat elements c·128 … c·128+127, that is about 14 consecutive time steps × all 9 sensors,
  interleaved.
- Along the convolution's width axis, neighbouring positions are different sensors at (nearly) the same time step.
- **Verified by test C3:** every width-5 kernel window spans 5 different sensors. Test C4: each "channel" holds all 9
  sensors and time steps 0–14.
- **The intended layout** follows from the decoder `px`, which outputs `(B, 9, 1, 128)` (ConvTranspose `out_channels =
  n_feature = 9`) and then calls `.permute(0, 3, 1, 2)` to return to `(B, 128, 9, 1)`.

**Corrected:**
```python
x_img = x.permute(0, 2, 3, 1)                       # (B,128,9,1) -> (B,9,1,128): channel = sensor, width = time
```
Implementation, `p3_variants.apply_layout_fix(model)`:
- replaces `forward` on the encoder *instances* (`qzd`, `qzy`; `qzx` does not exist because zx_dim = 0) with an exact
  copy of the released forward, where only this line differs;
- `model_GILE_ucihar.py` is not edited;
- parameters, initialisation, decoder, losses and the reconstruction target are unchanged.

Test C6 proves the copy is exact: the patched encoder on `x` gives **bit-identical** outputs to the released encoder on
memory rearranged so that its `.view()` produces the correct layout.

### 3.3 Change B: person-style sensor augmentation
`p3_variants.PersonStyleAug`, applied to **training batches only**, after the sampler and before the GPU transfer.
Validation, target, probe and selection data are never augmented. Each window is transformed with p = 0.5. Units are
raw physical units.

**Steps:**
1. **Intensity scaling.** s ~ N(1, 0.1²), clipped to [0.7, 1.3].
   - `body_acc ← s·body_acc + n_a`, `body_gyro ← s·body_gyro + n_g`.
   - `total_acc ← (total_acc − body_acc) + new body_acc`. The gravity component is preserved exactly; noise
     σ = 0.01.
2. **Orientation.** One Rodrigues rotation R per window (axis uniform on the sphere, angle U(−15°, 15°)), applied to
   all three triads, since all are in the same phone frame. Triad norms and the gravity magnitude are preserved
   (tests A1, A2).

**Why these, and nothing else:**
- they model between-person differences in movement vigour and phone placement;
- time-warping/slicing are excluded because they can alter cadence (walking vs stairs);
- large rotations are excluded because they destroy the gravity-orientation cue for sitting, standing and lying.

**Labels.** Activity label y and person label d are kept. Keeping d means z_d must recognise the person despite the
style perturbation (see limitations).

**RNG isolation.**
- The augmentation owns a `torch.Generator` (seed·7919 + 17).
- It draws a fixed number of random numbers per window, whether or not the window is transformed.
- So the model initialisation, batch order and VAE sampling noise are identical to the control (tests A5, A6, R2).

### 3.4 Training configuration (all four arms)
Identical to the stage-1 `ie_off` control (section 0). The only differences are the flags `layout_fix` and
`augmentation`. `config.json` of every run records:
- the flags;
- the augmentation parameters;
- the initial-parameter hash, identical across arms;
- the environment;
- SHA-256 prefixes of the code files.

### 3.5 Tests (`code/stage3/checks/test_stage3.py`)
| ID | Check |
|---|---|
| C1–C4 | shapes; fixed layout maps channel c → sensor c and width t → time t for all 1152 positions; released layout mixes 5 sensors per kernel window; each released "channel" holds 9 sensors × ~14 time steps |
| C5 | patch touches only `qzd`, `qzy`; initialisation identical |
| C6 | patched forward is bit-identical to the released forward on correctly rearranged memory (exact copy) |
| C7 | patched forward differs from the released one on the same input |
| C8 | finite loss; gradients reach the first conv of both encoders |
| C9 | reconstruction shape unchanged |
| A1–A2 | rotation is proper (RRᵀ = I, det = +1, at most 15°); preserves triad norms and the gravity magnitude |
| A3 | non-selected windows are bit-identical; selected windows change |
| A4 | scaling acts on the body components only; gravity unchanged; s within [0.7, 1.3] |
| A5–A6 | deterministic per seed; draws are independent of the data; global RNG untouched |
| R0 | all four arms and the stage-2 reference run end to end (synthetic data, 3 epochs) |
| R1 | the stage-3 control arm reproduces the stage-2 runner's `ie_off` exactly (history, target metrics, probe numbers) |
| R2 | identical initial parameters in all four arms |
| R3 | each new arm changes training |
| R4 | `fix_aug` interrupted after epoch 1 and resumed is identical to the uninterrupted run |

Results: 20 of 20 passed on the laptop CPU (`experiment/code/stage3/checks/test_stage3_local_output.txt`). They also
passed on the Colab T4 (cell 4 completed its `ALL PASSED` assertion) before any experiment.

### 3.6 Post-run integrity checks (`experiment/tools/verify_runs_p3.py`)
V1–V10:
- arm flags;
- target not among the sources;
- split and probe leak guards;
- T4 / torch 2.14.1 environment;
- code SHA-256 equal to the local files;
- identical initial parameters per seed;
- 150 finite epochs, with the selection rule recomputed;
- no resumes;
- checkpoints present;
- every manifest entry re-hashed.

Result for the 46 returned runs: **ALL CHECKS PASSED** (`experiment/analysis_p3/verify_runs_output.txt`).
