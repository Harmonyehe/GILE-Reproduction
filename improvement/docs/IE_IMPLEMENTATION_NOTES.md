# Independent Excitation (IE): paper vs released code vs repaired implementation

Evidence labels: **[PAPER]** checked visually against the rendered PDF (Qian et al., AAAI 2021, pp.3–4);
**[CODE]** read in the released code; **[RUN]** verified by executing `checks/verify_ie_design.py` / `checks/test_variants.py`.

## 1. What the paper specifies [PAPER]
- p.4: "the Independence Excitation mechanism minimizes the accuracy of DC_d when z is fed into DC_d. Likewise, the accuracy of DC_y is forced to be minimized when z_d is fed into DC_y."
- Objective (p.4), with ℓ = cross-entropy (defined under L_DC, p.3) and N_S = Σ_d N_d over the D **source** domains:

  L_IE = **−** (1/N_S) Σ_{d=1..D} Σ_{i=1..N_d} [ ℓ(y_i^d, DC_y(z_d; w_y)) + ℓ(d, DC_d(z; w_d)) ]

- Total loss (Eq. 2, p.4): L = L_elbo + α L_DC + γ L_IE, minimised. Because of the leading minus, minimising γL_IE **maximises** the cross-entropy of the swapped ("cross") predictions — i.e. pushes the cross classifiers toward wrong answers.
- Optimisation (p.4): "one optimizer for the generative model and the other optimizer for the disentangling classifiers, and the two optimizers are trained iteratively"; L_elbo is trained separately from L_DC + L_IE.
- Settings (p.5): Adam, lr 1e-3, weight decay 1e-3, batch 64, max 100 epochs.

## 2. What the released code does [CODE][RUN]
- `GILE.classifier()` runs entirely under `torch.no_grad()` and returns **hard one-hot argmax** vectors.
- `loss_function_false` = weight_true·[CE(DC_d(z_d)) + CE(DC_y(z))] − weight_false·[CE(DC_d(z)) + CE(DC_y(z_d))] computed on those one-hots, then `loss.requires_grad = True` → a detached leaf (`grad_fn=None`).
- Consequence [RUN]: the IE call changes **no weight**: training with vs without it gives bit-identical weights (difference 0.00 after 1 and 10 steps).
- Both optimisers (`Adam lr 1e-4` and `false_optimizer Adam lr 1e-3`) hold **all** parameters and see the **same** main-loss gradients, so released training is mathematically one Adam step at lr 1.1e-3 (difference 1.2e-7 after one step = rounding). This also differs from the paper's generator/classifier optimiser split and has no weight decay.
- The sign intent (`true − false`) matches the paper; only the gradient path is broken.

## 3. Repaired IE used here (variant `ie_on`) — a justified modification, not the paper's literal method
| Aspect | Paper | Released code | Repaired (`ie_on`) | Why |
|---|---|---|---|---|
| Inputs to cross classifiers | sampled z, z_d | posterior means, no gradient | sampled z, z_d from the same forward pass as the main loss | gradient must reach the encoders |
| Classifier outputs | probabilities (CE) | hard one-hot argmax | log-softmax of real logits | differentiable |
| Cross objective | −CE (maximise error) | −CE on one-hots (no effect) | **KL(Uniform ‖ softmax)** = −(1/K)Σ_k log p_k − log K, summed over the batch | −CE is unbounded below (→ −∞) and rewards confidently *wrong* predictions, which still carry information; KL to uniform is ≥ 0, = 0 iff the cross prediction is uninformative, and its logit gradient (p − 1/K) does not vanish [RUN T4] |
| Direction | lower cross accuracy | — | same direction: minimising KL raises cross-entropy toward its uninformative value | [RUN T3: steps move max-prob toward 1/K] |
| Person-head classes | D source domains (sum over d = 1..D) | 5 outputs incl. unseen target | **only the source persons (K = 4)** | the target person's class never occurs in training; a uniform-over-5 target would push z toward "looks like the unseen person" |
| Who receives the IE gradient | encoders and classifiers (classifier optimiser) | nobody | **encoders only** (classifier weights detached in the IE term) | a shared, non-detached head could satisfy the penalty by flattening its own outputs on cross inputs without changing z |
| Optimisers | two, iterative (generator / classifiers) | two, both on all params (≡ one Adam at 1.1e-3) | one Adam at lr 1.1e-3 for all of main loss + γ·IE | with detached heads the IE gradient touches only encoders, so one optimiser on the summed loss = simultaneous update; lr 1.1e-3 matches the released effective rate so `ie_off`↔`released` are equivalent and `ie_on`↔`ie_off` differ only by IE |
| γ | "similar order of magnitude" | weight_false = 1000 | γ = 1000, sum reduction (fixed a priori, never tuned on target) | matches the existing aux CE multipliers (1000, sum) |

Wording for the report: **"a repaired, differentiable IE in the spirit of the paper"** — not "the paper's IE".

## 4. Known limitation of the repair (and of the paper's shared-classifier design)
DC_d is a 5×50 linear head trained only on z_d. Making DC_d(z) uniform only requires z's person information to be orthogonal to that head's few weight directions; it does not require removing it (null-space shortcut). Hence disentanglement is judged by **independent probes**, not by cross-classifier accuracy. Pilot (target 0, seed 0): frozen-head cross accuracy person←z ≈ chance in all variants (incl. released) while the person probe found 57–66% (majority 27%).

## 5. Probe methodology [CODE][RUN]
- Model: the **selected-epoch** checkpoint (chosen on source-validation accuracy only).
- Features: encoder posterior **means** (z from `qzy`, z_d from `qzd`).
- Probe-train set: source-train windows of the 4 source persons (contiguous, overlap-purged split). Probe-test set: source-validation windows of the same persons.
- Leakage guards (asserted every run): train/val share no window and no half-window overlap (`split_person`); probe-train vs probe-test identical windows = 0; target windows used = 0. Standardisation is fitted inside the pipeline on probe-train only.
- Probes: StandardScaler + LogisticRegression (linear) and StandardScaler + MLP(64) (seeded); chance reported as the majority-class rate of the probe-test set.
- Reported: person←z (lower = less person information in the activity code), activity←z_d (lower = less activity information in the person code), person←z_d (sanity, should be high).
- Caveats: (i) source-val windows also determined the selected epoch (via activity accuracy, not person labels) — identical for all variants; (ii) val windows are the temporally last 20% of each activity, so the probe measures generalisation to later segments of the same persons; (iii) probes measure linear / shallow-nonlinear decodability only.
