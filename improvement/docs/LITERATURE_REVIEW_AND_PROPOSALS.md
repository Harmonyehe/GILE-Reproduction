# GILE, stage 2: literature review, diagnosis and improvement proposals

Date: 16 Nov 2026. Status: **research only**. No code was changed, nothing was run on a GPU, and no result was overwritten.

Evidence labels used throughout:
- **[MEASURED]**: from our 75 runs or from a read-only script run on our data.
- **[CODE]**: verified by reading the code or by a test.
- **[REPORTED]**: what a paper reports, not re-checked by us.
- **[READ: full / partial / abstract]**: how deeply I read a paper.
- **[HYPOTHESIS]** or **[MY INTERPRETATION]**: my own reasoning, not established fact.

Model note: Opus 5.5 (the strongest model available here) did the literature synthesis, diagnosis and recommendation. A balanced model is enough for later routine coding.

---

## 0. Executive summary

1. **The core problem is a person-shift gap, not an accuracy ceiling.**
   - On held-out windows from the 4 training people, the models reach 93.0–94.5% accuracy [MEASURED].
   - On the unseen person they reach 77.4–78.4% [MEASURED].
   - Repaired IE *raised* accuracy on the training people (+1.5 points against `ie_off`) while *lowering* accuracy on the new person (−1.0) [MEASURED].
   - So IE made the model fit the 4 training people better, not generalise better.

2. **A defect neither we nor the audit had noticed before [CODE]: GILE trains on single-person minibatches.**
   - Every epoch, the training loop goes through person A's batches, then person B's, and so on (`train.py` line 12; `run_experiment.py` line 228 copies this order).
   - Each person gives only about 3–4 batches of 64 per epoch.
   - This has four consequences:
     - (a) It is the root cause of the person-prior collapse we already found. Batch normalisation of a one-hot input that is identical across a batch gives the same prior for every person. Our test also shows that a mixed batch gives person-specific priors.
     - (b) The person classifier `DC_d` and its IE copy are trained on batches containing only one class.
     - (c) Every cross-person method in the literature (alignment, MMD/CORAL, conditional adversaries, cross-person contrastive learning) needs several people in a batch, so none of them can run as the code stands. This also invalidates the assumption behind the XPC proposal from earlier today.
     - (d) It departs from standard domain-generalisation practice. DomainBed builds every step from one minibatch *per training domain* [READ: code].
   - On the side: with `drop_last=True`, up to 63 windows per person (about 5–20%) are discarded each epoch.

3. **The literature does not support "remove person information" as a reliable route to accuracy.**
   - Adversarial removal leaves the attribute recoverable by a fresh probe (Elazar & Goldberg 2018) [REPORTED].
   - Forcing marginal invariance provably hurts when label distributions differ (Zhao et al. 2019) [REPORTED].
   - On DomainBed, no invariance method beat plain training (ERM) by more than about 1 point (Gulrajani & Lopez-Paz 2021) [REPORTED].
   - In an independent HAR benchmark, GILE was reported as the *weakest* cross-person method: 70.7 average, against 75.7 for ERM (Lu et al. 2023, DIVERSIFY) [REPORTED; summary-level read, verify before quoting].
   - In HAR, the methods with the strongest reported gains rely on **diversity** (sensor augmentation) and **class-conditional or cross-domain structure** (AFFAR, DDLearn, SelfReg) [REPORTED].

4. **Recommendation: person-balanced minibatches plus cross-person supervised contrastive regularisation on z ("XPC-B").**
   - It is the only candidate that:
     - acts on the *whole geometry* of z, which fits our finding that linearly decodable person information did not change;
     - removes *only the person variation inside each activity*, which is exactly the variation that can make a new person's activity look like another activity;
     - avoids discarding activity information, since it is class-conditional (Zhao et al.);
     - is stable and cheap.
   - The batching change is a necessary precondition, not a second idea. It also repairs GILE's own prior and person classifier.
   - We evaluate it with a **batching-only control arm**, so the effect of the contrastive term and the effect of batching are measured separately.
   - Runner-up: **person-style sensor augmentation**. It has the strongest published accuracy evidence in HAR, but only an indirect link to separation and a semantic conflict with z_d (§5).

---

## 1. Our current position (checked against the files)

Sources: `RESULTS_REPORT.md`, `IE_IMPLEMENTATION_NOTES.md`, `PROJECT_PROGRESS.md`, `runs_full/ucihar/*.csv`, and the run scripts. `RESULTS_REPORT_FINAL.md` **does not exist**.

| Item | Value | Status |
|---|---|---|
| Target weighted-F1, epoch picked on source validation (released / ie_off / ie_on) | 76.77 / 77.02 / 75.80 | [MEASURED] matches your summary |
| Accuracy | 78.15 / 78.35 / 77.35 | [MEASURED] |
| Source-validation accuracy at the selected epoch | 93.38 / 92.99 / 94.49 | [MEASURED] (new read-only computation) |
| ie_on − ie_off, weighted-F1 | −1.22, 95% CI [−6.92, +4.48]; per target −3.0, −6.7, +3.2, −3.5, +3.9 | [MEASURED] |
| Person ← z probe, linear / MLP (chance 27.1) | released 52.35 / 64.88; ie_off 52.31 / 64.83; ie_on 53.38 / 61.07 | [MEASURED] |
| Activity ← z_d probe, linear / MLP | 69.91 / 73.33; 68.58 / 72.20; 66.79 / 69.66 | [MEASURED] |
| Noise between two mathematically identical procedures (per-run paired SD of weighted-F1) | ≈ 4.8 | [MEASURED] |
| IE design | frozen ReLU→Linear heads, KL to uniform, person head limited to the 4 source people, γ = 1000, one Adam optimiser | [CODE] (`IE_IMPLEMENTATION_NOTES.md` §3) |
| Channel mixing (`x.view` relabels memory instead of transposing) | present in all variants | [CODE], effect never measured |
| Person prior identical for all people during training | present in all variants | [CODE] (`checks/test_variants.py` test 6) |
| Single-person minibatches, person-sequential order | present in all variants | [CODE] **new finding** (`train.py:12`, `run_experiment.py:228`) |
| Per-person activity shares, persons 0–4 (UCI subjects 1–5, labels only) | 13.5–27.4%; person 0 has 27.4% walking against 17–20% for the others | [MEASURED] (label counts only, no model output) |

A small correction to your summary: "53–61%" combines the linear probe (≈52–53%) and the MLP probe (≈61–65%). Person information in z is above chance under both.

---

## 2. Literature review

### 2.1 Overview by research direction

**A. Domain generalisation and cross-person HAR.**
- The generalisation-to-new-people problem is standard domain generalisation (DG): train on several source domains, test on an unseen one.
- The most important methodological paper is DomainBed [P3]. Under one shared protocol and honest model selection (training-domain validation, as we use), no DG algorithm beat ERM by more than about 1 point on average.
- HAR-specific DG work [P12, P13, P14] reports gains of 2–10 points over ERM. The gains come from:
  - combining invariant and domain-specific features (AFFAR);
  - augmentation-driven diversity plus supervised contrastive learning (DDLearn);
  - discovering latent sub-domains and aligning them adversarially (DIVERSIFY).
- DIVERSIFY's cross-person table includes GILE and reports it below ERM [P14].

**B. Disentanglement.**
- GILE is a close relative of DIVA [P2]: a VAE with separate class and domain latent codes and *conditional priors* p(z_d|d) and p(z_y|y). In DIVA, these conditional priors (plus auxiliary classifiers) drive the separation.
- In our runs, the person prior is not conditional during training (§1). So GILE's main built-in separation route is switched off, and IE was left to do all the work.
- Locatello et al. [P8] show that disentanglement needs supervision or inductive bias, and that better disentanglement scores did not reliably improve downstream tasks. This supports your caution: stronger separation does not automatically mean better accuracy.

**C. Contrastive and metric learning.**
- Supervised contrastive learning (SupCon) [P9] pulls same-class samples together and pushes other classes apart, using many positives per anchor.
- SelfReg [P10] applies a positive-only, same-class version as a DG regulariser. It reports a DomainBed average of 67.3 against 66.6 for ERM: small but positive.
- DDLearn [P13] adds SupCon to sensor augmentation for cross-person HAR, with large reported gains. But its ablation never isolates the contrastive term.
- Qian et al. 2022 [P16] study contrastive learning on small wearable datasets.
- I found **no paper that isolates cross-person positives as a single change in sensor HAR**. That is a gap our experiment can fill.

**D. Adversarial learning.**
- DANN [P4] learns invariance with a gradient-reversal discriminator. CDANN [P5] makes the invariance class-conditional, to avoid the label-shift problem proved by Zhao et al. [P6].
- Elazar & Goldberg [P7] show that even when the adversary is at chance, a post-hoc probe recovers the attribute (e.g. 56–63%). Stronger, larger or ensembled adversaries reduced this leakage but did not remove it, and made training unstable.
- In HAR benchmarks, DANN results are mixed: about 2.5 points below AFFAR [P12] and +2.0 over ERM in DIVERSIFY's table [P14] [REPORTED].

**E. Sensor-specific methods.**
- Um et al. [P15] introduced wearable augmentations: rotation, permutation, time-warping, scaling, magnitude-warping, jittering and cropping. They report 77.5 → 86.9% on a Parkinson's task.
- In DDLearn's ablation, augmentation alone added 9.21 points on DSADS [REPORTED].
- CrossHAR [P17] augments data by modelling how the sensor signal is generated, then pretrains with self-supervision. It reports +10.83% on cross-*dataset* HAR.
- I found no HAR paper about our specific channel-layout bug. The general point that a CNN needs physically meaningful neighbouring inputs is standard.

**F. Other approaches.**
- CORAL [P11] and MMD (used in AFFAR [P12]): distribution alignment between domains.
- IRM [P18]: learns predictors that are stable across environments. It was not better than ERM in DomainBed [REPORTED].
- MLDG [P19]: episodic meta-learning. It was poor on HAR in AFFAR's tables (67.5 against 82.1 for ERM on DSADS) [REPORTED].
- Self-supervised pretraining (CrossHAR, DDLearn): promising but too large for our deadline.

### 2.2 Individual paper summaries

How deeply each paper was read:
- **[READ: full]**: P1 (local PDF), P13 (full arXiv HTML), P3 (paper plus DomainBed code).
- **[READ: partial]**: verbatim text up to the method or results sections, with the rest from a summary: P6, P7, P10, P12, P14.
- **[READ: abstract]**: metadata plus established background knowledge: P2, P4, P5, P8, P9, P11, P15–P19.
- Every number marked [REPORTED] should be checked in the PDF before it goes into the report.

**P1. Qian, Pan & Miao (2021). Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition (GILE). AAAI 35(13):11921–11929.** [READ: full]
- **Problem:** cross-person HAR without any target-person data.
- **Method:**
  - A VAE with two encoders: z (domain-agnostic) and z_d (domain-specific).
  - Auxiliary classifiers DC_y and DC_d.
  - IE, which maximises the error of the "swapped" classifiers DC_d(z) and DC_y(z_d).
  - Two optimisers that alternate.
- **Data:** UCI HAR (5 people), Opportunity and UniMiB-SHAR, leave-one-person-out.
- **Result:** reports 88.17 weighted-F1 on UCI.
- **Limitations we found:**
  - IE has no gradient in the released code.
  - The epoch is selected on the target person.
  - Single-person batches.
  - Channel mixing.
- **Relevance to us:** this is the baseline.

**P2. Ilse, Tomczak, Louizos & Welling (2020). DIVA: Domain Invariant Variational Autoencoders. MIDL, PMLR 121:322–348.** [READ: abstract]
- **Problem:** DG through a generative model.
- **Method:** three latent codes (domain, class, residual), each with its own *conditional prior*, plus auxiliary classifiers. Independent priors are the mechanism that separates the codes.
- **Data:** rotated MNIST; malaria cell images, where each domain is a patient.
- **Result:** reported improvements over the DG methods of the time, and it can use unlabelled data.
- **Limitation:** evaluated on images only.
- **Relevance to us:** high. It explains why GILE's broken person prior (§1) matters. Without a working p(z_d|d), nothing in the objective pulls person information *into* z_d.

**P3. Gulrajani & Lopez-Paz (2021). In Search of Lost Domain Generalization. ICLR. arXiv:2007.01434.** [READ: full + code]
- **Problem:** DG results were not comparable across papers because protocols and model selection differed.
- **Method:** the DomainBed testbed with three model-selection rules (training-domain validation, leave-one-domain-out, oracle).
- **Finding:** with careful ERM, no method beat it by more than about 1 point on average.
- **Code detail [READ: code]:** `train_minibatches_iterator = zip(*train_loaders)`. Every step contains one minibatch per training domain.
- **Limitation:** image benchmarks only.
- **Relevance to us:**
  - It justifies our protocol (training-domain validation).
  - It sets realistic expectations: invariance methods rarely give large gains.
  - It shows multi-domain batches are standard practice.

**P4. Ganin et al. (2016). Domain-Adversarial Training of Neural Networks. JMLR 17(59):1–35.** [READ: abstract]
- **Method:** a feature extractor trained against a domain classifier through gradient reversal.
- **Relevance to us:** the template for any person adversary on z. Our frozen-head IE is a weak, non-adaptive version of it.

**P5. Li, Tian, Gong, Liu, Liu, Zhang & Tao (2018). Deep Domain Generalization via Conditional Invariant Adversarial Networks (CDANN). ECCV, pp. 624–639. doi:10.1007/978-3-030-01267-0_38.** [READ: abstract]
- **Method:** makes the *class-conditional* feature distribution P(z|y) invariant across domains, using adversaries conditioned on the class.
- **Limitation:** the authors note it needs the target not to be severely class-imbalanced.
- **Relevance to us:** the adversarial alternative to XPC.

**P6. Zhao, Tachet des Combes, Zhang & Gordon (2019). On Learning Invariant Representations for Domain Adaptation. ICML, PMLR 97:7523–7532.** [READ: partial, theorem verbatim]
- **Result (Thm 4.3):**
  ```
  ε_S + ε_T ≥ ½ (d_JS(P_S(Y), P_T(Y)) − d_JS(P_S(Z), P_T(Z)))²
  ```
  If the representation is made perfectly invariant (d_JS of Z = 0) while the label distributions differ, the combined error is bounded below.
- Their digit experiments show DANN's target accuracy rising and then falling as training continues.
- **Relevance to us:**
  - Complete *marginal* person-invariance is undesirable when people's activity mixes differ.
  - In UCI the label shift is small, except person 0 with 27% walking [MEASURED]. So this is a moderate concern for us, not a dominant one.
  - Class-conditional methods avoid it.

**P7. Elazar & Goldberg (2018). Adversarial Removal of Demographic Attributes from Text Data. EMNLP, pp. 11–21. ACL D18-1002.** [READ: partial]
- **Finding:** the adversary reaches chance, yet a post-hoc probe recovers race, gender or age at 56–63%.
- Larger adversaries, a larger weight λ (training failed above 5) and adversary ensembles reduced the leakage but did not remove it.
- **Relevance to us:** a direct analogue of what we see. Our frozen IE head is near chance on person ← z, while our probes reach 52–65%. Replacing IE with a trained adversary should improve separation, but the literature predicts it will remain incomplete.

**P8. Locatello et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. ICML, PMLR 97:4114–4124.** [READ: abstract]
- **Finding:** disentanglement is impossible without inductive biases. Across more than 12,000 trained models, better disentanglement did not reliably help downstream tasks.
- **Relevance to us:** separation scores are not a substitute for the accuracy test.

**P9. Khosla et al. (2020). Supervised Contrastive Learning. NeurIPS. arXiv:2004.11362.** [READ: abstract]
- **Method:** the SupCon loss, with many same-class positives per anchor.
- **Result:** ImageNet gains and robustness benefits.
- **Relevance to us:** the loss form for XPC.

**P10. Kim, Yoo, Park, Kim & Lee (2021). SelfReg: Self-supervised Contrastive Regularization for Domain Generalization. ICCV, pp. 9619–9628. doi:10.1109/ICCV48922.2021.00948.** [READ: partial]
- **Method:**
  - Pairs each sample with a random same-class sample from the batch (not explicitly from a different domain).
  - Uses positive pairs only, with an L2 loss through a small "perturbation" MLP to prevent collapse.
  - Adds same-class mixup across domains.
- **Result:** DomainBed average 67.3 against 66.6 for ERM [REPORTED].
- **Relevance to us:** evidence that same-class alignment is a safe, modestly positive DG regulariser.

**P11. Sun & Saenko (2016). Deep CORAL: Correlation Alignment for Deep Domain Adaptation. arXiv:1607.01719 (ECCV 2016 workshops).** [READ: abstract]
- **Method:** matches second-order statistics (covariances) of features across domains.
- **Relevance to us:** a cheap alignment alternative. It is marginal, not class-conditional.

**P12. Qin, Wang, Chen, Lu & Jiang (2022). Domain Generalization for Activity Recognition via Adaptive Feature Fusion (AFFAR). ACM TIST. arXiv:2207.11221.** [READ: partial]
- **Method:**
  - A shared CNN, plus one domain-specific branch per source domain.
  - Pairwise MMD alignment between source domains.
  - A domain classifier whose output fuses the branches at test time.
- **Data:** DSADS, USC-HAD and PAMAP2, leave-one-group-out, source validation.
- **Result (weighted-F1) [REPORTED]:**
  - DSADS: 85.4 against 82.1 for ERM and 82.9 for DANN.
  - USC-HAD: 72.6 against 69.5 for ERM.
  - PAMAP2: 82.2 against 77.1 for ERM.
  - Ablation: both the invariant and the domain-specific terms help.
- **Limitations:**
  - Groups of people, not single people.
  - Slower at inference.
  - GILE is not compared.
- **Relevance to us:**
  - The closest HAR analogue to GILE's "invariant + specific" idea.
  - It reaches invariance by *alignment*, not by IE.

**P13. Qin et al. (2023). Generalizable Low-Resource Activity Recognition with Diverse and Discriminative Representation Learning (DDLearn). KDD. doi:10.1145/3580305.3599360.** [READ: full]
- **Method:**
  - Seven sensor augmentations (rotation, permutation, time-warping, scaling, magnitude-warping, jittering, random sampling).
  - An auxiliary task that predicts which augmentation was applied.
  - A diversity-preserving discriminator.
  - SupCon on activity labels.
  - Batches of 64 original plus 64 augmented samples.
- **Data:** DSADS, PAMAP2 and USC-HAD, with groups of people as targets.
- **Result:** at 20% of the training data, 82.7 average against 64.6 for ERM [REPORTED]. The ablation is *cumulative*: augmentation alone gives +9.21 on DSADS, and the contrastive term is never isolated.
- **Limitations:** no comparison with GILE, DANN or CORAL; 3 seeds.
- **Relevance to us:** the strongest HAR evidence that **diversity** improves accuracy for unseen people.

**P14. Lu, Wang, Sun, Chen & Xie (2023). Out-of-Distribution Representation Learning for Time Series Classification (DIVERSIFY). ICLR. arXiv:2209.07027.** [READ: partial; HAR table from summary]
- **Method:**
  - Iteratively finds K latent sub-domains, those with the worst-case distribution gap.
  - Then trains DANN-style invariance to them.
- **Data:** DSADS, USC-HAD, PAMAP and UCI-HAR, plus EMG, speech and WESAD.
- **Cross-person HAR averages [REPORTED, verify]:** ERM 75.7, DANN 77.7, CORAL 75.7, Mixup 78.6, **GILE 70.7**, DIVERSIFY 81.0.
- **Limitations:** K must be set by hand; extra complexity.
- **Relevance to us:**
  - Independent evidence that GILE as published underperforms simple baselines.
  - Mixup and DANN are competitive.

**P15. Um et al. (2017). Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks. ICMI, pp. 216–220. doi:10.1145/3136755.3136817.** [READ: abstract]
- **Method:** defines the standard wearable augmentations above.
- **Result:** 77.54 → 86.88% [REPORTED].
- **Caution from the paper:** some transformations can change the label.
- **Relevance to us:** rotation could destroy gravity cues needed for sitting, standing and lying in UCI's `total_acc` channels [HYPOTHESIS].

**P16. Qian, Tian & Miao (2022). What Makes Good Contrastive Learning on Small-Scale Wearable-based Tasks? KDD, pp. 3761–3771. arXiv:2202.05998.** [READ: abstract]
- A systematic study of contrastive learning components for wearable HAR, with the CL-HAR library.
- **Relevance to us:** design choices for XPC, such as the projection head and temperature.

**P17. Hong et al. (2024). CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining. IMWUT. doi:10.1145/3659597.** [READ: abstract]
- **Method:** augmentation based on how sensor data is generated, then hierarchical self-supervised pretraining, then fine-tuning.
- **Result:** +10.83% on unseen datasets [REPORTED].
- **Relevance to us:** a heavyweight approach, outside our deadline.

**P18. Arjovsky, Bottou, Gulrajani & Lopez-Paz (2019). Invariant Risk Minimization. arXiv:1907.02893.** [READ: abstract]
- **Method:** finds predictors that are simultaneously optimal in every environment.
- **Relevance to us:** low. It was not better than ERM in DomainBed [P3].

**P19. Li, Yang, Song & Hospedales (2018). Learning to Generalize: Meta-Learning for Domain Generalization (MLDG). AAAI 32. doi:10.1609/aaai.v32i1.11596.** [READ: abstract]
- **Method:** simulates train/test domain shifts inside each batch.
- **Relevance to us:** low. It was reported to perform poorly on HAR [P12].

---

## 3. Diagnosis of our specific failure

**Q1. Why did IE lower the nonlinear probe but not the linear one?**
- [CODE] The IE target for the person head is `softmax(W_d · relu(z) + b_d) = uniform`. Softmax ignores constant shifts, so this holds exactly when `W̃_d · relu(z_i)` is constant, where W̃_d is W_d with its row mean removed (rank ≤ 3 for K = 4).
- So IE constrains **at most 3 directions of relu(z)** and leaves the other ~47 dimensions of z free.
- The ReLU also hides any information z stores as negative values.
- W_d is fitted to z_d, not to z, so those 3 directions are not where person information in z actually sits.
- [MEASURED] The IE term quickly fell close to its minimum (pilot: per-sample 0.068 → 0.008), while the linear probe stayed at 52–53%.
- [HYPOTHESIS] Keeping a nonlinear function (the ReLU pattern) constant along 3 directions removes some of the nonlinear structure that the MLP probe uses, which explains its −3.8 drop. A fresh linear probe can still use the free directions.

**Q2. Why didn't the reduction improve recognition?**
- [MEASURED] IE increased source-validation accuracy (+1.5) but not target accuracy (−1.0). Its effect was to reshape z in a way that helps the 4 training people.
- [HYPOTHESIS] The person information removed lay in directions the activity head does not rely on, so removing it cannot change the decision for a new person. What harms a new person is person variation *inside* each activity cluster, along directions the activity head *does* use. IE has no term aimed at that.
- [MEASURED] Even an effect this size would be hard to see. The noise between identical procedures is about 4.8 points per run, and the target-level 95% interval is roughly ±5.7.

**Q3. Is z still encoding person information?**
- [MEASURED] Yes. Linear probe 52–53% and MLP probe 61–65%, against 27% chance, in every variant.

**Q4. Is the frozen classifier too weak?**
- [CODE + MEASURED] Yes, in two ways.
  - Structurally: a rank-3 constraint on directions learned for a different code (Q1).
  - In training: `DC_d` sees **single-person batches**. Each of its gradient steps pushes all outputs toward one class, so it is a poorly conditioned person classifier.
- [MEASURED, pilot] The frozen-head "cross accuracy" for person ← z was already near chance in *all* variants, including released, so it was blind to leakage that the probes detect.

**Q5. Is the current IE objective fundamentally limited?**
- Yes, for any *fixed* head: it can only constrain the head's row space.
- The remedy is an *adaptive* critic (an adversary, P4/P5) or an objective defined on the *whole geometry* of z (alignment or contrastive, P9–P12).
- Even adaptive adversaries leave leakage behind [P7].

**Q6. Could the architecture, channel mixing or person prior be limiting results?**
- Person prior [CODE]: it is identical for all people during training. The cause is the single-person batching (a mixed batch gives person-specific priors, `test_variants.py` test 6). By DIVA's logic [P2], GILE's main built-in separation route is therefore switched off. Its effect on accuracy has not been measured.
- Channel mixing [CODE]: the CNN's temporal kernels see a fixed interleaving of all 9 sensors across about 14 time steps, not one sensor over time.
  - [HYPOTHESIS] This plausibly limits accuracy for *all* variants, possibly substantially.
  - Fixing it changes every arm, so the 75-run baseline would have to be re-run (about 2 h 10 min). It does not address separation.
- Single-person, person-sequential batches [CODE]: there are only about 3–4 steps per person before switching, so forgetting within an epoch is probably mild [HYPOTHESIS]. The structural consequences (prior, DC_d, no cross-person terms possible) are certain.

**Q7. Are we removing information the activity classifier needs?**
- [MEASURED] IE also lowered activity ← z_d (−2.6), and target accuracy fell for 3 of 5 people. That is *compatible* with useful information being distorted, but it doesn't prove it.
- [HYPOTHESIS] In UCI, how a person holds the phone is person-specific, yet the gravity direction is essential for telling sitting, standing and lying apart. Removing "person" information without conditioning on activity risks removing posture cues.

**Q8. Do differing activity distributions make full invariance undesirable?**
- [MEASURED] UCI label shift is modest: activity shares range 13.5–27.4%, and only person 0 is clearly different (27.4% walking).
- [REPORTED, P6] Under label shift, marginal invariance provably raises a lower bound on error.
- Conclusion: prefer **class-conditional** invariance (P5, P9, P10). It is never worse in principle and safer for person 0.

**Supported by measurements:**
- the 15-point gap between training people and the new person;
- IE improves fit to the training people but not to the new person;
- person information remains in z;
- the frozen head misses that leakage;
- label shift is small.

**Supported by code reading:**
- the person prior collapses;
- batches contain one person;
- channels are mixed;
- IE constrains at most 3 directions.

**Hypotheses only:**
- the effect of channel mixing on accuracy;
- forgetting within an epoch;
- posture cues being removed;
- why the MLP probe dropped.

---

## 4. Candidate improvements

Unless stated otherwise, each candidate is built on `ie_off` (single Adam at lr 1.1e-3), so its effect is isolated. B = person-balanced minibatches (C1), which every cross-person loss requires.

### C1. Person-balanced minibatches (B)
- **Motivation:**
  - The verified defect in §1.
  - DomainBed's per-domain minibatches [P3].
  - DIVA's need for a working conditional prior [P2].
- **Mechanism:** each step draws 64/K windows from each of the K = 4 source people. The number of steps per epoch is kept equal to today's S = Σ_p ⌊n_p/64⌋, so compute and data exposure stay the same.
- **Plain English:** every update sees all four people at once, instead of four person-by-person blocks.
- **What changes:** the data loader only. The model and losses stay the same.
- **Targets:** both, *indirectly*. It restores the person prior and the person classifier, which makes cross-person losses possible.
- **Benefits:** fixes a structural flaw. Very cheap. Low risk.
- **Drawbacks:** no direct pressure on z. The accuracy effect is uncertain and could be small.
- **Complexity:** low (~30 lines). Stability: high. GPU: same as now.
- **Evaluation:** `ie_off_bal` against `ie_off` (baseline). Same protocol.

### C2. XPC-B: cross-person supervised contrastive loss on z, with B
- **Motivation:**
  - SupCon [P9].
  - Same-class DG regularisation [P10].
  - SupCon in HAR (DDLearn) [P13].
  - Class-conditional invariance [P5, P6].
  - Our Q1, Q2 and Q8.
- **Maths:** see §6. The loss pulls together same-activity windows from *different* people and pushes away other activities.
- **Plain English:** "walking by person A" and "walking by person B" must land next to each other in z.
- **What changes:** the loader (B); a small projection head g (training only); one loss term on qzy's posterior mean. GILE's encoders, decoder, classifiers, priors and ELBO stay the same.
- **Targets:** both, *directly*. It removes person variation inside each activity, which is what decides recognition of a new person.
- **Benefits:**
  - acts on the whole geometry, not 3 directions;
  - keeps activity information;
  - no adversarial game.
- **Drawbacks:**
  - Invariance across only 4 people may not extend to a 5th.
  - Needs a weight λ.
  - Adjacent windows overlap by 50% (handled in the design: same-person, same-activity windows are excluded from the denominator).
- **Complexity:** low to medium (~80 lines). Stability: medium-high. GPU: about +5–10% [HYPOTHESIS].
- **Evaluation:** `xpc_bal` against `ie_off_bal`, which isolates the contrastive term.

### C3. Person-style sensor augmentation
- **Motivation:** Um et al. [P15], DDLearn (augmentation alone +9.21) [P13], Mixup strong in DIVERSIFY [P14], CrossHAR [P17].
- **Mechanism:** x' = T(x) with random per-window scaling s ~ U(0.8, 1.2) per sensor triad, a small 3-D rotation R(θ ≤ 15°) applied identically to the triads of the same device, and time-warping. Labels y stay the same.
- **Plain English:** each training window is made to look as if a slightly different person, or a differently held phone, recorded it. The model effectively sees more than 4 people.
- **What changes:** the input pipeline only.
- **Targets:** mainly accuracy. Separation only indirectly.
- **Benefits:** strongest published HAR accuracy evidence. Cheap. Tackles "only 4 training people".
- **Drawbacks:**
  - Semantic conflict in GILE: should an augmented window keep its person label d? If yes, z_d is taught that "style is not identity", weakening z_d. If no, it is unclear what d should be.
  - Rotation can damage the gravity cues that separate postures.
  - Requires the channel layout to be known exactly, and it is currently scrambled in the encoder. The augmentation itself works on raw (B, 128, 9) data, so it is still possible.
- **Complexity:** low to medium. Stability: high. GPU: small.
- **Evaluation:** `aug` against `ie_off`. Possible with or without B.

### C4. Class-conditional adversary on z (CDANN-style, replacing frozen-head IE), with B
- **Motivation:** DANN [P4], CDANN [P5], our Q4 and Q5.
- **Maths:** a person discriminator h_φ(d | z, y) (an MLP), trained by min_φ CE; the encoder trained by min_θ L_GILE − λ·CE through gradient reversal.
- **Plain English:** a person-guessing network keeps retraining on z, and the encoder keeps trying to fool it.
- **What changes:** IE is replaced by a trained adversary.
- **Targets:** mainly separation. Uncertain effect on accuracy.
- **Benefits:** closes IE's 3-direction loophole.
- **Drawbacks:**
  - Leakage persists [P7].
  - Min-max instability, which matters with our already large 4.8-point noise.
  - With 4 people the discriminator can memorise them.
  - DomainBed shows no reliable gain [P3].
- **Complexity:** medium. Stability: **low**. GPU: about +10–20%.
- **Evaluation:** `adv_bal` against `ie_off_bal`.

### C5. Class-conditional MMD or CORAL alignment across people, with B
- **Motivation:** AFFAR's pairwise MMD [P12], CORAL [P11], class-conditional reasoning [P5, P6].
- **Maths:** Σ_c Σ_{p<q} MMD²(z | y = c, d = p ; z | y = c, d = q).
- **Plain English:** within each activity, make the cloud of z values for every pair of people match.
- **Targets:** both.
- **Drawbacks:** with B, each (person, activity) cell has only about 2–3 windows per batch, so per-class MMD is very noisy. Using the marginal (all activities pooled) is less noisy but brings in the label-shift problem.
- **Complexity:** low to medium. Stability: medium-high. GPU: small.
- **Evaluation:** `mmd_bal` against `ie_off_bal`.

### C6. Channel-layout fix
- **Motivation:** the verified bug, and CNN locality.
- **Mechanism:** replace `x.view(-1, C, 1, T)` with `x.permute(0, 2, 3, 1)`, which lays the data out properly by channel and time.
- **Targets:** accuracy only. Possibly large but unmeasured [HYPOTHESIS].
- **Drawbacks:**
  - Changes every arm, so the 75-run baseline must be re-run (about 2 h 10 min).
  - Changes the encoder's output shape, so it may need layer-size changes. That counts as an architecture change, which needs your approval.
- **Complexity:** low code, high re-run cost.
- **Evaluation:** a full re-baseline. **Recommended as documented future work or a limitation**, not tonight.

### C7. (brief) HSIC(z, z_d), IRM, MLDG, self-supervised pretraining
- **HSIC:** enforces z ⊥ z_d, not z ⊥ person. It is weakly linked to the goal.
- **IRM, MLDG:** no reliable gain in DomainBed or on HAR [P3, P12].
- **Self-supervised pretraining:** too costly for the deadline.
- Not pursued.

---

## 5. Comparison and ranking

Scores run from 1 (worst) to 5 (best). For complexity, GPU cost and risk, 5 = cheapest or safest. These are my judgement, informed by the evidence above.

| Rank | Candidate | Scientific justification | Prior evidence | Fits GILE | Accuracy potential | Removes person info from z | Complexity (5 = easy) | GPU (5 = cheap) | Stability (5 = safe) | Fits deadline | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **C2 XPC-B** | 5 | 3 | 4 | 3 | 4 | 4 | 4 | 4 | 4 | **35** |
| 2 | C3 augmentation | 4 | 4 | 3 | 4 | 2 | 4 | 4 | 4 | 4 | 33 |
| 3 | C1 balanced batches alone | 4 | 3 | 5 | 2 | 2 | 5 | 5 | 5 | 5 | 36* |
| 4 | C5 class-conditional MMD | 4 | 3 | 4 | 3 | 3 | 4 | 4 | 4 | 3 | 32 |
| 5 | C6 channel fix | 4 | 2 | 5 | 3–4 | 1 | 5 | 2 | 5 | 2 | 29 |
| 6 | C4 class-conditional adversary | 4 | 2 | 4 | 2 | 4 | 3 | 3 | 2 | 2 | 26 |
| 7 | HSIC / IRM / MLDG | 2 | 1–2 | 3 | 2 | 2 | 4 | 4 | 3 | 3 | ~23 |

\* C1 has the highest raw total because it is cheap and safe, but it **targets neither objective directly**. It is a precondition and a control, not the improvement. It ranks 3rd on "potential to address both objectives", which is the criterion you set.

**Why XPC-B beats augmentation for *this* project:**
- Augmentation has better published accuracy evidence. But our measured problem is specific: person information persists in z (linearly too), and the separation we achieved did not reach the decision.
- XPC is the only method whose separation is *defined* on the decision-relevant structure, so any separation it achieves should show up in recognition.
- Augmentation conflicts with GILE's person code z_d and has no direct separation mechanism.
- Augmentation remains the best follow-up, and stacks naturally on XPC-B (DDLearn uses both).

**Why not the adversary:** the literature [P3, P7] and our noise level both predict an unstable, leaky result with no guaranteed accuracy gain.

---

## 6. Recommended approach: XPC-B

### 6.1 Hypotheses (fixed before any run)
- **H1 (accuracy, primary).** XPC raises unseen-person weighted-F1 over the matched control: `xpc_bal` > `ie_off_bal`.
- **H2 (separation, mechanism).** XPC lowers person ← z decodability under **both** the linear and the MLP probe: `xpc_bal` < `ie_off_bal`.
- **H3 (batching, descriptive).** Person-balanced batches change the results relative to `ie_off`. No direction is predicted, and it is reported descriptively.
- **Guardrail.** Lower probes with lower accuracy is a **failure**, not a success.

### 6.2 Formulation
Notation:
- μ_i = posterior mean of qzy(x_i), the activity code (dimension 50 = d_AE).
- g = projection head, Linear(50, 64) → ReLU → Linear(64, 32), used only in training.
- u_i = g(μ_i) / ‖g(μ_i)‖.

```
Batch (B):  64 windows = 16 from each of the K=4 source persons, sampled without replacement
            per person (reshuffled when exhausted); steps/epoch = S = Σ_p floor(n_p/64)  (unchanged).

P(i) = { j ≠ i : y_j = y_i ,  d_j ≠ d_i }                 # same activity, different person
A(i) = { j ≠ i : not (y_j = y_i and d_j = d_i) }          # excludes same-person-same-activity
                                                          # (50%-overlap near-duplicates)
L_XPC = (1/|I+|) Σ_{i∈I+}  −(1/|P(i)|) Σ_{p∈P(i)} log  exp(u_i·u_p/τ) / Σ_{a∈A(i)} exp(u_i·u_a/τ)
        I+ = anchors with |P(i)| ≥ 1  (expected |P(i)| ≈ 48 × 1/6 ≈ 8)

Total:  L = L_GILE_main(released loss_function, unchanged) + λ · L_XPC        (no IE term)
```

Settings, all fixed a priori and never tuned on any target:
- τ = 0.1. This is a commonly used SupCon temperature [P9, P16]. Check the exact value against those papers.
- **λ rule:**
  - At initialisation (seed 0), on the **source data of target 0 only**, measure the mean main loss per batch and the mean L_XPC per batch on the CPU.
  - Set λ = 0.1 × (main / L_XPC), so that XPC is about 10% of the loss at the start.
  - That λ is then frozen for all targets and seeds.
  - The main loss is summed over the batch, so a fixed λ = 1 would be negligible. The rule avoids that without looking at any result.
- The gradient reaches qzy and g only. The classifiers, decoder and priors are unchanged.

### 6.3 What changes and what stays
| Changes | Unchanged |
|---|---|
| Source batching (person-balanced, same number of steps) | Encoders, decoder, DC_y and DC_d, priors, ELBO terms and weights |
| + projection head g (dropped after training) | lr 1.1e-3, single Adam, 150 epochs, batch 64 |
| + λ·L_XPC | Data split, overlap purge, epoch selection on source validation, probes, metrics |
| | Released code and all existing runs (new output folder `runs_xpc/`) |
| | Channel mixing (kept, documented as a limitation) |

### 6.4 Arms, seeds and cost
- **New arms:**
  - `ie_off_bal`: balanced batches, no extra loss. This is the control.
  - `xpc_bal`: balanced batches plus XPC.
- **Reused baseline arms (read-only, from `runs_full/`):** `released`, `ie_off`, `ie_on`.
- **Optional third arm, with your approval:** `ie_on_bal`. It tests the diagnosis that IE failed partly because of single-person batches.
- **Size:** 5 targets × 5 seeds (0–4, paired with the baseline) × 2 arms = **50 runs**.
- **Runtime:** at about 100–110 s per run → **about 90–100 min on one T4**, plus 1 anchor run: `ie_off` target 0 / seed 0, which must reproduce the stored metrics and so guards against a changed Colab version.
- **Reduced fallback if time is short:** seeds 0–2 only, 30 runs, about 55 min. That leaves wider intervals.

### 6.5 Metrics and statistics
- **Primary:** target accuracy, macro-F1 and weighted-F1 at the epoch selected on source validation.
- **Secondary (descriptive):** mean of the last 10 epochs, and the last epoch.
- **Diagnostic only:** the leaky best-on-target score.
- **Probes:** person ← z (linear and MLP), activity ← z_d (linear and MLP), the same as in all 75 runs.
- **Statistics:** paired differences per (target, seed), averaged within each target, then a 95% t-interval over the 5 targets (df = 4), plus the number of targets that improve.
- **Main comparisons:** `xpc_bal − ie_off_bal` (H1, H2) and `ie_off_bal − ie_off` (H3). `released` and `ie_on` are context.
- **Decision rules (fixed in advance):**
  - **H1 supported:** mean weighted-F1 difference > 0, at least 4 of 5 targets positive, and the CI lower bound above −1.
  - **H1 harmful:** at least 4 of 5 targets negative.
  - Otherwise **inconclusive**.
  - **H2 supported:** both person ← z probes lower in at least 4 of 5 targets.
  - **Detection limit:** with the 4.8-point noise level, effects under about 3–4 points are unlikely to be detected. An inconclusive result is a legitimate finding.
- **No target data** is used for λ, τ, epochs or arm choice.

### 6.6 Implementation and verification (only after approval; CPU first)
1. Add `--variant {ie_off_bal, xpc_bal}` as **new** code paths in `run_experiment.py` and a new `xpc_term` method. Existing variants must stay byte-identical, re-verified with `checks/test_variants.py`.
2. New tests:
   - every batch contains 16 windows from each source person;
   - the number of steps per epoch equals today's;
   - positives are always from a different person with the same activity;
   - same-person, same-activity windows are excluded from A(i);
   - anchors with no positives are skipped;
   - the gradient reaches qzy and g only;
   - the prior is person-specific in balanced batches;
   - λ is computed by the rule and logged in `config.json`.
3. A 1-epoch CPU smoke test on real data, then a Colab notebook built like `GILE_IE_full.ipynb` (embedded code with SHA-256 checks, output to `runs_xpc/`).

### 6.7 Failure modes
- **λ too large:** z collapses into 6 tight clusters, which conflicts with the KL/ELBO terms. Watch for a falling ELBO share or overconfident source-validation predictions.
- **λ too small:** no measurable effect. The pre-set 10% rule makes this unlikely but possible.
- **Batching alone shifts everything:** `ie_off_bal` may differ a lot from `ie_off`. This is why the control arm exists.
- **Only 4 people:** alignment across 4 people may not cover a 5th person's style. This is exactly what augmentation (C3) would address next.
- **Overlap leakage within a batch:** near-duplicate windows of the same person are excluded from the denominator, but cross-person positives are never duplicates.
- **Probes share validation windows with epoch selection:** already true for all variants, so it doesn't bias the comparison.
- **Channel mixing** caps what any variant can reach. It is reported as a limitation.

---

## 7. Fallbacks
- If you prefer the method with the strongest published accuracy evidence, use **C3 augmentation** with the same two-arm design (`ie_off` against `aug`, or `ie_off_bal` against `aug_bal`).
- XPC-B and augmentation combine naturally later (augmented views give extra cross-style positives), but that would be a second change.

---

## 8. References
- [P1] Qian H., Pan S. J., Miao C. Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition. AAAI 35(13):11921–11929, 2021. https://doi.org/10.1609/aaai.v35i13.17416
- [P2] Ilse M., Tomczak J. M., Louizos C., Welling M. DIVA: Domain Invariant Variational Autoencoders. MIDL, PMLR 121:322–348, 2020. https://proceedings.mlr.press/v121/ilse20a.html
- [P3] Gulrajani I., Lopez-Paz D. In Search of Lost Domain Generalization. ICLR 2021. https://arxiv.org/abs/2007.01434 ; code https://github.com/facebookresearch/DomainBed
- [P4] Ganin Y. et al. Domain-Adversarial Training of Neural Networks. JMLR 17(59):1–35, 2016. https://www.jmlr.org/papers/v17/15-239.html
- [P5] Li Y., Tian X., Gong M., Liu Y., Liu T., Zhang K., Tao D. Deep Domain Generalization via Conditional Invariant Adversarial Networks. ECCV 2018, pp. 624–639. https://doi.org/10.1007/978-3-030-01267-0_38
- [P6] Zhao H., Tachet des Combes R., Zhang K., Gordon G. On Learning Invariant Representations for Domain Adaptation. ICML, PMLR 97:7523–7532, 2019. https://proceedings.mlr.press/v97/zhao19a.html
- [P7] Elazar Y., Goldberg Y. Adversarial Removal of Demographic Attributes from Text Data. EMNLP 2018, pp. 11–21. https://aclanthology.org/D18-1002/
- [P8] Locatello F. et al. Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. ICML, PMLR 97:4114–4124, 2019. https://proceedings.mlr.press/v97/locatello19a.html
- [P9] Khosla P. et al. Supervised Contrastive Learning. NeurIPS 2020. https://arxiv.org/abs/2004.11362
- [P10] Kim D., Yoo Y., Park S., Kim J., Lee J. SelfReg: Self-supervised Contrastive Regularization for Domain Generalization. ICCV 2021, pp. 9619–9628. https://doi.org/10.1109/ICCV48922.2021.00948
- [P11] Sun B., Saenko K. Deep CORAL: Correlation Alignment for Deep Domain Adaptation. arXiv:1607.01719, 2016 (ECCV 2016 workshops; venue to verify). https://arxiv.org/abs/1607.01719
- [P12] Qin X., Wang J., Chen Y., Lu W., Jiang X. Domain Generalization for Activity Recognition via Adaptive Feature Fusion. ACM TIST, 2022. https://arxiv.org/abs/2207.11221 (journal DOI reported as 10.1145/3552434; verify)
- [P13] Qin X. et al. Generalizable Low-Resource Activity Recognition with Diverse and Discriminative Representation Learning. KDD 2023. https://doi.org/10.1145/3580305.3599360
- [P14] Lu W., Wang J., Sun X., Chen Y., Xie X. Out-of-Distribution Representation Learning for Time Series Classification. ICLR 2023. https://arxiv.org/abs/2209.07027
- [P15] Um T. T. et al. Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks. ICMI 2017, pp. 216–220. https://doi.org/10.1145/3136755.3136817
- [P16] Qian H., Tian T., Miao C. What Makes Good Contrastive Learning on Small-Scale Wearable-based Tasks? KDD 2022, pp. 3761–3771. https://arxiv.org/abs/2202.05998
- [P17] Hong Z. et al. CrossHAR: Generalizing Cross-dataset Human Activity Recognition via Hierarchical Self-Supervised Pretraining. Proc. ACM IMWUT, 2024. https://doi.org/10.1145/3659597 (author list to verify)
- [P18] Arjovsky M., Bottou L., Gulrajani I., Lopez-Paz D. Invariant Risk Minimization. arXiv:1907.02893, 2019.
- [P19] Li D., Yang Y., Song Y.-Z., Hospedales T. Learning to Generalize: Meta-Learning for Domain Generalization. AAAI 32, 2018. https://doi.org/10.1609/aaai.v32i1.11596

**Verification note:** all bibliographic details were checked against publisher, arXiv, DBLP or proceedings pages during this session, except where marked "verify". Reported numbers come from my reading of the papers, as labelled in §2.2. Check them against the PDFs before they appear in the report.
