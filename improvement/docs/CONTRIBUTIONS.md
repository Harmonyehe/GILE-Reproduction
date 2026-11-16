# Individual contributions (for the work-division page)

1. **Audited the released GILE code.**
   - Proved that the published Independent Excitation mechanism sends no gradient to any weight (0/53 parameter
     groups), so all published numbers are effectively "GILE without IE".
   - Showed that the released protocol selects the epoch on the held-out person, which inflates weighted-F1 by about 10
     points.
2. **Found and fixed a defect in GILE's encoder input layout.**
   - `x.view()` scrambles the sensor and time axes, so every convolution mixes 5 different sensors.
   - Confirmed it with tensor-level tests.
   - Implemented the correction as an isolated variant.
3. **Built a leakage-free evaluation protocol**, used for all improvement experiments:
   - source-only epoch selection;
   - overlap-purged source validation split;
   - linear and MLP person-identification probes with leak guards;
   - target-level paired confidence intervals;
   - predeclared success rules.
4. **Stage 1: implemented and evaluated a repaired, differentiable IE** (75 GPU runs). It reduced non-linear person
   information (−3.8) but not accuracy.
5. **Stage 2: reviewed the domain-generalisation and HAR literature** (19 papers). Designed, implemented and evaluated
   XPC-B, a cross-person supervised contrastive loss with person-balanced batches (50 runs + anchor). The predeclared
   verdict was inconclusive, and it is documented honestly.
6. **Stage 3: designed and ran a 2×2 factorial** (layout correction × person-style sensor augmentation; 45 runs + anchor
   on a T4). Layout correction + augmentation raised unseen-person accuracy from **79.2% to 91.6%**
   (+12.4, 95% CI +6.1 to +18.8, all 5 persons), with layout correction alone at +8.5 (+5.0 to +11.9, all 5
   persons). Both met the predeclared STRONG criterion. The combination also gave the largest reduction of person
   information in the activity code (MLP probe −6.6, all 5 persons).
   - Main analysis: seeds 0–2.
   - Seed 3: partial, excluded.
   - Seed 4: not run; the free Colab GPU quota was exhausted.
7. **Engineering for reproducibility:**
   - deterministic, resumable runners;
   - SHA-256-verified Colab notebooks and result packages;
   - a bit-exact reproducibility anchor across sessions and accounts;
   - test suites: 20 stage-3 checks plus stage-1/2 tests;
   - all results preserved with configs, histories, logs and checkpoints.
8. **Wrote the improvement documentation** that underlies report Section 5:
   - `FINAL_REPORT.md`;
   - `RESULTS_SUMMARY.md`;
   - `IMPLEMENTATION_NOTES.md`;
   - `RESULTS_REPORT.md`;
   - `XPCB_FINAL_REPORT.md`;
   - `LITERATURE_REVIEW_AND_PROPOSALS.md`.
9. **Prepared the GitHub-ready package** `improvement/`:
   - code;
   - per-run results (configs, histories, logs, metrics);
   - SHA-256 manifests;
   - analysis outputs;
   - setup and dataset instructions;
   - reproduction commands.
