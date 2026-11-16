# GILE Source Reference

This directory contains the original GILE source material retained alongside the reproduction code. The method was introduced by Qian, Pan and Miao in the AAAI 2021 paper:

**Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition.**

The original implementation covers UCI HAR, Opportunity and UniMiB-SHAR. Its dataset preparation scripts expect the processed data in the relevant `data` directory. The original environment and command examples are documented in `Original Code/cross-person-HAR/README.md`.

For the completed reproduction, use the scripts and configuration under `Reproduction/Final Reproduction/Code/`. The final experimental configuration uses 100 epochs for all three datasets. The final results combine the Run 2 Opportunity and UniMiB-SHAR experiments with the UCI HAR experiments conducted using the corrected 100-epoch configuration.

The consolidated metrics, evaluation notes and analysis artefacts are documented in `Reproduction/Final Reproduction/README.md`.

## Reference

Qian, H., Pan, S. J., and Miao, C. (2021). *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*. Proceedings of the AAAI Conference on Artificial Intelligence, 11921–11929.
