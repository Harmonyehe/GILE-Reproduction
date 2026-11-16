# GILE Source Reference

This directory retains source material used by the reproduction. The method was introduced by Qian, Pan and Miao in the AAAI 2021 paper *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*.

The original implementation supports UCI HAR, Opportunity and UniMiB-SHAR. Dataset preprocessing scripts expect the prepared files in dataset-specific `data` directories. The original implementation details are in `Original Code/cross-person-HAR/README.md`.

All model-training experiments in this reproduction were run in Google Colab with GPU acceleration. Run 1 was interrupted and used 150 epochs for UCI HAR, so it was excluded. The final result set uses the corrected 100-epoch UCI HAR experiments and the completed 100-epoch Run 2 Opportunity and UniMiB-SHAR experiments.

Use `SETUP.md` for the final Colab workflow, `ANALYSIS_GUIDE.md` for interpreting the analysis outputs, and the main final reproduction README for results.

## Reference

Qian, H., Pan, S. J., and Miao, C. (2021). *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition*. Proceedings of the AAAI Conference on Artificial Intelligence, 11921–11929.