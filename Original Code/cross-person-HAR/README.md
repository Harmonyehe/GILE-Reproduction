# GILE — Original Implementation

This directory contains the original implementation of Generalizable Independent Latent Excitation (GILE), introduced in:

**Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition**
Hangwei Qian, Sinno Jialin Pan and Chunyan Miao, AAAI 2021.

The implementation provides the model and dataset-specific scripts used for cross-person human activity recognition with UCI HAR, Opportunity and UniMiB-SHAR.

## Datasets

The original project uses three datasets:

- UCI HAR
- Opportunity
- UniMiB-SHAR

Download and preprocess the datasets using the instructions and scripts supplied with the original implementation. Place the processed data in the expected `./data` directory.

## Environment

The original implementation specifies a Python 3.6 environment and the following packages:

```bash
conda create --name gile python=3.6
conda activate gile
conda install -c pytorch pytorch=1.3
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm
conda install -c pytorch torchvision
```

These commands document the original environment. The reproduction may require compatibility adjustments for newer Python and library versions.

## Running the original scripts

Specify the target participant when launching a dataset-specific script:

```bash
python main_ucihar.py --beta_d 5.0 --beta_y 5.0 --target_domain 0
python main_oppor.py --target_domain S1
python main_shar.py --target_domain 3
```

The commands above illustrate the original script interface. For the completed reproduction, use the organised code and experiment documentation under `Reproduction/Final Reproduction/`.

## Relationship to the reproduction

The original implementation is retained as the source reference for the reproduction. The completed experimental configuration and consolidated results are documented in the repository root README and in `Reproduction/Final Reproduction/README.md`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/QianPM21,
  author    = {Hangwei Qian and Sinno Jialin Pan and Chunyan Miao},
  title     = {Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  pages     = {11921--11929},
  year      = {2021}
}
```
