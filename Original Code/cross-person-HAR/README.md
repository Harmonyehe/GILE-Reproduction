# GILE — Original Implementation

This directory contains the source implementation associated with the AAAI 2021 paper *Latent Independent Excitation for Generalizable Sensor-based Cross-Person Activity Recognition* by Hangwei Qian, Sinno Jialin Pan and Chunyan Miao.

The implementation provides dataset-specific preprocessing, model and training scripts for UCI HAR, Opportunity and UniMiB-SHAR.

## Original environment

The original code documents a Python 3.6 environment and an older PyTorch stack. The commands below reproduce that historical environment specification; they are not a guarantee that the same versions are required or compatible in a current Colab runtime.

```bash
conda create --name gile python=3.6
conda activate gile
conda install -c pytorch pytorch=1.3
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm
conda install -c pytorch torchvision
```

## Datasets

The original project uses UCI HAR, Opportunity and UniMiB-SHAR. Download and preprocess each dataset using its official distribution and the scripts supplied with the implementation. The scripts expect preprocessed files in the relevant `data` subdirectories. See the reproduction `SETUP.md` for the directory layout used by the final project.

## Original command interface

The original dataset-specific entry points accept a target participant argument. Examples:

```bash
python main_ucihar.py --beta_d 5.0 --beta_y 5.0 --target_domain 0
python main_oppor.py --target_domain S1
python main_shar.py --target_domain 3
```

These are examples of the original interface, not a complete reproduction workflow. For the final experiment configuration and the 100-epoch commands, use `Reproduction/Final Reproduction/SETUP.md`.

## Relationship to the reproduction

All model-training experiments in the reproduction were run in Google Colab with GPU acceleration. Run 1 was excluded because it was interrupted and its UCI HAR run used 150 epochs. The final result set uses the corrected 100-epoch UCI HAR experiments plus the completed 100-epoch Run 2 Opportunity and UniMiB-SHAR experiments.

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