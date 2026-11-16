# GILE Reproduction — Setup and Execution Guide

## 1. Overview

This guide explains how to prepare and execute the GILE reproduction.
The completed reproduction contains 13 target experiments:

| Dataset | Targets | Epochs |
|---|---|---:|
| UCI HAR | 0, 1, 2, 3, 4 | 100 |
| Opportunity | S1, S2, S3, S4 | 100 |
| UniMiB-SHAR | 1, 2, 3, 5 | 100 |

**Important:** The commands below start new training or evaluation runs.
They are not required to inspect the completed reproduction.

## 2. Repository structure

The final reproduction code is located in:

`Reproduction/Final Reproduction/Code/`

Relevant directories:
- `Training/` — training entry points, models and utilities.
- `Preprocessing/` — dataset preprocessing modules.
- `Evaluation/` — evaluation scripts.

Training entry points:
- `main_ucihar.py`
- `main_oppor.py`
- `main_shar.py`

Preprocessing modules are located under `Code/Preprocessing/`.

## 3. Google Colab environment

Select Runtime → Change runtime type → GPU, if available.

Check GPU availability:

```python
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
```

The training scripts select `cuda:0` when CUDA is available and
otherwise fall back to CPU.

## 4. Dependencies

Install missing supporting packages if required:

```bash
pip install numpy scipy pandas scikit-learn matplotlib tqdm
```

Avoid unnecessarily reinstalling PyTorch.

## 5. Dataset preparation

Raw datasets are not included in this repository.
The expected dataset directories, relative to the Training directory, are:

| Dataset | Expected directory |
|---|---|
| UCI HAR | `data/ucihar/` |
| Opportunity | `data/oppor/OpportunityUCIDataset/dataset/` |
| UniMiB-SHAR | `data/UniMiB-SHAR/data/` |

For UniMiB-SHAR, the preprocessing module reads `acc_data.mat`
and `acc_labels.mat` from its data directory.

Place the dataset directories under:

`Code/Training/data/`

Preserve the original dataset file names and directory structures.

## 6. Training configuration

Move into the training directory:

```bash
cd "/content/GILE-Reproduction/Reproduction/Final Reproduction/Code/Training"
```

The training scripts accept `--target_domain` and `--n_epoch`.
The final experiments used 100 epochs.

### Checkpoint configuration

Each training entry point reads the `GILE_CHECKPOINT_DIR`
environment variable and assigns it to `args.eval_checkpoint_dir`.

The training implementation saves the best model as `best_model.pt`.
It does not create the checkpoint directory automatically.
Create the directory before starting each experiment.

**Use a different checkpoint directory for every target** to prevent
one experiment from overwriting another experiment's checkpoint.

### UCI HAR

```bash
cd "/content/GILE-Reproduction/Reproduction/Final Reproduction/Code/Training"
for target in 0 1 2 3 4; do
    CHECKPOINT_DIR="$PWD/checkpoints/ucihar/$target"
    mkdir -p "$CHECKPOINT_DIR"
    GILE_CHECKPOINT_DIR="$CHECKPOINT_DIR" python main_ucihar.py --target_domain "$target" --n_epoch 100
done
```

### Opportunity

```bash
cd "/content/GILE-Reproduction/Reproduction/Final Reproduction/Code/Training"
for target in S1 S2 S3 S4; do
    CHECKPOINT_DIR="$PWD/checkpoints/oppor/$target"
    mkdir -p "$CHECKPOINT_DIR"
    GILE_CHECKPOINT_DIR="$CHECKPOINT_DIR" python main_oppor.py --target_domain "$target" --n_epoch 100
done
```

### UniMiB-SHAR

```bash
cd "/content/GILE-Reproduction/Reproduction/Final Reproduction/Code/Training"
for target in 1 2 3 5; do
    CHECKPOINT_DIR="$PWD/checkpoints/shar/$target"
    mkdir -p "$CHECKPOINT_DIR"
    GILE_CHECKPOINT_DIR="$CHECKPOINT_DIR" python main_shar.py --target_domain "$target" --n_epoch 100
done
```

These commands start new training runs. Do not execute them merely
to inspect the existing results.

## 7. Output locations

The training utilities organise experiment results under:

`results/<dataset>/<model>/<target>/`

For example:
- `results/ucihar/GILE/0/`
- `results/oppor/GILE/S1/`
- `results/shar/GILE/1/`

The training utilities create these result directories automatically.

Checkpoints are saved separately under the directory specified by
`GILE_CHECKPOINT_DIR`, with the filename `best_model.pt`.

## 8. Evaluation

Evaluation uses saved checkpoints. It does not require training again.
The evaluation scripts require explicit checkpoint and output paths.

### UCI HAR

`evaluate_best.py` defaults to 150 epochs. Explicitly pass 100
when evaluating the corrected 100-epoch experiments.

```bash
cd "/content/GILE-Reproduction/Reproduction/Final Reproduction/Code/Evaluation"
python evaluate_best.py \
    --target_domain 0 \
    --n_epoch 100 \
    --checkpoint "/path/to/ucihar/0/best_model.pt" \
    --output "/path/to/evaluation/ucihar_target_0.json"
```

### Opportunity

```bash
python evaluate_oppor.py \
    --target_domain S1 \
    --checkpoint "/path/to/oppor/S1/best_model.pt" \
    --output "/path/to/evaluation/oppor_S1.json"
```

### UniMiB-SHAR

```bash
python evaluate_shar.py \
    --target_domain 1 \
    --checkpoint "/path/to/shar/1/best_model.pt" \
    --output "/path/to/evaluation/shar_target_1.json"
```

Replace example paths with the actual checkpoint and output locations.
Do not overwrite archived evaluation JSON files.

## 9. Final reproduction history

The final experiment set was assembled from the following executions:

- Run 1 was interrupted. Its UCI HAR execution used 150 epochs and
  is excluded from the final results.
- Run 2 Opportunity experiments completed 100 epochs for S1–S4.
- Run 2 UniMiB-SHAR experiments completed 100 epochs for targets
  1, 2, 3 and 5.
- Run 2 UCI HAR used 150 epochs and was superseded.
- UCI HAR targets 0–4 were subsequently rerun using 100 epochs.

The final result set contains five UCI HAR, four Opportunity and
four UniMiB-SHAR target experiments.

All model-training experiments were conducted in Google Colab
with GPU acceleration. Later evaluation, consolidation and
report-generation steps did not retrain the models.

## 10. Results and analysis

See `Reproduction/Final Reproduction/` for the completed results.

Important files:
- `FINAL_REPRODUCTION_REPORT.md` — final metrics and paper comparison.
- `ANALYSIS_GUIDE.md` — interpretation of metrics and analysis artefacts.
- `Analysis/Final Analysis Tables/` — consolidated evaluation tables.
- `Analysis/Training History Plots/` — recorded training histories.
- `Analysis/Fresh UCI HAR Confusion Matrices/` — fresh UCI HAR matrices.
- `Analysis/Audit/` — evaluation and methodological audit notes.

## 11. Reproducibility checklist

- Confirm that all required dataset files are present.
- Check that the working directory is `Code/Training`.
- Verify GPU availability if GPU training is intended.
- Record software and package versions.
- Create a separate checkpoint directory for each target.
- Set `GILE_CHECKPOINT_DIR` before launching training.
- Use the intended target IDs and 100-epoch configuration.
- Preserve existing experiment outputs and evaluation JSON files.
- Record checkpoint paths and evaluation provenance.
