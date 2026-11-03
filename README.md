# From Pixels to Language

Final project for CSE485/CSE488 Deep Learning.

We train a CNN encoder on the CUB-200-2011 bird dataset and reuse it for three tasks: species classification, image colorization and image captioning. This repository currently contains the first milestone (classification).

Data: [Multimodal-Fatima/CUB_train](https://huggingface.co/datasets/Multimodal-Fatima/CUB_train) and [Multimodal-Fatima/CUB_test](https://huggingface.co/datasets/Multimodal-Fatima/CUB_test) (200 species, ~5.9k train / ~5.8k test images, 10 captions per image).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

The dataset is downloaded from Hugging Face the first time it is used.

## Usage

Everything runs through one command line entry point, `python -m src.cli`. Experiment settings live in `configs/`, and any value can be overridden with `--set section.key=value`.

Data preparation:

```bash
python -m src.cli data --action split --config configs/base.yaml
python -m src.cli data --action eda --config configs/base.yaml
python -m src.cli data --action augment-preview --config configs/cls_resnet50.yaml
```

Training:

```bash
python -m src.cli train --task classification --config configs/cls_scratch.yaml
python -m src.cli train --task classification --config configs/cls_resnet50_probe.yaml
python -m src.cli train --task classification --config configs/cls_resnet50.yaml
```

Each run is saved in `outputs/runs/<name>/` (checkpoints, learning curves, metrics). An interrupted run can be resumed with `--resume outputs/runs/<name>/last.pt`.

Evaluation and inference:

```bash
python -m src.cli evaluate --task classification --checkpoint outputs/runs/cls-resnet50-ft/best.pt --split test
python -m src.cli compare --task classification --runs outputs/runs/cls-scratch outputs/runs/cls-resnet50-ft --split test
python -m src.cli predict --task classification --checkpoint outputs/runs/cls-resnet50-ft/best.pt --image bird.jpg
python -m src.cli gradcam --task classification --checkpoint outputs/runs/cls-resnet50-ft/best.pt outputs/runs/cls-scratch/best.pt --split test
```

Hyperparameter search:

```bash
python -m src.cli tune --task classification --config configs/tune_grid.yaml
python -m src.cli tune --task classification --config configs/tune_random.yaml
python -m src.cli tune --task classification --compare outputs/tuning/tune-grid outputs/tuning/tune-random
```

## Training on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Moazdarwishh/DeepLearning-Project/blob/main/notebooks/colab_train.ipynb)

We trained on Colab T4 GPUs. `notebooks/colab_train.ipynb` clones the repo, saves outputs to Google Drive and runs the same commands as above. Select a GPU runtime before running it.

## Experiment tracking

Runs are logged with MLflow to `mlflow.db`. To browse them:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Project structure

```
configs/        experiment configs
data/splits/    train/validation split and excluded images
notebooks/      Colab notebook
src/
  data/         dataset loading, EDA, caption cleaning, transforms
  models/       encoders (scratch CNN, ResNet) and classifier
  training/     trainer, optimizer, hyperparameter search
  evaluation/   metrics and result analysis
  pipelines/    train / evaluate / predict commands
  visualization/ plots and Grad-CAM
  cli.py
tests/
```

## Results

Results on the official CUB-200-2011 test split (5,794 images, 224 px). Each model was picked by its best validation epoch, and the test set was evaluated once.

| Model | Trainable params | Test accuracy | Top-5 | Macro F1 |
|---|---|---|---|---|
| Scratch CNN | 11.3M | 50.4% | 78.2% | 0.500 |
| ResNet-50 (linear probe) | 0.41M | 69.8% | 91.7% | 0.699 |
| ResNet-50 (fine-tuned) | 23.9M | **83.1%** | **95.6%** | **0.830** |

For comparison, Barz and Denzler (WACV 2020) report 82.5% for a fine-tuned ResNet-50 and 55.5% for one trained from scratch with label smoothing, both at 448 px.

Hyperparameter search on the linear probe (8 trials × 5 epochs each, same budget for both):

| Search | Best val accuracy | Mean | Trials to best |
|---|---|---|---|
| Grid | 65.1% | 59.6% | 5 |
| Random | 64.4% | 61.5% | 3 |

## Team

| Name | ID | GitHub | Contributions |
|---|---|---|---|
| Moaz Mohamed Zakaria | 22P0307 | [Moazdarwishh](https://github.com/Moazdarwishh) | Project setup (repo structure, configs, CLI, Colab notebook); data understanding and preparation (train/val split, EDA, duplicate and grayscale checks, caption cleaning, augmentation, data loaders); scratch CNN; grid search; Grad-CAM |
| Ahmed Abbady | 22P0308 | [AhmmedAbbady](https://github.com/AhmmedAbbady) | Model development and training (scratch CNN, ResNet-50 linear probe and fine-tuning, training loop, evaluation metrics); Grad-CAM; report |
| Anas Mansour | 22U0005 | [anasmansourr](https://github.com/anasmansourr) | Model development and training (scratch CNN, ResNet-50 linear probe and fine-tuning, training loop, evaluation metrics); Grad-CAM; report |

Trained models and outputs: [Google Drive](https://drive.google.com/drive/folders/1Gx7MUBF5f2cXeDkD_xpO2NJaKUlT5Hx-?usp=sharing)
