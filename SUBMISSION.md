# CSE485 / CSE488 Deep Learning Project

## Milestone 1 submission: bird species classification on CUB-200-2011

## Team

| Name | ID | GitHub |
|---|---|---|
| Moaz Mohamed Zakaria | 22P0307 | [Moazdarwishh](https://github.com/Moazdarwishh) |
| Ahmed Abbady | 22P0308 | [AhmmedAbbady](https://github.com/AhmmedAbbady) |
| Anas Mansour | 22U0005 | [anasmansourr](https://github.com/anasmansourr) |

## Links

- **Codebase (GitHub):** https://github.com/Moazdarwishh/DeepLearning-Project
- **Trained models and outputs (Google Drive):** https://drive.google.com/drive/folders/1Gx7MUBF5f2cXeDkD_xpO2NJaKUlT5Hx-?usp=sharing
- **Report:** `docs/Milestone1_Report.pdf` in the repository in docs folder and in the code zip

## What is on Google Drive

- `outputs/runs/cls-scratch/`: CNN trained from scratch (`best.pt`, `last.pt`, learning curves, test evaluation)
- `outputs/runs/cls-resnet50-probe/`: ResNet-50 linear probe
- `outputs/runs/cls-resnet50-ft/`: fine-tuned ResNet-50, **final model**
- `outputs/tuning/`: grid search and random search (16 trials) with comparison figures
- `outputs/gradcam/`: Grad-CAM figures
- `mlflow.db`: MLflow tracking database for all runs

## Results (official test split, 5,794 images)

| Model | Test accuracy | Top-5 | Macro F1 |
|---|---|---|---|
| Scratch CNN | 50.4% | 78.2% | 0.500 |
| ResNet-50 linear probe | 69.8% | 91.7% | 0.699 |
| ResNet-50 fine-tuned | **83.1%** | **95.6%** | **0.830** |

## How to run

```bash
pip install -r requirements.txt
python -m src.cli evaluate --task classification --checkpoint outputs/runs/cls-resnet50-ft/best.pt --split test
python -m src.cli predict --task classification --checkpoint outputs/runs/cls-resnet50-ft/best.pt --image bird.jpg
```

To use the trained models, download `outputs/` from Google Drive into the project folder. Training on Colab: `notebooks/colab_train.ipynb`. More details are in `README.md`.
