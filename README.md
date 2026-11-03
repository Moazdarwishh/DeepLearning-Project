# From Pixels to Language
**CSE485 / CSE488 Deep Learning — Final Project**

One CNN encoder trained on CUB-200-2011 birds and reused for three tasks:

| Task | Input → Output | Milestone |
|------|----------------|-----------|
| Classification | image → 1 of 200 species | M1 |
| Colorization | grayscale → color | M2 |
| Captioning | image → sentence | M2 |

Dataset: [`Multimodal-Fatima/CUB_train`](https://huggingface.co/datasets/Multimodal-Fatima/CUB_train) and [`Multimodal-Fatima/CUB_test`](https://huggingface.co/datasets/Multimodal-Fatima/CUB_test).

## Setup

```bash
# 1. Create and activate a virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Check the setup
python -m src.cli info --config configs/cls_resnet50.yaml
pytest -q
```

On Colab / Kaggle:
```bash
!git clone <repo-url> && cd <repo> && pip install -r requirements.txt
```

## Project structure

```
configs/          YAML experiment configs (inherit configs/base.yaml via `_base_`)
notebooks/        EDA & experiments only — final code lives in src/
src/
  data/           dataset wrappers, stratified splits, caption cleaning, transforms
  models/         Encoder interface (multi-scale features), registry, encoders, heads
  training/       generic Trainer, callbacks, hyperparameter tuning
  evaluation/     metrics, benchmarking
  visualization/  EDA plots, learning curves, Grad-CAM
  utils/          config, seeding/device, experiment tracking (MLflow / W&B)
  cli.py          single entry point for every task
tests/            pytest unit tests
data/splits/      saved train/val split indices (committed — everyone uses the same split)
outputs/          checkpoints, figures, logs (git-ignored)
docs/             project plan and notes
```

## CLI

```bash
# data
python -m src.cli data --action inspect         --config configs/base.yaml
python -m src.cli data --action split           --config configs/base.yaml
python -m src.cli data --action eda             --config configs/base.yaml
python -m src.cli data --action augment-preview --config configs/cls_resnet50.yaml
python -m src.cli data --action loader-check    --config configs/cls_resnet50.yaml

# models
python -m src.cli info     --config configs/cls_resnet50.yaml
python -m src.cli train    --task classification --config configs/cls_scratch.yaml
python -m src.cli train    --task classification --config configs/cls_resnet50.yaml --set train.lr=3e-4 model.freeze_backbone=true
python -m src.cli evaluate --task classification --checkpoint outputs/cls-resnet50-ft/best.pt --split test
python -m src.cli predict  --task classification --checkpoint outputs/cls-resnet50-ft/best.pt --image bird.jpg --topk 5
python -m src.cli tune     --task classification --config configs/tune_grid.yaml
python -m src.cli tune     --task classification --config configs/tune_random.yaml
python -m src.cli gradcam  --task classification --checkpoint outputs/cls-resnet50-ft/best.pt
```

Any config value can be overridden with `--set section.key=value`.

## Experiment tracking

Set `tracking.backend` in a config to `mlflow`, `wandb` or `none`.

- **MLflow:** runs are stored in `./mlruns`. View them with `mlflow ui`, then open http://127.0.0.1:5000.
- **W&B:** run `wandb login` once and set `tracking.entity` to your team.

## Status

- [x] Project setup: structure, config system, tracking wrapper, encoder interface, CLI skeleton
- [x] M1 — EDA & data preparation (split, EDA figures, caption cleaning, augmentation)
- [ ] M1 — Scratch CNN baseline
- [ ] M1 — Transfer learning
- [ ] M1 — Grid vs random search
- [ ] M1 — Evaluation & Grad-CAM
- [ ] M2 — Colorization
- [ ] M2 — Captioning

## Team

| Member | GitHub | Main ownership |
|--------|--------|----------------|
|  |  |  |

## Links
- Trained models (Google Drive): _TBD_
- Report: _TBD_
