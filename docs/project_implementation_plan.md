# CSE 485 / CSE 488 — "From Pixels to Language"
## Implementation Plan & Milestone 1 Step-by-Step Guide

---

## Part 0 — Requirements Breakdown (read this first)

### 0.1 The big idea in one sentence
You train **one CNN encoder** on bird images, then reuse it as the backbone for **three tasks**:

| # | Task | Input → Output | Milestone | Encoder role |
|---|------|----------------|-----------|--------------|
| 1 | Classification | RGB image → 1 of 200 species | **M1** | Trained here (scratch + transfer) |
| 2 | Colorization | Grayscale image → color image | M2 | Encoder of a U-Net / encoder-decoder |
| 3 | Captioning | Image → sentence | M2 | **Frozen** feature extractor feeding a text decoder |

**Key consequence for Milestone 1:** the encoder you build now must be easy to plug into a U-Net (needs multi-scale feature maps) and into a caption decoder (needs a spatial feature grid or a pooled vector). Design for that *now*.

### 0.2 Dataset (CUB-200-2011 + captions)
- Hugging Face: `Multimodal-Fatima/CUB_train` (5,994 imgs) and `Multimodal-Fatima/CUB_test` (5,794 imgs), 200 classes, ~30 train images per class.
- Fields: `image`, `label`, `description` (10 captions joined by `\n` — **must be split and cleaned**), `file_name`, `id`.
- No validation split is provided → **you must create one from train**.
- Colorization pairs are made by you: grayscale(image) → original.
- This is a **fine-grained** problem (many species look very alike, few images per class). From-scratch accuracy will be low; that's expected and fine — it's a baseline.

### 0.3 Non-negotiable engineering rules
| Rule | What it means in practice |
|------|---------------------------|
| Python + PyTorch | No Keras/TF for the main code |
| Matplotlib for plots | Others allowed alongside |
| Experiment tracking | MLflow or W&B — log **every** run |
| Modular code, SOLID | data / models / training / evaluation / visualization in separate modules; swapping encoders must not touch the training loop |
| CLI for train **and** inference on **every** task | `python -m src.cli train --task classification --config ...` |
| Notebooks = experiments only | Final code is `.py` run via CLI |
| GitHub from day one | Every member commits from their own account, incrementally |
| Report (PDF) | Documents *everything*, including failures |

### 0.4 What Milestone 1 specifically requires
1. **Task 1.1 — EDA & data prep**
   - Class distribution, image sizes & quality, caption stats (length, vocabulary, typos), anomalies (imbalance, duplicates, corrupt files, noisy captions).
   - Plots for all findings.
   - Clean & preprocess (incl. splitting/cleaning `description`), design an augmentation strategy.
2. **Task 1.2 — Classification & encoder**
   - CNN **from scratch** (baseline).
   - **Transfer learning** with pretrained models.
   - **Compare** the two.
   - **Hyperparameter tuning** (required in M1): **both Grid Search and Random Search**, on a subset of hyperparameters, and **compare the two methods** (best score, #trials, time).
   - **Experiment tracking** for all runs.
   - **Grad-CAM** on correct **and** wrong predictions.
   - Multiple metrics (accuracy, precision, recall, F1, confusion matrix…), each explained.
3. Deadline in the spec: **Week 7 (1 Nov)**, submitted on LMS. Not graded separately, but its content is graded later (EDA 10% + Task 1 10% + parts of code/CLI/benchmarking/discussion).

### 0.5 Grading — where the marks are
| Criterion | Weight | Note |
|-----------|--------|------|
| **Discussion (individual, in person)** | **35%** | Every member must understand the *whole* project |
| Report | 10% | Completeness, honesty |
| EDA / cleaning / augmentation | 10% | **M1** |
| Code quality & reproducibility | 5% | |
| CLI | 5% | |
| Task 1 Classification/Encoder | 10% | **M1** |
| Task 2 Colorization | 10% | M2 |
| Task 3 Captioning | 10% | M2 |
| Benchmarking | 5% | Fair comparison > high score |

> The discussion is worth more than all three tasks combined. Don't let one person write the code everyone else can't explain.

---

## Part 1 — Whole-Project Implementation Steps

### Phase A — Setup (all members, day 1)
1. Create GitHub repo; add all members as collaborators; protect `main`; use branches + pull requests.
2. Create the project skeleton (below), `requirements.txt`, `.gitignore` (ignore `outputs/`, checkpoints, `mlruns/`, `wandb/`, data caches).
3. Choose tracking tool (W&B is easiest for Colab/Kaggle teams; MLflow if you want local/self-hosted).
4. Fix a global seed utility and a config system (YAML + dataclasses).

```
project/
├── README.md
├── requirements.txt
├── configs/
│   ├── cls_scratch.yaml
│   ├── cls_resnet50.yaml
│   ├── tune_grid.yaml
│   └── tune_random.yaml
├── notebooks/            # EDA & experiments only
├── src/
│   ├── data/             # dataset wrappers, caption cleaning, transforms, splits
│   ├── models/           # encoders (scratch, pretrained), heads, registry
│   ├── training/         # trainer, callbacks, tuning
│   ├── evaluation/       # metrics, benchmarking
│   ├── visualization/    # plots, Grad-CAM
│   ├── utils/            # seed, config, logging/tracking wrapper
│   └── cli.py
└── outputs/              # figures, logs (not committed if large)
```

### Phase B — Milestone 1 (Classification & encoder) → detailed in Part 2

### Phase C — Milestone 2a: Colorization
1. Build pairs on the fly: convert RGB → **Lab**; input = L channel, target = ab channels (standard and easier than predicting RGB).
2. Model: U-Net whose encoder = your M1 encoder (use multi-scale features as skip connections). Feed L as 1 channel (adapt first conv) or repeat L to 3 channels.
3. Experiments: encoder frozen vs fine-tuned; scratch encoder vs pretrained encoder; L1 vs L2 loss (L2 tends to give washed-out sepia colors — document this).
4. Metrics: PSNR, MAE (+ optionally SSIM). Visuals: grayscale | prediction | ground truth.
5. CLI: `train --task colorization`, `predict --task colorization --image x.jpg`.

### Phase D — Milestone 2b: Captioning
1. Captions: use the cleaned captions from M1; build vocabulary (min frequency threshold), add `<pad> <start> <end> <unk>`, cap max length using your EDA length stats.
2. **Freeze** encoder; extract spatial features (e.g., 7×7×C grid) → linear projection to decoder dim.
3. Decoder: LSTM with attention (Show, Attend and Tell style) or a small Transformer decoder. Teacher forcing in training; greedy + beam search at inference (autoregressive).
4. Metrics: BLEU-1..4, ROUGE-L (optionally METEOR/CIDEr). Use all 10 references per image.
5. Visuals: image + generated caption + references; attention maps if you use attention.

### Phase E — Benchmarking (both milestones)
- Classification: compare with published CUB-200-2011 accuracies of standard backbones (e.g., fine-tuned ResNet-50) — cite the papers you take numbers from, and use the **same official train/test split**.
- Colorization / captioning: run open-source models (or reported numbers) on the same test split with the same metrics.
- Caveat to mention: the CUB homepage warns that some CUB images overlap with ImageNet, so ImageNet-pretrained models may have a slight unfair advantage.

### Phase F — Deliverables
Google Drive link with trained models · GitHub link (full history) · README (setup + exact commands) · requirements.txt · PDF report (13 sections in §7 of the spec) · video demo → all in `TeamXX.zip` on LMS.

---

## Part 2 — Milestone 1: Step-by-Step

### Step 1 — Load data and create the validation split
1. Load both HF splits; check: number of rows, `label` type (ClassLabel? names available?), image modes (RGB / L / RGBA / CMYK), and whether every `description` really has 10 captions.
2. Create a **stratified** validation split from train (e.g., 85/15, fixed seed) → ~4–5 val images per class.
3. **Save the split indices** (`data/splits/train_idx.json`, `val_idx.json`) so every run and every member uses identical splits.
4. Test set is touched **only** for final evaluation — never for tuning or model selection.
5. Optional speedup on Colab/Kaggle: pre-resize images (e.g., shorter side 256) and cache to disk.

### Step 2 — EDA (Task 1.1) — produce a plot or table for each
**Images**
- Class distribution (bar plot of 200 classes; also the min/max/mean count) for train, val, test.
- Image width/height distributions + aspect-ratio histogram.
- Color modes: count non-RGB images (CUB contains some grayscale images — convert them, and note this matters for colorization!).
- Per-channel mean/std (compute on train only → use for from-scratch normalization).
- Brightness/contrast/blur statistics (e.g., Laplacian variance) to flag low-quality images.
- Corrupted files: try opening/decoding every image.
- Duplicates: perceptual hash (`imagehash`) within train and **between train and test** (leakage check).
- Sample grid: a few images per class; also show the most similar-looking species pairs (fine-grained difficulty).

**Captions**
- Captions per image (should be 10 — verify), empty lines.
- Caption length distribution (words) → choose max length for M2 (e.g., 95th percentile).
- Vocabulary size, word frequency (top-N), number of words appearing once.
- Typos: run a spell checker (e.g., `pyspellchecker`) and list frequent misspellings.
- Noise: non-ASCII chars, odd punctuation, duplicate captions within an image.
- Nice extra: most frequent color words per species — connects to colorization and captioning.

### Step 3 — Caption cleaning (do it now, used in M2)
```python
import re

def split_captions(description: str) -> list[str]:
    return [c.strip() for c in description.split("\n") if c.strip()]

def clean_caption(text: str, spell_fixes: dict[str, str] | None = None) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)        # drop punctuation/digits
    tokens = text.split()
    if spell_fixes:
        tokens = [spell_fixes.get(t, t) for t in tokens]
    return " ".join(tokens)
```
Save a processed file (e.g., `captions_clean.json`: `{id: [cap1..cap10]}`), and document each step and why in the report.

### Step 4 — Preprocessing & augmentation design
- **Train**: `RandomResizedCrop(224, scale=(0.6, 1.0))`, `RandomHorizontalFlip()`, mild `ColorJitter(brightness, contrast, saturation)`, small rotation.
- **Avoid hue jitter**: color is a key species feature (and the target in colorization) — justify this in the report.
- **No vertical flips**: birds aren't upside-down in real photos.
- **Val/Test**: `Resize(256)` → `CenterCrop(224)`.
- Normalization: ImageNet mean/std for pretrained models; your computed train mean/std for scratch model.
- Plot an "augmentation examples" figure (original + 6–8 augmented versions).

### Step 5 — Code architecture (design for M2 now)
Define an encoder interface that every encoder implements:
```python
from abc import ABC, abstractmethod
import torch.nn as nn

class Encoder(nn.Module, ABC):
    out_channels: list[int]   # channels of each feature stage

    @abstractmethod
    def forward_features(self, x) -> list:  # multi-scale maps (for U-Net / captioning)
        ...

    def forward(self, x):                   # pooled vector (for classification)
        return self.forward_features(x)[-1].mean(dim=(2, 3))

class Classifier(nn.Module):
    def __init__(self, encoder: Encoder, num_classes=200, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(encoder.out_channels[-1], num_classes))
    def forward(self, x):
        return self.head(self.encoder(x))
```
- A **registry/factory** (`build_encoder(name, **cfg)`) so configs choose the model by name (Open/Closed principle).
- A generic `Trainer` that takes model, optimizer, loss, loaders, metrics and a tracker — no model-specific code inside (Dependency Inversion).
- A `Tracker` wrapper around W&B/MLflow so you can switch tools without changing the trainer.

### Step 6 — From-scratch baseline CNN
- Architecture idea: 4–5 conv stages (64→128→256→512), each = [Conv-BN-ReLU ×2 + MaxPool] or small residual blocks; global average pooling; dropout; FC-200.
- Training: AdamW or SGD+momentum, cosine LR schedule with warmup, label smoothing (0.1), weight decay, mixed precision (`torch.amp`), 60–100+ epochs, early stopping on val accuracy.
- Expect modest accuracy (6k images, 200 fine-grained classes). Try architecture variations (depth, width, residual vs plain) and log each — these are "architecture experiments" for the report.
- Plot train vs val loss/accuracy curves; discuss overfitting.

### Step 7 — Transfer learning
- Candidates: ResNet-50 (good default — its 4 stages give perfect U-Net skips), EfficientNet-B0, ConvNeXt-Tiny.
- Strategies to compare:
  1. **Linear probe**: freeze backbone, train only the head.
  2. **Full fine-tuning**: unfreeze all, lower LR for backbone (e.g., backbone 1e-4, head 1e-3).
  3. Optional: gradual unfreezing.
- Wrap each pretrained model in your `Encoder` interface (`forward_features` returns the stage outputs).
- Produce the scratch-vs-transfer comparison table + overlaid learning curves.

### Step 8 — Hyperparameter tuning (required in M1)
- Choose a small subset: learning rate, weight decay, batch size, dropout (maybe optimizer).
- Tune on **one** model (e.g., the transfer model, or the scratch model) with a **reduced budget** (fewer epochs, maybe smaller image size) to fit T4 time.
- **Grid Search**: e.g., lr ∈ {1e-4, 3e-4, 1e-3} × wd ∈ {1e-4, 1e-3} × dropout ∈ {0.2, 0.5} = 12 trials.
- **Random Search**: same budget (12 trials) sampling lr log-uniform [1e-5, 1e-2], wd log-uniform, dropout uniform.
- Optional: Optuna (TPE) with same budget.
- Compare: best val score, trials to reach it, total time → table + plot (best-so-far score vs trial number).
- Retrain the best config fully, then evaluate on test once.

### Step 9 — Evaluation
- Metrics: accuracy, top-5 accuracy, macro precision/recall/F1, per-class accuracy, confusion matrix.
- A 200×200 confusion matrix is unreadable — also show the **top-k most confused class pairs** with sample images.
- For each metric, write: what it is, what it measures, what your score means.

### Step 10 — Grad-CAM
```python
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model, None, None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x, class_idx=None):
        self.model.eval()
        logits = self.model(x)
        idx = logits.argmax(1) if class_idx is None else class_idx
        self.model.zero_grad()
        logits.gather(1, idx.view(-1, 1)).sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)        # GAP over gradients
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam_min = cam.amin(dim=(2, 3), keepdim=True)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        return ((cam - cam_min) / (cam_max - cam_min + 1e-8)).squeeze(1), idx
```
- Target layer = last conv stage of each encoder.
- Show grids for: correct predictions, wrong predictions (with true vs predicted class), and the **same images under scratch vs transfer** models.
- Analyse: does the model look at the bird (head, wing patterns) or at background (water, branches)? Background focus on wrong predictions is a great discussion point.

### Step 11 — Experiment tracking checklist (per run)
Config/hyperparameters · git commit hash · seed · per-epoch train/val loss & metrics · learning rate · final test metrics · best checkpoint (or Drive path) · key figures (curves, confusion matrix, Grad-CAM). Use clear run names/tags (`cls-scratch-v3`, `cls-resnet50-ft`, `tune-grid`, `tune-random`).

### Step 12 — CLI for M1
```bash
python -m src.cli train    --task classification --config configs/cls_resnet50.yaml
python -m src.cli evaluate --task classification --checkpoint outputs/cls_resnet50/best.pt
python -m src.cli predict  --task classification --image path/to/bird.jpg --topk 5
python -m src.cli tune     --task classification --config configs/tune_grid.yaml
python -m src.cli gradcam  --task classification --image path/to/bird.jpg
```
Use `argparse` subcommands (or `typer`). Test every command from a clean environment before submission.

### Step 13 — Benchmark (classification part)
Find published CUB-200-2011 accuracies for the same backbone (e.g., ResNet-50 fine-tuned), confirm they use the official split, put your numbers next to theirs, and explain differences (resolution, epochs, augmentation, specialized fine-grained methods, ImageNet overlap).

### Step 14 — Wrap up & submit M1
- README section for M1 commands; requirements.txt pinned.
- Short M1 write-up (it becomes sections 2, 4, 5, 6, 7, 8 of the final report).
- Upload checkpoints to Drive; submit on LMS.

---

## Part 3 — Suggested Team Split (4 members example)
| Member | Main ownership | Also must understand |
|--------|----------------|----------------------|
| A | Data module, EDA, caption cleaning, augmentation | Grad-CAM, metrics |
| B | Scratch CNN + trainer + tracking wrapper | Transfer models |
| C | Transfer learning + encoder interface (M2-ready) | Tuning |
| D | Tuning (grid/random/Optuna), evaluation, Grad-CAM, CLI | EDA |

Rotate code reviews via pull requests so everyone reads everyone's code — this is your preparation for the 35% individual discussion.

## Part 4 — Common Pitfalls
- Tuning or selecting models on the **test** set.
- Different members using different random splits.
- Hue jitter / aggressive color augmentation destroying the color signal.
- Encoder that only returns a pooled vector → painful rewrite for U-Net in M2.
- Committing datasets / checkpoints to Git.
- One giant final commit (explicitly penalized).
- Reporting only one metric, or metrics without interpretation.
