# Image Classification — CIFAR-10

Trains a VGG-style CNN to classify 32×32 color images into 10 categories:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

The dataset (~170 MB) downloads automatically to `./data` on first run.

## Run

```powershell
py train.py                 # 15 epochs (default)
py train.py --epochs 5      # shorter
py train.py --smoke-test    # 1 epoch on a tiny subset to verify it works
```

## What it does

- **`model.py`** — `CIFARNet`: three conv blocks (32→64→128 channels, each
  two 3×3 convs + batch-norm + max-pool) followed by a dropout classifier head.
- **`train.py`** — downloads CIFAR-10, applies light data augmentation
  (random crop + horizontal flip), trains with Adam + cosine LR schedule,
  evaluates on the test set each epoch, saves the best model to `cifar_cnn.pth`,
  and writes `training_curve.png`.

## Expected results

On CPU, expect roughly 70–80% test accuracy after ~15 epochs. A GPU or more
epochs would push it higher. Each epoch on CPU takes a few minutes.
# MP-Online-Assignment-Image-Classification
