# Run this project on Google Colab

## Step 1: Open Colab and enable GPU

1. Go to [colab.research.google.com](https://colab.research.google.com).
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (e.g. T4) → Save.

---

## Step 2: Clone the project

In a new notebook cell, run:

```python
!git clone https://github.com/YOUR_USERNAME/ML_Projectt.git
%cd ML_Projectt
```

If the project is not on GitHub, upload the folder to Google Drive, then:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/ML_Projectt   # adjust path to your folder
```

---

## Step 3: Install dependencies

```python
!pip install -r requirements.txt
```

If you get errors, install one by one:

```python
!pip install torch torchvision torchmetrics pillow numpy facenet-pytorch
```

---

## Step 4: Prepare data

**Option A – You already have data in the repo/Drive**

- Ensure these folders exist and contain `.jpg` images:
  - `data/train`
  - `data/val`
  - `data/test`
- Optional: put another face dataset in `data/test_other` for generalization.

**Option B – Download CelebA and split**

- Download CelebA (e.g. from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) or official source).
- Upload to Drive or Colab, then copy images into `data/train`, `data/val`, `data/test` (e.g. 80% train, 10% val, 10% test).

---

## Step 5: Set more epochs (optional)

Edit `config.py` and increase `EPOCHS` (e.g. 50 or 100). Or in a Colab cell before training run:

```python
import config
config.EPOCHS = 50   # or 100 for better results
config.EARLY_STOPPING_PATIENCE = 10
```

---

## Step 6: Run training

Open `train.ipynb` in Colab (File → Upload notebook, or open from Drive).

Run all cells in order. The first cells install packages and set the working directory; make sure the `%cd` points to the project root (e.g. `ML_Projectt` or your Drive path).

Training will save the best model to `model.pth`.

---

## Step 7: Run evaluation

Open `evaluate.ipynb` and run all cells. You will get:

- PSNR, SSIM, and face embedding distance on validation and test.
- If `data/test_other` exists, generalization metrics on that set.
- Triplet images: degraded | restored | ground truth.

---

## Quick checklist

| Step | Action |
|------|--------|
| 1 | GPU enabled in Runtime |
| 2 | Clone or `%cd` to project folder |
| 3 | `pip install -r requirements.txt` |
| 4 | `data/train`, `data/val`, `data/test` filled with images |
| 5 | Increase `EPOCHS` in config if desired |
| 6 | Run all cells in `train.ipynb` |
| 7 | Run all cells in `evaluate.ipynb` |

---

## Tips

- **Out of memory:** Reduce `BATCH_SIZE` in `config.py` (e.g. 8 or 4).
- **Faster training:** Set `USE_ALIGNMENT = False` in config (only if images are already face-cropped).
- **Colab disconnects:** Save `model.pth` to Drive periodically or use “Run all” and hope it finishes; better: save checkpoints to Drive in code.
