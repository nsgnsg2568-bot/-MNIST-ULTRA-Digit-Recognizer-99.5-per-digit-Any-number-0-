# MNIST Ultra — Multi-Digit Number Recognizer (0 → ∞)
## ~99.5% accuracy per digit · reads whole numbers of any length

### ✨ New in v4: numbers, not just digits
Draw `7`, `2048` or `90210000000000001` — the app splits your drawing into digits,
reads each one with the CNN, and joins them left → right. There is **no built-in limit**
on the number of digits (the answer is handled as text, so it never overflows).

* **Bigger canvas** (1000×450) with **brush sizes XXS · XS · S · M · L · XL**
* **Canvas width 1× / 2× / 3× / 5×** — keep writing past the edge (auto-scrolls while you draw)
* **Overlay boxes** show what the model detected; green/orange/red = confidence
* Click any digit card to see its full 0–9 probability breakdown
* **No retraining needed** — your existing saved model works as-is

### How the number reading works (`segmenter.py`)
1. Binarize the canvas → find connected strokes (`scipy.ndimage.label`)
2. Drop specks of noise
3. Merge strokes that sit on top of each other (the bar of a `5`, an open `4`)
4. Split blobs that are too wide (two digits that touch) at their thinnest "neck"
5. Sort left → right
6. Normalize each digit like MNIST: pen thickness normalized, fitted in 20×20, centred by centre-of-mass in 28×28

**Tips / limits:** leave a small gap between digits for best results. Touching digits are split
heuristically, so very tangled handwriting may occasionally be mis-split. Only whole non-negative
numbers are supported (no minus sign, decimal point or commas).

Run the tests (no TensorFlow needed): `python tests/test_segmenter.py`

---

---

## ✅ Full Pro Checklist Implemented

| # | Technique | Status |
|---|-----------|--------|
| 1 | Normalize pixels [0–255] → [0–1] | ✅ |
| 2 | Data Augmentation (rotate ±10°, zoom ±10%, shift ±10%) | ✅ |
| 3 | Reshape to 28×28×1 for CNN | ✅ |
| 4 | Conv2D layers (32 → 64 → 128 filters) | ✅ |
| 5 | Batch Normalization after every Conv layer | ✅ |
| 6 | MaxPooling2D for downsampling | ✅ |
| 7 | Dropout (0.25 + 0.5) for regularization | ✅ |
| 8 | Dense(512) + Softmax(10) classifier | ✅ |
| 9 | Adam optimizer with Learning Rate Decay | ✅ |
| 10 | ReduceLROnPlateau scheduler | ✅ |
| 11 | EarlyStopping (restores best weights) | ✅ |
| 12 | Confusion Matrix | ✅ |
| 13 | Wrong Predictions Visualization | ✅ |
| 14 | Live training charts (accuracy + loss) | ✅ |

---

## 🚀 How to Run (Step by Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start the server
```bash
python server.py
```

### Step 3 — Open browser
```
http://localhost:5000
```

### Step 4 — Click "START TRAINING"
- Watch all 4 steps complete live in your browser
- Training takes ~15–25 minutes on CPU
- After training, draw digits and test the model!

---

## Project Structure

```
mnist_best/
├── server.py           ← Flask backend (training + prediction API)
├── segmenter.py        ← Multi-digit splitting + MNIST-style preprocessing
├── tests/              ← Tests for segmenter.py
├── templates/
│   └── index.html      ← Full frontend UI
├── static/             ← Auto-created: confusion_matrix.png, wrong_predictions.png
├── saved_model/        ← Auto-created: trained model saved here
├── requirements.txt
└── README.md
```

---

## Expected Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~99.2–99.5% |
| Errors per 10,000 | ~50–80 |
| Training Epochs | ~15–20 (early stopping) |

The model is saved after training. Next time you click START, it loads the saved model instantly — no retraining needed. Click "RETRAIN FROM SCRATCH" to train fresh.