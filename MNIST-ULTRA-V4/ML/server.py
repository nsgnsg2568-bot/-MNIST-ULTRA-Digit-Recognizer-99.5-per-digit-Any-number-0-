'''server.py  —  MNIST Ultra Backend (v4 — multi-digit numbers)
=============================================================
Reads ANY whole number (0 to as many digits as you can fit on the canvas):
the drawing is split into digits (segmenter.py), each digit is read by the
CNN, and the digits are joined left -> right.

Run:  python server.py
Open: http://localhost:5000
'''

import os, io, base64, threading, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix
import seaborn as sns

from segmenter import segment_digits, load_gray

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Global state ──────────────────────────────────────────────────────────────
training_log   = []
model          = None
training_done  = False
test_accuracy  = 0.0
history_data   = {}
wrong_imgs     = []
conf_matrix    = None

predict_lock   = threading.Lock()     # Keras is not guaranteed thread-safe

MODEL_PATH = "saved_model/mnist_best.keras"
os.makedirs("saved_model", exist_ok=True)
os.makedirs("static",      exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUGMENTATION LAYERS — created ONCE at module level (required by tf.function)
# Creating Keras layers inside tf.data.map / tf.function causes:
#   "tf.Variables created on first call" ValueError — this fixes it.
# ─────────────────────────────────────────────────────────────────────────────
_aug_rotate    = keras.layers.RandomRotation(factor=0.05,   fill_mode='nearest', seed=1)
_aug_translate = keras.layers.RandomTranslation(height_factor=0.10, width_factor=0.10,
                                                fill_mode='nearest', seed=2)
_aug_zoom      = keras.layers.RandomZoom(height_factor=(-0.10, 0.10),
                                         fill_mode='nearest', seed=3)

def augment_batch(imgs, labels):
    imgs = tf.cast(imgs, tf.float32)
    imgs = _aug_rotate(imgs,    training=True)
    imgs = _aug_translate(imgs, training=True)
    imgs = _aug_zoom(imgs,      training=True)
    return imgs, labels

# ─────────────────────────────────────────────────────────────────────────────
# PUSH EVENT HELPER
# ─────────────────────────────────────────────────────────────────────────────
def push(kind, id_, label="", status="", data=None):
    training_log.append({
        "kind": kind, "id": id_,
        "label": label, "status": status,
        "data": data or {},
        "ts": time.time(),
    })

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    push("step", "step1", "Data Preparation", "loading")
    push("log", "d1", "Downloading MNIST dataset...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    push("log", "d2", f"Raw shapes: train={x_train.shape}  test={x_test.shape}")

    # Normalize [0,255] → [0.0, 1.0]
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    push("log", "d3", "Normalized pixels: [0-255] to [0.0-1.0]")

    # Reshape to (N, 28, 28, 1) — CNN needs channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test  = x_test.reshape(-1, 28, 28, 1)
    push("log", "d4", "Reshaped to 28x28x1 for CNN channel dimension")

    # One-hot encode labels
    y_train_oh = keras.utils.to_categorical(y_train, 10)
    y_test_oh  = keras.utils.to_categorical(y_test,  10)
    push("log", "d5", "One-hot encoded labels (10 classes)")
    push("step", "step1", "Data Preparation", "done")

    return (x_train, y_train_oh, y_train), (x_test, y_test_oh, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BUILD CNN
# ─────────────────────────────────────────────────────────────────────────────
def build_model():
    push("step", "step2", "Build Deep CNN", "loading")

    inputs = keras.Input(shape=(28, 28, 1), name="input")

    # Block 1 — 32 filters
    x = keras.layers.Conv2D(32, (3,3), padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(32, (3,3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Dropout(0.25)(x)
    push("log", "a1", "Block 1: Conv2D(32) x2 -> BN -> ReLU -> MaxPool -> Dropout(0.25)")

    # Block 2 — 64 filters
    x = keras.layers.Conv2D(64, (3,3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Conv2D(64, (3,3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Dropout(0.25)(x)
    push("log", "a2", "Block 2: Conv2D(64) x2 -> BN -> ReLU -> MaxPool -> Dropout(0.25)")

    # Block 3 — 128 filters
    x = keras.layers.Conv2D(128, (3,3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(2, 2)(x)
    x = keras.layers.Dropout(0.25)(x)
    push("log", "a3", "Block 3: Conv2D(128) -> BN -> ReLU -> MaxPool -> Dropout(0.25)")

    # Classifier
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10, activation="softmax", name="output")(x)

    m = keras.Model(inputs, outputs, name="MNIST_Ultra_CNN")
    push("log", "a4", f"Classifier: Flatten -> Dense(512) -> BN -> Dropout(0.5) -> Softmax(10)")
    push("log", "a5", f"Total parameters: {m.count_params():,}")
    push("step", "step2", "Build Deep CNN", "done")
    return m

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TRAINING WITH AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def train_model(m, x_train, y_train_oh, x_test, y_test_oh):
    global history_data
    push("step", "step3", "Augmentation + Training", "loading")
    push("log", "t1", "Augmentation: RandomRotation(+-18deg) + RandomTranslation(+-10%) + RandomZoom(+-10%)")

    m.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    push("log", "t2", "Compiled: Adam(lr=0.001) + Categorical CrossEntropy")

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=0
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=0
        ),
        TrainingCallback(),
    ]
    push("log", "t3", "Callbacks: ReduceLROnPlateau(patience=3) + EarlyStopping(patience=6)")
    push("log", "t4", "Building tf.data pipeline with augmentation...")

    BATCH = 64

    # Build tf.data datasets
    # Augmentation layers are pre-built at module level — safe inside map()
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
        .shuffle(buffer_size=10000, seed=42)
        .batch(BATCH)
        .map(augment_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_oh))
        .batch(256)
        .prefetch(tf.data.AUTOTUNE)
    )

    push("log", "t5", "Training started — up to 30 epochs with early stopping...")

    hist = m.fit(
        train_ds,
        epochs=30,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=0,
    )

    history_data = {
        "accuracy":     [float(v) for v in hist.history.get("accuracy",     [])],
        "val_accuracy": [float(v) for v in hist.history.get("val_accuracy", [])],
        "loss":         [float(v) for v in hist.history.get("loss",         [])],
        "val_loss":     [float(v) for v in hist.history.get("val_loss",     [])],
        "lr":           [float(v) for v in hist.history.get("lr",           [])],
    }
    push("step", "step3", "Augmentation + Training", "done")
    return hist

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(m, x_test, y_test_oh, y_test_int):
    global test_accuracy, wrong_imgs, conf_matrix
    push("step", "step4", "Evaluation & Analysis", "loading")

    loss, acc = m.evaluate(x_test, y_test_oh, verbose=0)
    test_accuracy = acc
    push("log", "e1", f"Test Loss     : {loss:.4f}")
    push("log", "e2", f"Test Accuracy : {acc*100:.2f}%")

    probs  = m.predict(x_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test_int, y_pred)
    conf_matrix = cm.tolist()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10), ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("static/confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    push("log", "e3", "Confusion Matrix computed and saved")

    # Wrong predictions — top 12 most confident mistakes
    wrong_idx   = np.where(y_pred != y_test_int)[0]
    push("log", "e4", f"Total wrong: {len(wrong_idx)} / {len(y_test_int)}")

    confidences = probs[wrong_idx, y_pred[wrong_idx]]
    top_wrong   = wrong_idx[np.argsort(-confidences)[:12]]

    wrong_imgs.clear()
    for idx in top_wrong:
        img_arr = (x_test[idx].reshape(28, 28) * 255).astype(np.uint8)
        pil     = Image.fromarray(img_arr).resize((56, 56), Image.NEAREST)
        buf     = io.BytesIO()
        pil.save(buf, format="PNG")
        wrong_imgs.append({
            "img":  "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "true": int(y_test_int[idx]),
            "pred": int(y_pred[idx]),
            "conf": round(float(probs[idx][y_pred[idx]] * 100), 1),
        })
    push("log", "e5", "Top 12 confident mistakes captured for display")
    push("step", "step4", "Evaluation & Analysis", "done")

# ─────────────────────────────────────────────────────────────────────────────
# KERAS CALLBACK — pushes live epoch data
# ─────────────────────────────────────────────────────────────────────────────
class TrainingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        push("epoch", str(epoch+1), "", "", {
            "epoch":    epoch + 1,
            "acc":      round(float(logs.get("accuracy",     0)) * 100, 2),
            "val_acc":  round(float(logs.get("val_accuracy", 0)) * 100, 2),
            "loss":     round(float(logs.get("loss",         0)), 4),
            "val_loss": round(float(logs.get("val_loss",     0)), 4),
            "lr":       float(logs.get("lr", 0.001)),
        })

# ─────────────────────────────────────────────────────────────────────────────
# METRICS FOR A MODEL LOADED FROM DISK
# (after a server restart the accuracy / confusion matrix would otherwise be empty)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics_for_loaded(m):
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = (x_test.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
    y_oh   = keras.utils.to_categorical(y_test, 10)
    evaluate_model(m, x_test, y_oh, y_test)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING THREAD
# ─────────────────────────────────────────────────────────────────────────────
def run_training():
    global model, training_done
    try:
        (x_train, y_train_oh, _), (x_test, y_test_oh, y_test_int) = load_data()
        m = build_model()
        train_model(m, x_train, y_train_oh, x_test, y_test_oh)
        evaluate_model(m, x_test, y_test_oh, y_test_int)
        m.save(MODEL_PATH)
        push("log",  "s1",   f"Model saved to {MODEL_PATH}")
        push("done", "done", "Training Complete!", "done", {
            "accuracy": round(test_accuracy * 100, 2),
        })
        model = m
    except Exception as e:
        import traceback
        push("error", "error", f"Error: {str(e)}", "error")
        push("log",   "err",   traceback.format_exc())
    finally:
        training_done = True

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/start", methods=["POST"])
def start_training():
    global training_log, training_done, model
    training_log  = []
    training_done = False
    if os.path.exists(MODEL_PATH):
        try:
            model = keras.models.load_model(MODEL_PATH)
            push("log",  "load", f"Pre-trained model loaded from {MODEL_PATH}")
            if not conf_matrix:
                try:
                    compute_metrics_for_loaded(model)
                except Exception as e:
                    push("log", "metrics_err", f"Could not compute test metrics: {e}")
            push("done", "done", "Model ready!", "done",
                 {"accuracy": round(test_accuracy * 100, 2)})
            training_done = True
            return jsonify({"status": "loaded"})
        except Exception as e:
            push("log", "load_err", f"Load failed ({e}). Retraining...")
    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route("/retrain", methods=["POST"])
def retrain():
    global training_log, training_done, model, history_data, wrong_imgs, conf_matrix, test_accuracy
    training_log  = []
    training_done = False
    model         = None
    history_data  = {}
    wrong_imgs    = []
    conf_matrix   = None
    test_accuracy = 0.0
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({"status": "started"})

@app.route("/poll")
def poll():
    since      = float(request.args.get("since", 0))
    new_events = [e for e in training_log if e["ts"] > since]
    return jsonify({
        "events":      new_events,
        "done":        training_done,
        "accuracy":    round(test_accuracy * 100, 2),
        "history":     history_data,
        "wrong_imgs":  wrong_imgs  if training_done else [],
        "conf_matrix": conf_matrix if training_done else None,
    })

def _group_digits(s: str) -> str:
    """'1234567' -> '1,234,567'  (string based, so there is NO size limit)."""
    core = s.lstrip("0") or "0"
    out = []
    for i, ch in enumerate(reversed(core)):
        if i and i % 3 == 0:
            out.append(",")
        out.append(ch)
    return "".join(reversed(out))


def _preview_png(img28):
    """28x28 float image -> small base64 PNG for the UI."""
    pil = Image.fromarray((img28 * 255).astype(np.uint8)).resize((56, 56), Image.NEAREST)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/predict", methods=["POST"])
def predict():
    """
    Reads a whole NUMBER from the drawing (1 digit or thousands of them).

    Response
    --------
    number     : "2048"            digits joined left -> right (string, any length)
    formatted  : "2,048"           same number with thousands separators
    count      : 4                 how many digits were found
    avg_conf / min_conf            confidence summary (percent)
    digits     : [ {digit, conf, probs[10], box[x0,y0,x1,y1], preview} , ... ]
    digit/conf/probs               (first digit - kept for backwards compatibility)
    """
    if model is None:
        return jsonify({"error": "Model not ready"}), 400

    data = request.get_json(silent=True) or {}
    try:
        img_b64 = data["image"].split(",", 1)[1]
        pil     = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        orig_w  = pil.width
        gray    = load_gray(pil)
    except Exception:
        return jsonify({"error": "Could not read the image"}), 400

    segments = segment_digits(gray)
    if not segments:
        return jsonify({"number": "", "formatted": "", "count": 0,
                        "avg_conf": 0, "min_conf": 0, "digits": []})

    batch = np.stack([sg["image"] for sg in segments])[..., np.newaxis]   # (N,28,28,1)
    with predict_lock:
        probs_all = model.predict(batch, batch_size=256, verbose=0)

    scale  = orig_w / gray.shape[1]          # boxes back to the browser's canvas pixels
    digits = []
    for sg, probs in zip(segments, probs_all):
        d = int(np.argmax(probs))
        digits.append({
            "digit":   d,
            "conf":    round(float(probs[d]) * 100, 1),
            "probs":   [round(float(p) * 100, 1) for p in probs],
            "box":     [int(round(v * scale)) for v in sg["box"]],
            "preview": _preview_png(sg["image"]),
        })

    number = "".join(str(d["digit"]) for d in digits)
    confs  = [d["conf"] for d in digits]
    return jsonify({
        "number":    number,
        "formatted": _group_digits(number),
        "count":     len(digits),
        "avg_conf":  round(sum(confs) / len(confs), 1),
        "min_conf":  min(confs),
        "digits":    digits,
        # backwards compatible single-digit fields
        "digit":     digits[0]["digit"],
        "conf":      digits[0]["conf"],
        "probs":     digits[0]["probs"],
    })

@app.route("/results")
def results():
    return jsonify({
        "history":     history_data,
        "wrong_imgs":  wrong_imgs,
        "conf_matrix": conf_matrix,
        "accuracy":    round(test_accuracy * 100, 2),
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  MNIST Ultra — v4 (multi-digit numbers)")
    print("  Open: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
