"""
visualizer.py
=============
Uses matplotlib to display the images the model
predicted incorrectly, so you can understand *why*
it was confused.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def show_wrong_predictions(model, x_test, y_test, max_images: int = 20):
    """
    Find all test images the model got wrong and display them
    in a grid with the true label vs. the predicted label.

    Parameters
    ----------
    model      : trained Keras model
    x_test     : float32 array (N, 784)
    y_test     : int array     (N,)
    max_images : int — cap how many wrong images to show
    """
    # ── Get predictions ───────────────────────────────────────────────────
    probs        = model.predict(x_test, verbose=0)          # (N, 10) probabilities
    y_pred       = np.argmax(probs, axis=1)                  # (N,)   predicted class
    y_true       = y_test                                    # (N,)   true class

    # ── Find wrong indices ────────────────────────────────────────────────
    wrong_idx    = np.where(y_pred != y_true)[0]
    total_wrong  = len(wrong_idx)
    show_count   = min(max_images, total_wrong)

    print(f"    Total wrong   : {total_wrong} / {len(y_test)}")
    print(f"    Showing first : {show_count}")

    if show_count == 0:
        print("    Perfect score — nothing wrong to display! 🎉")
        return

    # ── Plot grid ─────────────────────────────────────────────────────────
    cols = 5
    rows = (show_count + cols - 1) // cols  # ceiling division

    fig = plt.figure(figsize=(cols * 2.4, rows * 2.8))
    fig.suptitle(
        f"Wrong Predictions  ({total_wrong} errors out of {len(y_test)} test images)",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )

    for plot_i, idx in enumerate(wrong_idx[:show_count]):
        ax = fig.add_subplot(rows, cols, plot_i + 1)

        # Reshape flat 784-vector back to 28×28 for display
        image = x_test[idx].reshape(28, 28)

        ax.imshow(image, cmap="gray_r", interpolation="nearest")
        ax.set_title(
            f"True: {y_true[idx]}   Got: {y_pred[idx]}",
            fontsize=9,
            color="#c0392b" if y_pred[idx] != y_true[idx] else "#27ae60",
        )
        ax.axis("off")

        # Confidence bar under each image
        confidence = probs[idx][y_pred[idx]] * 100
        ax.set_xlabel(f"conf: {confidence:.1f}%", fontsize=7, color="#888888")

    plt.tight_layout()
    plt.savefig("wrong_predictions.png", dpi=150, bbox_inches="tight")
    print("    Saved chart → wrong_predictions.png")
    plt.show()


def plot_training_history(history):
    """
    Plot training & validation accuracy and loss curves.

    Parameters
    ----------
    history : keras.callbacks.History (returned by model.fit)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax1.plot(history.history["accuracy"],     label="Train",      color="#2980b9")
    ax1.plot(history.history["val_accuracy"], label="Validation", color="#e74c3c", linestyle="--")
    ax1.set_title("Accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax2.plot(history.history["loss"],     label="Train",      color="#2980b9")
    ax2.plot(history.history["val_loss"], label="Validation", color="#e74c3c", linestyle="--")
    ax2.set_title("Loss over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches="tight")
    print("    Saved chart → training_history.png")
    plt.show()