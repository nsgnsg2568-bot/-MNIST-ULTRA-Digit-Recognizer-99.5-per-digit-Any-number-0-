"""
MNIST Handwritten Digit Recognizer
===================================
A simple Neural Network using Keras/TensorFlow.
Trains on the MNIST dataset, evaluates accuracy,
and visualizes the images the model gets wrong.
"""

from model import build_model, train_model, evaluate_model
from data_loader import load_and_preprocess
from visualizer import show_wrong_predictions


def main():
    # ── 1. Load & preprocess ──────────────────────────────────────────────
    print("\n📦  Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    print(f"    Train samples : {len(x_train)}")
    print(f"    Test  samples : {len(x_test)}")

    # ── 2. Build model ────────────────────────────────────────────────────
    print("\n🧠  Building Neural Network...")
    model = build_model()
    model.summary()

    # ── 3. Train ──────────────────────────────────────────────────────────
    print("\n🚀  Training...")
    history = train_model(model, x_train, y_train)

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    print("\n📊  Evaluating on test set...")
    test_loss, test_acc = evaluate_model(model, x_test, y_test)
    print(f"    Test accuracy : {test_acc * 100:.2f}%")
    print(f"    Test loss     : {test_loss:.4f}")

    # ── 5. Show wrong predictions ─────────────────────────────────────────
    print("\n🔍  Finding wrong predictions...")
    show_wrong_predictions(model, x_test, y_test, max_images=20)

    # ── 6. Save model ─────────────────────────────────────────────────────
    import os
    os.makedirs("saved_model", exist_ok=True)   # create folder if it doesn't exist
    model.save("saved_model/mnist_model.keras")
    print("\n✅  Model saved to  saved_model/mnist_model.keras")


if __name__ == "__main__":
    main()
