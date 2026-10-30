"""
model.py
========
Defines, compiles, trains, and evaluates a simple
3-layer Neural Network:

    Input (784)  →  Hidden (128, ReLU)  →  Output (10, Softmax)
"""

from tensorflow import keras


def build_model() -> keras.Sequential:
    """
    Build and compile the Neural Network.

    Architecture
    ------------
    - Input layer  : 784 neurons  (one per pixel)
    - Hidden layer : 128 neurons  (ReLU activation)
    - Output layer : 10 neurons   (Softmax → digit probabilities)

    Returns
    -------
    keras.Sequential
        Compiled model ready for training.
    """
    model = keras.Sequential(
        [
            # ── Input layer ───────────────────────────────────────────────
            keras.layers.Input(shape=(784,), name="input"),

            # ── Hidden layer ──────────────────────────────────────────────
            # 128 neurons with ReLU activation.
            # ReLU (Rectified Linear Unit): f(x) = max(0, x)
            # Helps the network learn non-linear patterns.
            keras.layers.Dense(128, activation="relu", name="hidden"),

            # ── Output layer ──────────────────────────────────────────────
            # 10 neurons — one for each digit (0–9).
            # Softmax converts raw scores into probabilities that sum to 1.
            keras.layers.Dense(10, activation="softmax", name="output"),
        ],
        name="mnist_net",
    )

    model.compile(
        optimizer="adam",               # Adaptive learning-rate optimizer
        loss="sparse_categorical_crossentropy",  # Works with integer labels
        metrics=["accuracy"],
    )

    return model


def train_model(model: keras.Sequential, x_train, y_train):
    """
    Train the model on MNIST training data.

    Parameters
    ----------
    model   : compiled Keras model
    x_train : float32 array (60000, 784)
    y_train : int array    (60000,)

    Returns
    -------
    keras.callbacks.History
    """
    history = model.fit(
        x_train,
        y_train,
        epochs=10,          # 10 full passes through the training data
        batch_size=32,      # Update weights every 32 samples
        validation_split=0.1,  # Hold out 10 % of training data for val loss
        verbose=1,
    )
    return history


def evaluate_model(model: keras.Sequential, x_test, y_test):
    """
    Evaluate trained model on the test set.

    Returns
    -------
    (test_loss, test_accuracy)
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return test_loss, test_acc