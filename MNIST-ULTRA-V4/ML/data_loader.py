"""
data_loader.py
==============
Loads the MNIST dataset via Keras and preprocesses it
for training a dense (fully-connected) Neural Network.
"""

import numpy as np
from tensorflow import keras


def load_and_preprocess():
    """
    Downloads (on first run) and preprocesses MNIST.

    Returns
    -------
    (x_train, y_train), (x_test, y_test)
        x  : float32 arrays of shape (N, 784), values in [0, 1]
        y  : int arrays of shape (N,), values 0–9
    """
    # Load raw data — shape: (60000, 28, 28) and (10000, 28, 28)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Flatten 28×28 images → 784-element vectors
    x_train = x_train.reshape(-1, 784).astype("float32")
    x_test  = x_test.reshape(-1, 784).astype("float32")

    # Normalize pixel values from [0, 255] → [0.0, 1.0]
    x_train /= 255.0
    x_test  /= 255.0

    return (x_train, y_train), (x_test, y_test)