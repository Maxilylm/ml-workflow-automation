"""Simple MNIST training script."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
sys.path.insert(0, ".")

import numpy as np


def main():
    print("Loading data...")
    images = np.load("data/raw/mnist_images.npy")
    labels = np.load("data/raw/mnist_labels.npy")

    # Normalize
    X = images / 255.0
    y = labels

    # Reshape for CNN
    X = X.reshape(-1, 28, 28, 1)

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )

    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # Build model
    print("Building CNN model...")
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=128,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_cnn.keras")
    print("\nModel saved to models/mnist_cnn.keras")

    # Save training history
    import json
    with open("models/training_history.json", 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)

    return test_acc


if __name__ == "__main__":
    main()
