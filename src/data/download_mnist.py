"""Download and save MNIST dataset."""

import os
import numpy as np
from sklearn.datasets import fetch_openml


def download_mnist(data_dir: str = "data/raw") -> dict:
    """
    Download MNIST dataset from OpenML and save to disk.

    Args:
        data_dir: Directory to save the data

    Returns:
        Dictionary with dataset info
    """
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading MNIST dataset from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(np.int32)

    # Save to disk
    np.save(os.path.join(data_dir, "mnist_images.npy"), X)
    np.save(os.path.join(data_dir, "mnist_labels.npy"), y)

    info = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "image_shape": (28, 28),
        "n_classes": len(np.unique(y)),
        "classes": sorted(np.unique(y).tolist()),
    }

    print(f"Dataset saved to {data_dir}")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']} (28x28 images)")
    print(f"  Classes: {info['n_classes']} (digits 0-9)")

    return info


if __name__ == "__main__":
    download_mnist()
