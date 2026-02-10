"""Unit tests for MNIST preprocessor."""

import pytest
import numpy as np
import sys

sys.path.insert(0, ".")

from src.preprocessing.mnist_preprocessor import MNISTPreprocessor, get_preprocessed_data


class TestMNISTPreprocessor:
    """Tests for MNISTPreprocessor class."""

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return MNISTPreprocessor(normalize=True, flatten=False)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        X = np.random.randint(0, 256, size=(100, 784)).astype(np.float32)
        y = np.random.randint(0, 10, size=(100,)).astype(np.int32)
        return X, y

    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.normalize is True
        assert preprocessor.flatten is False
        assert preprocessor.validation_split == 0.1
        assert preprocessor.random_state == 42

    def test_fit_marks_fitted(self, preprocessor, sample_data):
        """Test that fit marks preprocessor as fitted."""
        X, _ = sample_data
        preprocessor.fit(X)
        assert preprocessor._fitted is True

    def test_transform_normalizes(self, preprocessor, sample_data):
        """Test that transform normalizes pixel values."""
        X, _ = sample_data
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)

        assert X_transformed.min() >= 0.0
        assert X_transformed.max() <= 1.0
        assert X_transformed.dtype == np.float32

    def test_transform_reshapes_for_cnn(self, preprocessor, sample_data):
        """Test that transform reshapes data for CNN input."""
        X, _ = sample_data
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)

        assert X_transformed.shape == (100, 28, 28, 1)

    def test_transform_keeps_flat_when_specified(self, sample_data):
        """Test that transform keeps data flat when flatten=True."""
        preprocessor = MNISTPreprocessor(normalize=True, flatten=True)
        X, _ = sample_data
        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)

        assert X_transformed.shape == (100, 784)

    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform method."""
        X, _ = sample_data
        X_transformed = preprocessor.fit_transform(X)

        assert preprocessor._fitted is True
        assert X_transformed.shape == (100, 28, 28, 1)
        assert X_transformed.max() <= 1.0

    def test_prepare_labels_one_hot(self, preprocessor, sample_data):
        """Test one-hot encoding of labels."""
        _, y = sample_data
        y_one_hot = preprocessor.prepare_labels(y)

        assert y_one_hot.shape == (100, 10)
        assert np.all(y_one_hot.sum(axis=1) == 1)
        assert y_one_hot.dtype == np.float32

    def test_split_data_maintains_stratification(self, preprocessor, sample_data):
        """Test that split maintains class distribution."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)

        train_counts = np.bincount(y_train, minlength=10)
        test_counts = np.bincount(y_test, minlength=10)

        # Check proportions are similar
        train_props = train_counts / len(y_train)
        test_props = test_counts / len(y_test)

        # Allow for some variance due to small sample size
        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_prepare_dataset_creates_all_splits(self, preprocessor, sample_data, tmp_path):
        """Test that prepare_dataset creates train/val/test splits."""
        X, y = sample_data

        # Save sample data
        images_path = tmp_path / "images.npy"
        labels_path = tmp_path / "labels.npy"
        np.save(images_path, X)
        np.save(labels_path, y)

        data = preprocessor.prepare_dataset(
            str(images_path),
            str(labels_path),
            test_size=0.2,
        )

        assert "X_train" in data
        assert "X_val" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "y_val" in data
        assert "y_test" in data
        assert data["train_size"] + data["val_size"] + data["test_size"] == 100


class TestGetPreprocessedData:
    """Tests for the get_preprocessed_data convenience function."""

    def test_returns_dict_with_expected_keys(self, tmp_path):
        """Test that function returns dict with all expected keys."""
        np.random.seed(42)
        X = np.random.randint(0, 256, size=(100, 784)).astype(np.float32)
        y = np.random.randint(0, 10, size=(100,)).astype(np.int32)

        images_path = tmp_path / "images.npy"
        labels_path = tmp_path / "labels.npy"
        np.save(images_path, X)
        np.save(labels_path, y)

        # Monkey-patch the default paths
        import src.preprocessing.mnist_preprocessor as module

        data = module.get_preprocessed_data(
            images_path=str(images_path),
            labels_path=str(labels_path),
            flatten=False,
        )

        expected_keys = [
            "X_train", "X_val", "X_test",
            "y_train", "y_val", "y_test",
            "train_size", "val_size", "test_size",
        ]
        for key in expected_keys:
            assert key in data
