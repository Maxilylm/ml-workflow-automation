"""Unit tests for MNIST API."""

import pytest
import numpy as np
from fastapi.testclient import TestClient
import sys

sys.path.insert(0, ".")


class TestMNISTAPI:
    """Tests for MNIST API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from api.mnist_api import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns expected info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs_url" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_predict_validates_input_length(self, client):
        """Test that predict validates input has correct length."""
        # Too few pixels
        response = client.post(
            "/predict",
            json={"pixels": [0.0] * 100}
        )
        assert response.status_code == 422

        # Too many pixels
        response = client.post(
            "/predict",
            json={"pixels": [0.0] * 1000}
        )
        assert response.status_code == 422

    def test_batch_predict_validates_batch_size(self, client):
        """Test that batch predict validates batch size."""
        # Create too many images
        images = [{"pixels": [0.0] * 784} for _ in range(101)]
        response = client.post("/predict/batch", json=images)
        # Should fail with 400 (too many) or 422 (validation error)
        assert response.status_code in [400, 422]


class TestPredictionResponse:
    """Tests for prediction response format."""

    @pytest.fixture
    def valid_pixels(self):
        """Create valid pixel input."""
        np.random.seed(42)
        return list(np.random.randint(0, 256, size=784).astype(float))

    def test_prediction_response_has_required_fields(self):
        """Test PredictionResponse model."""
        from api.mnist_api import PredictionResponse

        response = PredictionResponse(
            predicted_digit=5,
            confidence=0.95,
            probabilities=[0.01] * 10,
        )
        assert response.predicted_digit == 5
        assert response.confidence == 0.95
        assert len(response.probabilities) == 10


class TestImageInput:
    """Tests for ImageInput validation."""

    def test_image_input_accepts_valid_pixels(self):
        """Test ImageInput accepts 784 pixels."""
        from api.mnist_api import ImageInput

        pixels = [0.0] * 784
        input_data = ImageInput(pixels=pixels)
        assert len(input_data.pixels) == 784

    def test_image_input_rejects_wrong_length(self):
        """Test ImageInput rejects wrong pixel count."""
        from api.mnist_api import ImageInput
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ImageInput(pixels=[0.0] * 100)
