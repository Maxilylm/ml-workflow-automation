"""FastAPI application for MNIST digit classification."""

import sys
sys.path.insert(0, ".")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from typing import List, Optional
import base64
from io import BytesIO
from pathlib import Path

app = FastAPI(
    title="MNIST Digit Classifier API",
    description="API for classifying handwritten digit images",
    version="1.0.0",
)

# Global model variable
model = None


class ImageInput(BaseModel):
    """Input model for image prediction."""
    pixels: List[float] = Field(
        ...,
        description="Flattened pixel values (784 values for 28x28 image)",
        min_length=784,
        max_length=784,
    )


class Base64ImageInput(BaseModel):
    """Input model for base64 encoded image."""
    image_base64: str = Field(..., description="Base64 encoded image")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predicted_digit: int = Field(..., description="Predicted digit (0-9)")
    confidence: float = Field(..., description="Prediction confidence")
    probabilities: List[float] = Field(
        ..., description="Probability for each digit (0-9)"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


@app.on_event("startup")
async def load_model():
    """Load the model on startup."""
    global model
    try:
        from tensorflow import keras
        model_path = "models/mnist_cnn.keras"
        if Path(model_path).exists():
            model = keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model not found at {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "MNIST Digit Classifier API",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: ImageInput):
    """
    Predict digit from pixel values.

    Args:
        input_data: ImageInput with 784 pixel values

    Returns:
        PredictionResponse with predicted digit and probabilities
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate and preprocess input
    pixels = np.array(input_data.pixels, dtype=np.float32)

    # Normalize if needed (assume input is 0-255)
    if pixels.max() > 1.0:
        pixels = pixels / 255.0

    # Reshape for CNN
    pixels = pixels.reshape(1, 28, 28, 1)

    # Predict
    probabilities = model.predict(pixels, verbose=0)[0]
    predicted_digit = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_digit])

    return PredictionResponse(
        predicted_digit=predicted_digit,
        confidence=confidence,
        probabilities=probabilities.tolist(),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(images: List[ImageInput]):
    """
    Predict digits for multiple images.

    Args:
        images: List of ImageInput objects

    Returns:
        BatchPredictionResponse with predictions for all images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(images) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")

    # Prepare batch
    batch = np.zeros((len(images), 28, 28, 1), dtype=np.float32)
    for i, img in enumerate(images):
        pixels = np.array(img.pixels, dtype=np.float32)
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        batch[i] = pixels.reshape(28, 28, 1)

    # Predict batch
    all_probabilities = model.predict(batch, verbose=0)

    # Format responses
    predictions = []
    for probs in all_probabilities:
        predicted_digit = int(np.argmax(probs))
        predictions.append(PredictionResponse(
            predicted_digit=predicted_digit,
            confidence=float(probs[predicted_digit]),
            probabilities=probs.tolist(),
        ))

    return BatchPredictionResponse(predictions=predictions)


@app.get("/model/info")
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "input_shape": model.input_shape[1:],
        "output_shape": model.output_shape[1:],
        "num_layers": len(model.layers),
        "total_params": model.count_params(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
