"""MNIST Digit Classification Dashboard with Streamlit."""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys

sys.path.insert(0, ".")


st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="wide",
)


@st.cache_resource
def load_model():
    """Load the trained CNN model."""
    try:
        from tensorflow import keras
        model = keras.models.load_model("models/mnist_cnn.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_test_data():
    """Load test data for evaluation."""
    try:
        images = np.load("data/raw/mnist_images.npy")
        labels = np.load("data/raw/mnist_labels.npy")
        return images, labels
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


@st.cache_data
def load_metrics():
    """Load evaluation metrics."""
    metrics_path = "reports/figures/mnist/metrics.json"
    if Path(metrics_path).exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def create_sample_grid(images, labels, n_samples=25):
    """Create a grid of sample images."""
    indices = np.random.choice(len(images), n_samples, replace=False)
    fig = make_subplots(
        rows=5, cols=5,
        subplot_titles=[f"Label: {labels[i]}" for i in indices],
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    for idx, i in enumerate(indices):
        row = idx // 5 + 1
        col = idx % 5 + 1
        img = images[i].reshape(28, 28)
        fig.add_trace(
            go.Heatmap(z=img[::-1], colorscale='gray', showscale=False),
            row=row, col=col
        )
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(height=600, title_text="Sample MNIST Images")
    return fig


def create_class_distribution(labels):
    """Create class distribution plot."""
    unique, counts = np.unique(labels, return_counts=True)
    fig = px.bar(
        x=unique, y=counts,
        labels={'x': 'Digit', 'y': 'Count'},
        title='Class Distribution',
        color=counts,
        color_continuous_scale='Blues',
    )
    fig.update_layout(showlegend=False)
    return fig


def create_metrics_chart(metrics):
    """Create metrics bar chart."""
    per_class = metrics.get('per_class', {})
    if not per_class:
        return None

    digits = list(range(10))
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Precision',
        x=digits,
        y=per_class['precision'],
        marker_color='steelblue',
    ))
    fig.add_trace(go.Bar(
        name='Recall',
        x=digits,
        y=per_class['recall'],
        marker_color='darkorange',
    ))
    fig.add_trace(go.Bar(
        name='F1 Score',
        x=digits,
        y=per_class['f1'],
        marker_color='forestgreen',
    ))

    fig.update_layout(
        title='Per-Class Metrics',
        xaxis_title='Digit',
        yaxis_title='Score',
        barmode='group',
        yaxis_range=[0.9, 1.0],
    )
    return fig


def create_confusion_matrix_plot(cm):
    """Create confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=list(range(10)),
        y=list(range(10)),
        color_continuous_scale='Blues',
        title='Confusion Matrix',
    )
    fig.update_layout(height=500)
    return fig


def predict_digit(model, image):
    """Predict digit from image."""
    img = image.reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img, verbose=0)
    return np.argmax(prediction), prediction[0]


def main():
    st.title("🔢 MNIST Digit Classification Dashboard")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "Model Performance", "Interactive Prediction", "Data Explorer"]
    )

    # Load resources
    model = load_model()
    images, labels = load_test_data()
    metrics = load_metrics()

    if page == "Overview":
        st.header("Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        if images is not None:
            col1.metric("Total Samples", f"{len(images):,}")
            col2.metric("Image Size", "28x28 pixels")
            col3.metric("Classes", "10 (digits 0-9)")
            col4.metric("Features", "784")

        st.subheader("Sample Images")
        if images is not None and labels is not None:
            if st.button("Refresh Samples"):
                st.cache_data.clear()
            fig = create_sample_grid(images, labels)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Class Distribution")
        if labels is not None:
            fig = create_class_distribution(labels)
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Performance":
        st.header("Model Performance")

        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col2.metric("Precision (Macro)", f"{metrics['precision_macro']:.2%}")
            col3.metric("Recall (Macro)", f"{metrics['recall_macro']:.2%}")
            col4.metric("F1 Score (Macro)", f"{metrics['f1_macro']:.2%}")

            st.subheader("Per-Class Metrics")
            fig = create_metrics_chart(metrics)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Detailed Per-Class Report")
            per_class = metrics.get('per_class', {})
            if per_class:
                import pandas as pd
                df = pd.DataFrame({
                    'Digit': list(range(10)),
                    'Precision': per_class['precision'],
                    'Recall': per_class['recall'],
                    'F1 Score': per_class['f1'],
                    'Support': per_class['support'],
                })
                st.dataframe(df.style.format({
                    'Precision': '{:.4f}',
                    'Recall': '{:.4f}',
                    'F1 Score': '{:.4f}',
                }), use_container_width=True)
        else:
            st.warning("Metrics not available. Please run model evaluation first.")

        # Show confusion matrix if available
        cm_path = "reports/figures/mnist/confusion_matrix.png"
        if Path(cm_path).exists():
            st.subheader("Confusion Matrix")
            st.image(cm_path)

    elif page == "Interactive Prediction":
        st.header("Interactive Digit Prediction")

        if model is None:
            st.error("Model not loaded. Please train the model first.")
            return

        st.write("Select a random test image or draw your own digit.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Random Test Image")
            if images is not None:
                if st.button("Get Random Image"):
                    idx = np.random.randint(0, len(images))
                    st.session_state['random_idx'] = idx

                if 'random_idx' in st.session_state:
                    idx = st.session_state['random_idx']
                    img = images[idx]
                    true_label = labels[idx]

                    fig = px.imshow(
                        img.reshape(28, 28),
                        color_continuous_scale='gray',
                        title=f"True Label: {true_label}",
                    )
                    fig.update_layout(height=300, width=300)
                    st.plotly_chart(fig)

                    pred_label, probs = predict_digit(model, img)
                    st.write(f"**Predicted Label:** {pred_label}")
                    st.write(f"**Confidence:** {probs[pred_label]:.2%}")

                    st.subheader("Prediction Probabilities")
                    fig = px.bar(
                        x=list(range(10)),
                        y=probs,
                        labels={'x': 'Digit', 'y': 'Probability'},
                        color=probs,
                        color_continuous_scale='Blues',
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Model Information")
            if model:
                st.write("**Model Architecture:**")
                st.text(f"Input Shape: (28, 28, 1)")
                st.text(f"Output Classes: 10")

                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary[:15]))

    elif page == "Data Explorer":
        st.header("Data Explorer")

        if images is None or labels is None:
            st.error("Data not loaded.")
            return

        st.subheader("Explore by Digit")
        selected_digit = st.selectbox("Select Digit", list(range(10)))

        digit_images = images[labels == selected_digit]
        st.write(f"Found {len(digit_images):,} images of digit {selected_digit}")

        n_show = st.slider("Number of images to show", 5, 50, 20)
        indices = np.random.choice(len(digit_images), min(n_show, len(digit_images)), replace=False)

        cols = st.columns(5)
        for i, idx in enumerate(indices):
            with cols[i % 5]:
                fig = px.imshow(
                    digit_images[idx].reshape(28, 28),
                    color_continuous_scale='gray',
                )
                fig.update_layout(height=150, width=150, showlegend=False)
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pixel Intensity Analysis")
        avg_image = digit_images.mean(axis=0).reshape(28, 28)
        fig = px.imshow(
            avg_image,
            color_continuous_scale='hot',
            title=f"Average Image for Digit {selected_digit}",
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
