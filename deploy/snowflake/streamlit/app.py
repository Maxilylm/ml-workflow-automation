"""Streamlit in Snowflake Dashboard Template."""

import streamlit as st
from snowflake.snowpark.context import get_active_session

# Get Snowflake session
session = get_active_session()

# Page configuration
st.set_page_config(
    page_title="ML Model Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("ML Model Dashboard")
st.markdown("Template for Streamlit in Snowflake")

# Sidebar for prediction input
st.sidebar.header("Model Input")

# Example input fields - customize for your model
feature_1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
feature_2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
feature_3 = st.sidebar.selectbox("Feature 3", options=[1, 2, 3])

# Make prediction
if st.sidebar.button("Predict", type="primary"):
    try:
        query = f"""
        SELECT MODELS.PREDICT_TEMPLATE({feature_1}, {feature_2}, {feature_3}) as prediction
        """
        result = session.sql(query).collect()[0][0]

        st.subheader("Prediction Result")
        if result >= 0:
            st.success(f"Prediction: {result}")
        else:
            st.error("Prediction error - model may not be deployed")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Make sure the model is deployed to the registry")

# Main content area
tab1, tab2 = st.tabs(["Model Performance", "Explore Data"])

with tab1:
    st.header("Model Performance Metrics")

    try:
        metrics_query = """
        SELECT * FROM ANALYTICS.MODEL_METRICS
        ORDER BY CREATED_AT DESC LIMIT 1
        """
        metrics_df = session.sql(metrics_query).to_pandas()

        if not metrics_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics_df['ACCURACY'].iloc[0]:.2%}")
            col2.metric("Precision", f"{metrics_df['PRECISION'].iloc[0]:.2%}")
            col3.metric("Recall", f"{metrics_df['RECALL'].iloc[0]:.2%}")
            col4.metric("F1 Score", f"{metrics_df['F1'].iloc[0]:.2%}")
        else:
            st.info("No model metrics available yet. Train and register a model first.")

    except Exception as e:
        st.warning("Model metrics not available")
        st.info("Run the training pipeline to generate metrics")

with tab2:
    st.header("Data Explorer")
    st.info("Add your data exploration queries here")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit in Snowflake | Data Science Automation Framework")
