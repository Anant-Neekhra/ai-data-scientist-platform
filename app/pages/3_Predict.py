import _path_setup  # noqa: F401

import streamlit as st
from frontend.utils import predict_batch, list_trained_datasets

st.title("3. Predict")

try:
    available_datasets = list_trained_datasets()
except Exception as e:
    st.error(f"Could not load trained models: {e}")
    st.stop()

if not available_datasets:
    st.warning("No trained models yet. Go to the Train page first.")
    st.stop()

# Pre-select whatever was just trained in this session, if it's in the list.
default_index = 0
last_trained = st.session_state.get("trained_dataset")
if last_trained:
    from ml_pipeline.registry.model_registry import _sanitize_dataset_name
    sanitized = _sanitize_dataset_name(last_trained)
    if sanitized in available_datasets:
        default_index = available_datasets.index(sanitized)

dataset_name = st.selectbox("Dataset (with a trained model)", options=available_datasets, index=default_index)
predict_file = st.file_uploader("Upload CSV to predict on", type="csv", key="predict_upload")

if predict_file is not None and st.button("Run Predictions"):
    try:
        csv_bytes = predict_batch(dataset_name, predict_file)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.success("Predictions complete.")
    st.download_button(
        "Download predictions.csv", data=csv_bytes,
        file_name="predictions.csv", mime="text/csv",
    )