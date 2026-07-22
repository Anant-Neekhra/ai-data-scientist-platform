import _path_setup

import streamlit as st
from frontend.utils import predict_batch

st.title("3. Predict")

dataset_name = st.text_input(
    "Dataset name (must have a trained model)",
    value=st.session_state.get("trained_dataset", ""),
)
predict_file = st.file_uploader("Upload CSV to predict on", type="csv", key="predict_upload")

if predict_file is not None and dataset_name and st.button("Run Predictions"):
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