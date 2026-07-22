"""Upload page: upload a CSV, review/correct the auto-detected schema,
and hand off to the Train page."""

import streamlit as st
import pandas as pd

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from frontend.utils import upload_dataset

st.title("1. Upload Dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        upload_result = upload_dataset(uploaded_file)
    except Exception as e:
        st.error(f"Upload failed: {e}")
        st.stop()

    st.success(f"Uploaded '{upload_result['dataset_name']}' — {upload_result['n_rows']} rows, {upload_result['n_columns']} columns")

    if upload_result["validation_warnings"]:
        st.warning("Warnings: " + "; ".join(upload_result["validation_warnings"]))

    # Re-read the uploaded file locally to detect schema for display --
    # this is read-only analysis, so we call ml_pipeline directly rather
    # than adding a backend endpoint solely for the frontend to consume.
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    detected_schema = SchemaDetector().detect_feature_types(df)

    st.subheader("Detected Schema — review and correct if needed")
    st.caption(
        "Automatic detection can be wrong, especially for ambiguous columns "
        "(e.g. numeric codes that are really categories, or IDs that "
        "coincidentally look like real features). Override anything below."
    )

    schema_options = [t.value for t in FeatureType]
    overrides = {}
    for col, ftype in detected_schema.items():
        chosen = st.selectbox(
            col, options=schema_options,
            index=schema_options.index(ftype.value),
            key=f"schema_{col}",
        )
        overrides[col] = chosen

    st.session_state["dataset_name"] = upload_result["dataset_name"]
    st.session_state["columns"] = list(df.columns)
    st.session_state["schema_override"] = overrides

    st.info("Schema saved. Go to the **Train** page in the sidebar to continue.")