import streamlit as st
from frontend.utils import train_model

st.title("2. Train Model")

if "dataset_name" not in st.session_state:
    st.warning("Upload a dataset first.")
    st.stop()

dataset_name = st.session_state["dataset_name"]
columns = st.session_state["columns"]

st.write(f"Dataset: **{dataset_name}**")
target_column = st.selectbox("Target column", options=columns)
cv_folds = st.slider("Cross-validation folds", min_value=2, max_value=10, value=5)

if st.button("Train Models"):
    with st.spinner("Comparing models... this can take a few minutes."):
        try:
            result = train_model(
                dataset_name, target_column, cv_folds,
                schema_override=st.session_state.get("schema_override"),
            )
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    st.success(f"Best model: **{result['best_model_name']}** (v{result['model_version']})")
    st.subheader("Leaderboard")
    st.dataframe(result["leaderboard"])
    st.session_state["trained_dataset"] = dataset_name