"""Thin HTTP client wrapper for the Streamlit frontend -- every page
calls the backend through this module instead of building requests
inline, same "one place owns how we talk to X" principle as the
logger/config modules."""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000/api/v1"


def upload_dataset(file) -> dict:
    response = requests.post(f"{API_BASE}/datasets/upload", files={"file": file})
    response.raise_for_status()
    return response.json()


def train_model(dataset_name: str, target_column: str, cv_folds: int, schema_override: dict | None = None) -> dict:
    payload = {
        "dataset_name": dataset_name,
        "target_column": target_column,
        "cv_folds": cv_folds,
        "schema_override": schema_override,
    }
    response = requests.post(f"{API_BASE}/train", json=payload)
    response.raise_for_status()
    return response.json()


def predict_batch(dataset_name: str, file) -> bytes:
    response = requests.post(
        f"{API_BASE}/predict/batch",
        params={"dataset_name": dataset_name},
        files={"file": file},
    )
    response.raise_for_status()
    return response.content