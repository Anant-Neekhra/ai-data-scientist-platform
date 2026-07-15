"""
Central configuration for the AI Data Scientist Platform.
All paths and constants are defined here so the rest of the
codebase never hardcodes a path or magic value.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file, if present
load_dotenv()

# Project root = two levels up from this file (config/settings.py -> project root)
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Key directories
DATASETS_DIR: Path = BASE_DIR / "datasets"
MODELS_DIR: Path = BASE_DIR / "models"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
LOGS_DIR: Path = BASE_DIR / "logs"

# Ensure directories exist at import time
for directory in (DATASETS_DIR, MODELS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# App-level constants
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2

# Read from environment, with sensible defaults
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")