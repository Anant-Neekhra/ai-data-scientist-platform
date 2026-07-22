"""
Ensures the project root is on sys.path so ml_pipeline/ and frontend/
are importable from any file under app/pages/ -- Streamlit runs
nested page files without automatically adding the project root to
the import path the way running a script directly from the root
does. Import this FIRST, before any ml_pipeline or frontend imports,
in every page file and in streamlit_app.py itself.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))