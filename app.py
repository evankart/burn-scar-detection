"""
Top-level Streamlit entrypoint for HF Spaces (and local use).
Lives at the repo root so sys.path is already correct — no path manipulation needed.
Delegates everything to the actual app module.
"""
from src.app.streamlit_app import main

main()
