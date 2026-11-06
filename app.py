import runpy
from pathlib import Path

# Execute the UI file fresh on every Streamlit rerun (instead of importing a cached
# module) so edits to it show up without restarting the server.
runpy.run_path(
    str(Path(__file__).parent / "src" / "app" / "streamlit_app.py"),
    run_name="__main__",
)
