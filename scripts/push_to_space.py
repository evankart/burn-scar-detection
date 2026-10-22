"""Push all necessary files to HF Space and verify the manifest."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()
space = "evankart/wildfire-burn-scar-detection"

uploads = [
    ("Dockerfile", "Dockerfile"),
    ("requirements.txt", "requirements.txt"),
    ("packages.txt", "packages.txt"),
    ("app.py", "app.py"),
    ("cloud/space_README.md", "README.md"),
    ("src/__init__.py", "src/__init__.py"),
    ("src/utils.py", "src/utils.py"),
    ("src/visualize.py", "src/visualize.py"),
    ("src/model.py", "src/model.py"),
    ("src/data.py", "src/data.py"),
    ("src/train.py", "src/train.py"),
    ("src/infer.py", "src/infer.py"),
    ("src/app/__init__.py", "src/app/__init__.py"),
    ("src/app/streamlit_app.py", "src/app/streamlit_app.py"),
    ("configs/train_config.yaml", "configs/train_config.yaml"),
]

# add perimeters
for f in os.listdir("data/perimeters"):
    uploads.append((f"data/perimeters/{f}", f"data/perimeters/{f}"))

# single commit so the Space rebuilds once, not once per file
ops = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=local)
       for local, remote in uploads]
api.create_commit(repo_id=space, repo_type="space", operations=ops,
                  commit_message=sys.argv[1] if len(sys.argv) > 1 else "Update Space")
for _, remote in uploads:
    print(f"  {remote}")

print("\n=== Space manifest ===")
files = sorted(f for f in api.list_repo_files(space, repo_type="space")
               if not f.startswith("."))
for f in files:
    print(f"  {f}")
print(f"\nTotal: {len(files)} files — done.")
