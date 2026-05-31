# Deploy to Hugging Face Spaces (Streamlit, free CPU)

Hosts both tabs: precomputed held-out results, and live custom-area detection
(Prithvi runs on CPU — slow but works in the 16 GB Space).

## What you do (account-side)
1. **Create a Write token:** huggingface.co/settings/tokens → New token → type **Write**.
2. **Log in locally** (so the assistant can push/upload on your behalf):
   ```bash
   huggingface-cli login        # paste the Write token
   ```
3. **Set Earthdata secrets on the Space** (needed for the live-detection download).
   In the Space → **Settings → Variables and secrets → New secret**, add:
   - `EARTHDATA_USERNAME` = your NASA Earthdata login
   - `EARTHDATA_PASSWORD` = your NASA Earthdata password
   (These power `earthaccess` via the env-var auth the code now supports.)

## What the assistant automates (once you're logged in)
4. **Upload the model checkpoint + current predictions** to the dataset repo
   `evankart/burn-scar-detection-data` (the app downloads them at runtime):
   ```bash
   huggingface-cli upload evankart/burn-scar-detection-data \
     checkpoints/balanced_chaparral/best_model.pt \
     checkpoints/balanced_chaparral/best_model.pt --repo-type dataset
   for f in woolsey_fire_2018 east_troublesome_2020 thomas_fire_2017; do
     huggingface-cli upload evankart/burn-scar-detection-data \
       data/predictions/$f.npz predictions/$f.npz --repo-type dataset
   done
   ```
5. **Create the Space and push the app** (the assistant handles this):
   Uploads `app.py`, all `src/**/*.py`, `configs/train_config.yaml`, `requirements.txt`,
   and `cloud/space_README.md` → `README.md`. The `app_file: app.py` in the README
   front-matter tells Spaces to run the top-level entrypoint (which is in the repo root,
   so `from src.X import` always resolves without sys.path hacks).

## Notes
- **Latency:** first live detection on CPU is ~2–4 min (download + model). Cached after.
- **Memory:** Prithvi fits the 16 GB CPU Space. If you want fast inference later,
  switch the Space hardware to a GPU (or enable ZeroGPU) — no code change.
- **Cost:** free on the CPU tier.
