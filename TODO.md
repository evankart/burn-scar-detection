# Remaining work

## Done ✅
- Optuna hyperparameter search (7 trials × 5 epochs, TPE, frozen encoder)
- Global fire expansion: 55 GlobFire/GWIS events across 6 biomes → 100 train fires total
- Fmask per-pixel cloud masking in `src/data.py`
- Full retrain (finetune_v3) with best Optuna HPs on 100 fires
- Results: Woolsey 0.828 / Thomas 0.704 / Eaton 0.708 / Palisades 0.393 / **Macro 0.658**

## Pending

### A. Upload finetune_v3 to HF + deploy ☁️
```bash
huggingface-cli upload evankart/burn-scar-detection-data \
  checkpoints/finetune_v3/best_model.pt \
  checkpoints/finetune_v3/best_model.pt --repo-type dataset
python scripts/push_to_space.py
```
Re-run `run_inference.py` on the four test fires to regenerate `.npz` predictions for the app.

### B. Notebook improvements (`notebooks/demo_analysis.ipynb`)
- Remove brightness gain investigation section (Prithvi 1.0 artifact, no longer relevant)
- Add finetune_v3 training curves (from `checkpoints/finetune_v3/history.pt`)
- Add Optuna results: best trial, param importance plot, search trajectory (from `checkpoints/optuna/`)
- Add inference visualizations: side-by-side pred vs dNBR label on all 4 test fires

### C. Prithvi BurnScars model comparison (notebook)
Compare against `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` (IBM/NASA's
purpose-built burn scar model, UNet decoder). Run their model on the 4 held-out test
fires at fixed threshold 0.5. Report P/R/IoU alongside ours.
```bash
pip install terratorch   # on GPU instance or locally
```

### D. UI: multi-tile merge in custom AOI tab
When user draws a bounding box spanning two MGRS tiles, current code locks to one tile
and silently drops the other half. Fix: detect overlap, warn user, merge scenes from
both tiles into a mosaic. Implement in `src/app/streamlit_app.py` and `src/infer.py`.
