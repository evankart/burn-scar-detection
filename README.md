# Wildfire Burn Scar Detection

Burn scar segmentation from Harmonized Landsat Sentinel-2 (HLS) satellite imagery using **Prithvi-EO-1.0-100M** — the IBM × NASA geospatial foundation model — with an FPN decoder fine-tuned for pixel-level segmentation.

**Live demo:** [huggingface.co/spaces/evankart/burn-scar-detection](https://huggingface.co/spaces/evankart/burn-scar-detection)

Trained on **37 wildfires across 5 US states** (CA, OR, AZ, NM, WA), evaluated on 3 held-out fires in different biomes (Southern California chaparral, Colorado Rockies, Central California coast). Macro IoU **0.64** on the held-out test fires.

## Architecture

```
HLS (6 bands, 30m) → brightness gain → Prithvi-EO ViT encoder → FPN decoder → burn mask
                                         (100M params, frozen;         (3.6M params,
                                          pretrained on 640k HLS)       trained from scratch)
```

- **Encoder**: [Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M) — 12-layer ViT pretrained by IBM/NASA on HLS. Frozen at inference; features tapped from encoder blocks [2, 4, 7, 11] (0-indexed).
- **Decoder**: FPN-style decoder fusing multi-scale encoder features via top-down lateral connections, then upsampling 14×14 → 224×224 in four transposed-conv stages.
- **Labels**: Auto-derived from dNBR (dNBR = NBR_pre − NBR_post, threshold 0.10). No manual annotation.
- **Data**: HLS from [NASA Earthdata](https://www.earthdata.nasa.gov/) via `earthaccess`.

## Key findings

### HLS brightness correction
HLS surface reflectance (LaSRC + BRDF) runs **~1.4–1.9× darker** than the HLS distribution Prithvi was pretrained on. The frozen encoder reads dark input as burn-like, flooding predictions. A fixed per-band **brightness gain** — calibrated from training-fire medians against Prithvi's expected reflectance — corrects this before normalization. No test-fire leakage.

Effect: Woolsey IoU **0.53 → 0.73**, macro IoU **0.54 → 0.64**.

### What didn't work (documented honestly)
Asymmetric Tversky loss, threshold recalibration, encoder fine-tuning (overfit on small data), and SoCal hard-negative data all failed to beat the frozen + brightness-corrected baseline on the held-out fires. Full writeup: [`results/over_prediction_analysis.md`](results/over_prediction_analysis.md).

### Encoder fine-tuning (staged)
A **Prithvi-EO-2.0-300M** fine-tune is staged (`configs/finetune_config.yaml`): layer-wise LR decay + gradual unfreeze (`src/train.py`), a fire-based validation split, and the 37-fire dataset. The 2.0 normalization stats and any brightness gain are verified first by `scripts/band_stats_v2.py`. The cloud run is pending AWS GPU quota approval — infrastructure is ready (`cloud/RUNBOOK.md`). Not yet trained/evaluated; the deployed model remains the frozen 1.0 + gain baseline.

## Project structure

```
run_training.py          train the model (downloads HLS + Prithvi weights on first run)
run_inference.py         run on any region, save predictions
app.py                   Streamlit entrypoint (HF Spaces)
src/
  data.py                HLS download, preprocessing, patch dataset
  model.py               BurnScarModel (Prithvi encoder + FPN decoder)
  train.py               training loop (Tversky loss, LLRD, unfreeze)
  infer.py               on-demand inference for the custom-AOI tab
  visualize.py           map overlays + comparison plots
  app/
    streamlit_app.py     interactive demo (held-out fires + live custom detection)
configs/
  train_config.yaml      37-fire training configuration
  finetune_config.yaml   encoder fine-tune configuration (LLRD, gradual unfreeze)
scripts/
  calibrate_threshold.py honest threshold calibration on train fires only
  eval_sweep.py          evaluate checkpoints on held-out fires
  brightness_diag.py     brightness correction diagnostic
  push_to_space.py       push all files to HF Space
  build_demo_notebook.py build the demo/analysis notebook
cloud/
  run_job.sh             self-terminating AWS GPU job (download → train → upload → terminate)
  RUNBOOK.md             step-by-step cloud fine-tune runbook
  HF_SPACES_RUNBOOK.md   HF Spaces deployment runbook
notebooks/
  demo_analysis.ipynb    results notebook (pre-executed, renders on GitHub)
results/
  over_prediction_analysis.md  full investigation + brightness-fix writeup
```

## Quick start

```bash
pip install -e .

# Train (downloads HLS + Prithvi weights on first run; ~6 hr locally, ~40 min on A100)
python run_training.py --experiment-name my_run

# Run inference on held-out fires
python run_inference.py --region woolsey_fire_2018

# Launch the app locally
streamlit run app.py
```

Requires a free [NASA Earthdata account](https://urs.earthdata.nasa.gov/users/new) — credentials go in `~/.netrc` or as `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` env vars.

## Results (held-out test fires, fixed 0.5 threshold)

| Fire | Precision | Recall | IoU |
|---|---|---|---|
| Woolsey (2018, SoCal chaparral) | 76% | 94% | **73%** |
| East Troublesome (2020, CO Rockies) | 56% | 80% | **49%** |
| Thomas (2017, CA coast) | 96% | 71% | **69%** |
| **Macro** | **76%** | **82%** | **64%** |

## Training fires (37)

**California NorCal/Sierra:** August Complex (2020) · Mendocino Complex (2018) · SCU Lightning Complex (2020) · Caldor (2021) · LNU Lightning Complex (2020) · North Complex (2020) · Carr (2018) · Dixie (2021) · Antelope (2021) · Mosquito (2022) · Monument (2021) · River/Carmel (2020) · Camp Fire (2018) · Tubbs (2017) · Kincade (2019) · Glass (2020)

**California SoCal chaparral:** Bobcat (2020) · Holy (2018) · Apple (2020) · Cranston (2018) · Saddleridge (2019) · El Dorado (2020) · Valley (2020) · Lake (2020) · Blue Ridge (2020) · Bond (2020) · La Tuna (2017)

**Oregon:** Bootleg (2021) · Pearl Hill (2020, WA) · Holiday Farm (2020) · Beachie Creek (2020)

**Colorado:** Cameron Peak (2020) · Calwood (2020) · Spring Creek (2018)

**Arizona:** Bighorn (2020) · Bush (2020) · Telegraph (2021)

## Test fires (3, held out)

| Fire | Year | Location | Biome |
|---|---|---|---|
| Woolsey | 2018 | Southern California | Coastal chaparral |
| East Troublesome | 2020 | Colorado Rockies | Subalpine conifer |
| Thomas | 2017 | Ventura/Santa Barbara, CA | Coastal mountains |
