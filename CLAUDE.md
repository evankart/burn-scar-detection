# Burn Scar Detection — Claude Code Instructions

## Project overview
Wildfire burn scar segmentation from HLS (Harmonized Landsat Sentinel-2) imagery using Prithvi-EO-2.0-300M (IBM/NASA) with an FPN decoder. Deployed at huggingface.co/spaces/evankart/wildfire-burn-scar-detection. All data is HLS (`HLSS30.v2.0`) via `earthaccess` — no Sentinel-2/Planetary Computer code remains.

## Hard constraints (never violate)
- **Never tune the decision threshold on test fires.** Threshold is fixed at 0.5 a priori. Test fires are `woolsey_fire_2018`, `thomas_fire_2017`, `palisades_fire_2025`, `eaton_fire_2025`.
- **Never commit secrets.** Earthdata credentials, AWS keys, HF tokens go in env vars or `~/.netrc` only.
- **Use `cloud-deploy` branch** for all changes, not `main`.
- **Co-author commits:** `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

## Current model
- **Checkpoint:** `checkpoints/finetune_v3/best_model.pt` (on HF dataset repo `evankart/burn-scar-detection-data`)
- **Encoder:** Prithvi-EO-2.0-300M (ViT-Large, embed_dim=1024, depth=24, frozen epochs 1-2 then unfrozen)
- **Decoder:** FPN, 3.8M trainable params
- **Bands:** B02, B03, B04, B8A, B11, B12 (HLS Sentinel naming)
- **Results (finetune_v3, 100 fires, Optuna-tuned HPs, fixed inference masking):**
  - Woolsey 0.890/0.850/**0.769** | Thomas 0.947/0.729/**0.700** | Palisades 0.969/0.710/**0.694** | Eaton 0.958/0.771/**0.746** | **Macro IoU 0.727**
  - Palisades improved substantially after fixing pre-inference water masking (was 0.393)
- No brightness gain applied (fine-tuning adapted encoder to HLS distribution)

## Architecture
```
HLS (6 bands, 30m) → normalize_bands → Prithvi-EO-2.0 ViT encoder → FPN decoder → burn mask
```
- Single post-fire scene replicated across temporal frames to satisfy `(B, C, T, H, W)` input
- Pre-fire scene used only for dNBR labels, never as model input
- NDWI water mask applied at inference (not tuned on test data)
- Sliding window inference (224px, half-patch stride)

## Key design decisions
- **dNBR threshold: 0.10** — USGS boundary between unburned and low-severity burn (Keeley 2009). Do not change without justification. Low-severity pixels (dNBR 0.10–0.27) are the primary source of label noise.
- **Cloud cover:** post-fire scenes capped at 20% cloud cover; pre-fire uses progressive fallback (20%/28d → 30%/60d → 40%/90d). Per-pixel Fmask masking implemented in `src/data.py` (`FMASK_BAD_BITS = snow only`; cloud/shadow intentionally excluded from NaN masking to preserve burn signal under haze).
- **Multi-tile fires:** add `allow_multitile: true` to any fire config whose AOI spans MGRS tile boundaries (e.g. thomas_fire_2017). Without it, the downloader locks to the first tile returned.
- **No brightness gain for 2.0** — fine-tuning already adapted the encoder to the HLS distribution. The 1.0 gain (`[1.879, 1.717, 1.574, 1.410, 1.130, 1.128]`) is no longer used.
- **Config inheritance:** `finetune_config.yaml` extends `train_config.yaml` via `extends:` key. All fires live in `train_config.yaml` tagged `role: train|test`. Never duplicate fires across configs.
- **Frozen encoder forward:** wrapped in `torch.no_grad()` when frozen to save GPU memory. Training loop uses streaming confusion matrix (TP/FP/FN/TN counters) not accumulated predictions — critical for RAM on 16GB instances.

## Fire registry
All fires defined in `configs/train_config.yaml` under `data.fires`, each with `role: train|test`. `load_config()` in `src/data.py` derives splits. Never hardcode splits elsewhere.

**Current training fires: 92** — 37 US fires (CA, OR, AZ, NM, WA, CO) plus 55 global GlobFire/GWIS events across 6 biomes (S. American cerrado, sub-Saharan savanna, Mediterranean shrubland, Australian eucalyptus, Canadian boreal, Siberian taiga). Global events were selected from `data/globfire/*.csv` via `scripts/globfire_to_config.py` (names tagged `gwis_<biome>_<year>_<id>`, all `role: train`).
**Test fires (never train on, never tune threshold on): 4** — palisades_fire_2025, eaton_fire_2025, woolsey_fire_2018, thomas_fire_2017.

## Pending work (priority order)

### 1. Hyperparameter search (Optuna, ~10 trials, ~$15 on AWS)
Run before next full retraining. Tune:
- `learning_rate`: 1e-4 to 1e-3
- `backbone_lr_multiplier`: 0.01 to 0.1
- `tversky_alpha`: 0.3 to 0.7
- `class_weights[1]` (burn class): 0.4 to 0.7

Metric: val IoU on carr_fire_2018 + holy_2018 (fire-based val split, never test fires).
Implement in `scripts/optuna_search.py`. Each trial trains to early stopping. Upload best trial checkpoint to S3. Do not tune on test fires.

### 2. Fmask per-pixel cloud masking
Implement before adding global fires. HLS QA band has per-pixel bit flags:
- Bit 1: cirrus, Bit 2: cloud, Bit 8: cloud shadow, Bit 16: snow, Bit 32: water
Apply in `src/data.py` `load_and_merge_scenes()` — download QA band alongside spectral bands, mask flagged pixels to nodata before computing dNBR or saving cache. Without this, cloudy pixels can contaminate labels in tropical/boreal regions with persistent cloud.

### 3. Global fire expansion (target: ~100 fires total) — DONE (92 fires)
Source: **GWIS / GlobFire** — single source for both US and global fires.
Filter: year ≥ 2015 (HLS era), burned area > 10,000 ha, biome diversity required.
55 global events were added from `data/globfire/*.csv` (one CSV per biome) via
`python scripts/globfire_to_config.py --csv <biome>.csv --tag <tag> --min-sep-km 50 --max 10 --append-to configs/train_config.yaml`.
Imagery still has to be downloaded in-region on AWS (`run_training.py --download-only`) before the retrain.

Target biomes (now represented):
- Australia — eucalyptus/dry sclerophyll
- Canada — boreal forest
- Siberia/Russia — taiga
- Mediterranean — shrubland (Spain, Greece, Portugal)
- South America — cerrado/savanna
- Sub-Saharan Africa — savanna

Keep current 37 US fires. Add ~63 global fires. Implement Fmask (item 2) before downloading global fires.

For each new fire, add to `configs/train_config.yaml` with:
```yaml
- name: fire_name_year
  lat: <center_lat>
  lon: <center_lon>
  buffer_km: <radius>
  post_fire_date: "YYYY-MM-DD"
  pre_fire_date: "YYYY-MM-DD"
  role: train
```

### 4. Retrain with best hyperparams + global fires
After items 1–3: retrain on AWS g5.xlarge using `cloud/run_job.sh`. Update `finetune_config.yaml` with best Optuna hyperparams. Evaluate on same 3 test fires at fixed 0.5 threshold. Update README results table.

### 5. Notebook improvements (`notebooks/demo_analysis.ipynb`)
- **Remove** brightness gain investigation section (solved, no longer relevant)
- **Fill in** 1.0 vs 2.0 comparison with real numbers (note: pipeline comparison not controlled ablation — encoder size, fine-tuning, batch size all changed)
- **Add** finetune_v2 training curves (loss + IoU over epochs, from `checkpoints/finetune_v2/history.pt`)
- **Add** Optuna hyperparameter search results (best trial, param importance plot, search trajectory) — add after search completes
- **Add** more inference visualizations: side-by-side pred vs dNBR label on all 3 test fires, severity overlay

### 6. Prithvi BurnScars model comparison (notebook)
Add a notebook section comparing our model against `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` (IBM/NASA's purpose-built burn scar model, UNet decoder). Run their model on our 3 held-out test fires at fixed 0.5 threshold. Report P/R/IoU alongside ours.

### 7. UI: multi-tile merge in custom AOI tab
When user draws a bounding box spanning two MGRS tiles, current code locks to one tile and silently drops the other half. Fix: detect when the drawn box overlaps multiple tiles, warn the user, and offer to merge scenes from both tiles into a mosaic. Implement in `src/app/streamlit_app.py` and `src/infer.py`.

## AWS / deployment
- **Instance:** g5.xlarge (A10G 24GB GPU, 16GB RAM), us-west-2
- **Critical:** use `batch_size: 2` for 2.0 (ViT-Large OOMs at 4 on A10G)
- **Critical:** training loop must use streaming confusion matrix (not `all_preds` list accumulation) — RAM OOM otherwise on 16GB instance
- **Self-terminate:** always run with `SELF_TERMINATE=1 bash cloud/run_job.sh`
- **S3 bucket:** `s3://burn-scar-detection` (us-west-2)
- **Key pair:** `burn-scar-detection` (pem file must be kept safe — not recoverable if lost)
- **EC2 Instance Connect:** available as SSH fallback if pem is lost (Ubuntu 24.04 AMI supports it)
- **AWS credentials on instance:** not auto-provisioned — run `aws configure` manually after SSH if S3 upload fails
- **pip install:** always `source /opt/pytorch/bin/activate` first; `h5py` must be installed alongside `h5netcdf`

## Deployment pipeline
1. Make changes on `cloud-deploy` branch
2. `python scripts/push_to_space.py` — deploys code to HF Space
3. `hf upload evankart/burn-scar-detection-data <checkpoint> <checkpoint> --repo-type dataset` — uploads checkpoint
4. HF Space loads checkpoint from dataset repo on cold start via `src/infer.py:load_model()`

## README update checklist (after any retraining)
- Results table (P/R/IoU per fire + macro)
- Model comparison table if baseline changes
- Architecture section if encoder/decoder changes
- Training fires count if fire list changes
