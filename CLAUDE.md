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
- **Results (finetune_v3, 92 fires, Optuna-tuned HPs, fixed inference masking):**
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

## Completed work

### Optuna hyperparameter search — DONE
7 trials (TPE), 14-fire fast subset, 5 epochs each, frozen encoder.
Best trial #4: `lr=2.01e-4`, `backbone_lr_multiplier=0.033`, `tversky_alpha=0.473`, `class_weight_burn=0.487` (val burn-class IoU=0.6578).
Artifacts in `checkpoints/optuna/` (study.pkl, best_params.yaml, plots).

### Fmask per-pixel cloud masking — DONE
`FMASK_BAD_BITS = 0b00010000` (snow only) in `src/data.py`. Cloud/shadow intentionally excluded — triggers on smoke/haze over burned land (Thomas fire recall dropped 0.73→0.26 when cloud pixels were masked).

### Global fire expansion — DONE (92 fires)
55 global events added via `scripts/globfire_to_config.py` from `data/globfire/*.csv` (one CSV per biome). All biomes represented: Australian eucalyptus, Canadian boreal, Siberian taiga, Mediterranean shrubland, South American cerrado, sub-Saharan savanna.

### finetune_v3 retrain — DONE
92 fires, 12 epochs, best epoch 7 (val mean_iou=0.8005). Checkpoint at `checkpoints/finetune_v3/best_model.pt` on HF dataset repo.

### BurnScars baseline comparison — DONE
IBM/NASA `Prithvi-EO-2.0-300M-BurnScars` (UNet decoder) evaluated on same 4 test fires.
Results: Woolsey 0.757 | Thomas 0.655 | Palisades 0.519 | Eaton 0.376 | **Macro 0.577** (vs our 0.727).
Implemented in `notebooks/demo_analysis.ipynb` via `terratorch.models.EncoderDecoderFactory`.

### Notebook (`notebooks/demo_analysis.ipynb`) — DONE
- Per-fire 3×2 visualization: RGB · GT | our model · our errors | baseline · baseline errors
- finetune_v3 training curves (loss + IoU, unfreeze epoch marked)
- Full Optuna trial table + param importance plots
- Pipeline comparison table (frozen → finetune_v2 → finetune_v3)

## Pending work

### 1. UI: multi-tile merge in custom AOI tab
When user draws a bounding box spanning two MGRS tiles, current code locks to one tile and silently drops the other half. Fix: detect when the drawn box overlaps multiple tiles, warn the user, and offer to merge scenes from both tiles into a mosaic. Implement in `src/app/streamlit_app.py` and `src/infer.py`.

### 2. Next retrain (if pursued)
After adding more fires or improving labels: retrain on AWS g5.xlarge using `cloud/run_job.sh`. Run Optuna again on the new fire set first. Evaluate on same 4 test fires at fixed 0.5 threshold. Update README results table.

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
- **HF Space:** `evankart/wildfire-burn-scar-detection` (Docker SDK, Streamlit on port 7860)
- **SDK change:** HF removed native Streamlit SDK; Space now uses Docker via `Dockerfile` in repo root.
- **`packages.txt`:** empty — rasterio bundles its own GDAL; system `libgdal-dev` caused conflicts.
- **NASA Earthdata credentials:** set `EARTHDATA_USERNAME` + `EARTHDATA_PASSWORD` as HF Space secrets (Settings → Variables and secrets). Required for custom AOI inference.

1. Make changes on `cloud-deploy` branch
2. `python scripts/push_to_space.py` — deploys code to HF Space (`evankart/wildfire-burn-scar-detection`)
3. `hf upload evankart/burn-scar-detection-data <checkpoint> <checkpoint> --repo-type dataset` — uploads checkpoint
4. Push updated prediction `.npz` files to `predictions/*.npz` in dataset repo (path the app downloads from)
5. HF Space loads checkpoint from dataset repo on cold start via `src/infer.py:load_model()`

## README update checklist (after any retraining)
- Results table (P/R/IoU per fire + macro)
- Model comparison table if baseline changes
- Architecture section if encoder/decoder changes
- Training fires count if fire list changes
