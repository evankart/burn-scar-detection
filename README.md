# Wildfire Burn Scar Detection

Burn scar segmentation from Harmonized Landsat Sentinel-2 (HLS) satellite imagery using **Prithvi-EO-2.0-300M** — the IBM × NASA geospatial foundation model — fine-tuned with an FPN decoder for pixel-level segmentation.

**Live demo:** [huggingface.co/spaces/evankart/wildfire-burn-scar-detection](https://huggingface.co/spaces/evankart/wildfire-burn-scar-detection)

Trained on **92 wildfires** — 37 US fires across 5 states (CA, OR, AZ, NM, WA, CO) plus 55 global GlobFire/GWIS events spanning 6 biomes — with Optuna-tuned hyperparameters. Evaluated on 4 held-out fires at a fixed decision threshold of 0.5.

## Results (held-out test fires, threshold 0.5)

| Fire | Year | Biome / Type | Precision | Recall | IoU |
|---|---|---|---|---|---|
| Woolsey | 2018 | SoCal coastal chaparral | 89% | 85% | **77%** |
| Thomas | 2017 | CA coastal mountains | 95% | 73% | **70%** |
| Eaton | 2025 | SoCal urban interface | 96% | 77% | **75%** |
| Palisades | 2025 | SoCal urban interface | 97% | 71% | **69%** |
| **Macro** | | | **94%** | **77%** | **73%** |

All four test fires exceed 69% IoU. Palisades previously scored 39% IoU due to cloud/ocean pixels being misclassified as burn; pre-inference NDWI water masking resolved this. The fire burned through dense residential areas (Pacific Palisades, Altadena) where debris fields have a different spectral signature from wildland char — high precision (97%) reflects the model's conservatism in urban areas.

These four fires are held out of training entirely. The decision threshold (0.5) is fixed a priori — never tuned on the test fires.

## Architecture

```
HLS (6 bands, 30m) → z-score normalize → Prithvi-EO-2.0 ViT encoder → FPN decoder → burn mask
                     (2.0 pretrain stats)  (300M params, fine-tuned;    (3.8M params,
                                            pretrained on ~640k HLS)     trained from scratch)
```

- **Encoder** — [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M), a ViT-Large pretrained by IBM/NASA on HLS. Fine-tuned end-to-end with LLRD; multi-scale features tapped from encoder blocks `[5, 11, 17, 23]`.
- **Decoder** — FPN that fuses those features via top-down lateral connections, then upsamples 14×14 → 224×224 in four transposed-conv stages.
- **Labels** — auto-derived from dNBR (`dNBR = NBR_pre − NBR_post`, `NBR = (NIR − SWIR2)/(NIR + SWIR2)`, threshold 0.10). No manual annotation.
- **Data** — HLS surface reflectance from [NASA Earthdata](https://www.earthdata.nasa.gov/) via `earthaccess`.

## Methodology & design decisions

**Leakage discipline.** The decision threshold is fixed at 0.5 a priori — never tuned on test data. Hyperparameters (LR, loss weights) were tuned via Optuna on a held-out subset of *training* fires (carr_fire_2018, holy_2018, caldor_fire_2021, bootleg_fire_2021); test fires are never seen during search.

**Mono-temporal input.** The model sees a **single post-fire scene**. The encoder's `(B, C, T, H, W)` input is satisfied by replicating that one scene across the temporal frames. The pre-fire scene is used *only* to build the dNBR label, never as model input — so the deployed model needs just one image of a burned area.

**Normalization.** Bands are z-scored with Prithvi-EO-2.0's pretraining statistics (`src/data.normalize_bands`). No brightness gain is applied — Prithvi 2.0 was pretrained on a distribution that matches HLS LaSRC reflectance directly.

**Loss.** `CETverskyLoss` = weighted cross-entropy + soft Tversky on the burn class. The Tversky index `TI = TP / (TP + α·FP + β·FN)` recovers Dice at `α = β = 0.5`. Optuna-tuned: `α=0.37`, `class_weights=[0.55, 0.45]`.

**Hyperparameter search.** 7 Optuna trials (TPE sampler) on a 14-fire fast subset (10 train + 4 val), 5 epochs each, frozen encoder. Best trial: `lr=6.8e-4`, `backbone_lr_multiplier=0.016`, `tversky_alpha=0.37`, `class_weight_burn=0.46` (val burn-class IoU=0.655).

**Patch sampling.** Burn scars are rare, so patches are sampled to ensure burn coverage. A patch is kept if ≥ half its pixels are valid; remaining nodata is imputed to 0.

**Water masking (inference).** NDWI + MNDWI water mask applied *pre-inference* — water and ocean pixels are excluded from `valid_px` before the model scores any patch, preventing cloud/ocean false positives from accumulating burn probability. The HLS Fmask cloud flag is intentionally excluded from this mask: it triggers on smoke/haze over burned land and would zero out valid burn pixels.

**Sliding-window inference.** Scenes are tiled into overlapping 224-px windows (half-patch stride), with a final row/col anchored flush to the edge; overlaps are averaged.

**Configuration registry.** Every fire lives in one `data.fires` list in `configs/train_config.yaml`, each tagged `role: train | test`; `src.data.load_config` derives the splits so they can't drift.

## The over-prediction investigation (Prithvi 1.0)

Early experiments with a **frozen** Prithvi-EO-1.0-100M encoder over-predicted burn — high recall, mediocre precision (Woolsey recall 0.92 / precision 0.53). The root cause was that HLS LaSRC runs systematically **~1.4–1.9× darker** than the 1.0 pretraining distribution, pushing features toward the low-NIR "burn" signature. A fixed per-band brightness gain fixed this for 1.0. Prithvi 2.0, fine-tuned end-to-end on HLS, does not have this issue.

## Training details (finetune_v3)

- **Dataset:** 92 training fires — 37 US + 55 global GlobFire/GWIS events across 6 biomes
- **Schedule:** Frozen encoder epochs 1–2 (decoder warmup), unfrozen epoch 3+ with LLRD (decay=0.75, backbone LR multiplier=0.016), cosine LR, early stopping patience=5
- **Best checkpoint:** epoch 7 (val burn-class IoU=0.738)
- **Hardware:** AWS g5.xlarge (A10G 24GB), ~2.5 hrs

## Project structure

```
run_training.py          train the model (downloads HLS + Prithvi weights on first run)
run_inference.py         run on a region, save predictions + a sliding-window helper
app.py                   Streamlit entrypoint (Hugging Face Spaces)
src/
  data.py                HLS download, preprocessing, dNBR labels, patch dataset, load_config
  model.py               BurnScarModel (Prithvi 2.0 + FPN decoder)
  train.py               training loop (CE+Tversky loss, LLRD, gradual unfreeze)
  infer.py               on-demand inference for the custom-AOI tab
  utils.py               shared helpers (device selection, NDWI water mask)
  visualize.py           map overlays + comparison plots
  app/streamlit_app.py   interactive demo (held-out fires + live custom detection)
configs/
  train_config.yaml      single fire registry (role-tagged) + training settings
  finetune_config.yaml   fine-tune overlay (extends train_config.yaml)
  finetune_optuna_fast.yaml   fast 14-fire subset used for Optuna search
  finetune_optuna_config.yaml auto-generated best-HP config (extends finetune_config.yaml)
scripts/
  eval_sweep.py          evaluate checkpoints on the held-out fires (fixed threshold)
  optuna_search.py       Optuna hyperparameter search (7 trials × 5 epochs)
  globfire_to_config.py  convert GlobFire/GWIS CSV rows to train_config.yaml entries
  push_to_space.py       deploy all files to the HF Space
cloud/
  run_job.sh             self-terminating AWS GPU job (download → train → upload)
  launch_training.sh     full pipeline launcher (Optuna + retrain) via EC2 user data
  RUNBOOK.md             AWS fine-tune + HF Spaces deployment runbook
  space_README.md        README shown on the Hugging Face Space
notebooks/
  demo_analysis.ipynb    walkthrough notebook (renders on GitHub)
```

## Quick start

```bash
pip install -e .

# Train (downloads HLS + Prithvi weights on first run)
python run_training.py --config configs/finetune_config.yaml --experiment-name my_run

# Evaluate on the held-out fires (fixed threshold 0.5)
python scripts/eval_sweep.py --checkpoints checkpoints/finetune_v3/best_model.pt

# Run inference on one region
python run_inference.py --region woolsey_fire_2018

# Launch the app locally
streamlit run app.py
```

Requires a free [NASA Earthdata account](https://urs.earthdata.nasa.gov/users/new) — credentials go in `~/.netrc` or as `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` env vars.

## Training fires (100)

**California NorCal/Sierra:** August Complex (2020) · Mendocino Complex (2018) · SCU Lightning Complex (2020) · Caldor (2021) · LNU Lightning Complex (2020) · North Complex (2020) · Carr (2018) · Dixie (2021) · Antelope (2021) · Mosquito (2022) · Monument (2021) · River/Carmel (2020) · Camp Fire (2018) · Tubbs (2017) · Kincade (2019) · Glass (2020)

**California SoCal chaparral:** Bobcat (2020) · Holy (2018) · Apple (2020) · Cranston (2018) · Saddleridge (2019) · El Dorado (2020) · Valley (2020) · Lake (2020) · Blue Ridge (2020) · Bond (2020) · La Tuna (2017)

**Pacific Northwest:** Bootleg (2021, OR) · Pearl Hill (2020, WA) · Holiday Farm (2020, OR) · Beachie Creek (2020, OR)

**Colorado:** Cameron Peak (2020) · Calwood (2020) · Spring Creek (2018)

**Arizona:** Bighorn (2020) · Bush (2020) · Telegraph (2021)

**Global — South America (cerrado/savanna):** 8 GlobFire/GWIS events (2019–2021)

**Global — Sub-Saharan Africa (savanna):** 10 GlobFire/GWIS events (2017–2020)

**Global — Mediterranean (shrubland):** 6 GlobFire/GWIS events (2017–2021)

**Global — Australia (eucalyptus):** 10 GlobFire/GWIS events (2016–2020)

**Global — Canada (boreal):** 7 GlobFire/GWIS events (2017–2021)

**Global — Siberia/Russia (taiga):** 10 GlobFire/GWIS events (2016–2021)
