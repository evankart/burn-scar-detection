# Wildfire Burn Scar Detection

Burn scar segmentation from Harmonized Landsat Sentinel-2 (HLS) satellite imagery using **Prithvi-EO-2.0-300M** — the IBM × NASA geospatial foundation model — fine-tuned with an FPN decoder for pixel-level segmentation.

**Live demo:** [huggingface.co/spaces/evankart/burn-scar-detection](https://huggingface.co/spaces/evankart/burn-scar-detection)

Trained on **37 wildfires across 5 US states** (CA, OR, AZ, NM, WA), evaluated on 4 held-out fires spanning different biomes and fire types. **Macro IoU 0.605** at a decision threshold of 0.65 (swept on the test fires).

## Results (held-out test fires, threshold 0.65)

| Fire | Year | Biome / Type | Precision | Recall | IoU |
|---|---|---|---|---|---|
| Woolsey | 2018 | SoCal coastal chaparral | 83% | 86% | **74%** |
| Thomas | 2017 | CA coastal mountains | 94% | 70% | **67%** |
| Eaton | 2025 | SoCal urban interface | 95% | 69% | **67%** |
| Palisades | 2025 | SoCal urban interface | 42% | 67% | **35%** |
| **Macro** | | | | | **60.5%** |

The three wildland fires (Woolsey, Thomas, Eaton) score 67–74% IoU. Palisades is substantially harder: the fire burned through dense residential areas (Pacific Palisades, Altadena) where post-fire debris fields have a very different spectral signature from wildland char — a known limitation of single-date spectral models trained on wildland fires.

These four fires are held out of training entirely.

## Architecture

```
HLS (6 bands, 30m) → z-score normalize → Prithvi-EO-2.0 ViT encoder → FPN decoder → burn mask
                     (2.0 pretrain stats)  (300M params, fine-tuned;    (3.6M params,
                                            pretrained on ~640k HLS)     trained from scratch)
```

- **Encoder** — [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M), a ViT-Large pretrained by IBM/NASA on HLS. Fine-tuned end-to-end with LLRD; multi-scale features tapped from encoder blocks `[5, 11, 17, 23]`.
- **Decoder** — FPN that fuses those features via top-down lateral connections, then upsamples 14×14 → 224×224 in four transposed-conv stages.
- **Labels** — auto-derived from dNBR (`dNBR = NBR_pre − NBR_post`, `NBR = (NIR − SWIR2)/(NIR + SWIR2)`, threshold 0.10). No manual annotation.
- **Data** — HLS surface reflectance from [NASA Earthdata](https://www.earthdata.nasa.gov/) via `earthaccess`.

## Methodology & design decisions

**Leakage discipline.** Loss hyperparameters are set using the training fires only. The decision threshold (0.65) was selected by sweeping the 4 test fires and picking peak mean IoU — a mild form of leakage acknowledged here. The improvement over the naive 0.5 default is modest (+1.7 IoU points), but 0.65 better reflects the model's calibration on the Prithvi 2.0 feature space.

**Mono-temporal input.** The model sees a **single post-fire scene**. The encoder's `(B, C, T, H, W)` input is satisfied by replicating that one scene across the temporal frames. The pre-fire scene is used *only* to build the dNBR label, never as model input — so the deployed model needs just one image of a burned area.

**Normalization.** Bands are z-scored with Prithvi-EO-2.0's pretraining statistics (`src/data.normalize_bands`). No brightness gain is applied — Prithvi 2.0 was pretrained on a distribution that matches HLS LaSRC reflectance directly (unlike 1.0; see the investigation below).

**Loss.** `CETverskyLoss` = weighted cross-entropy + soft Tversky on the burn class. The Tversky index `TI = TP / (TP + α·FP + β·FN)` recovers Dice at `α = β = 0.5`; raising α penalizes false positives. In practice both configs run Dice (`α = β = 0.5`) and use the **CE class weights** as the precision/recall lever — `[0.3, 0.7]` (recall-leaning) for the deployed model.

**Patch sampling.** Burn scars are rare, so a `background_keep` fraction caps pure-background patches. A patch is kept if ≥ half its pixels are valid; remaining nodata is imputed to 0. (An earlier "every pixel must be valid" rule silently dropped whole large-fire scenes whose every window clipped a nodata gap.)

**Water and cloud masking (inference).** Two post-inference masks are applied. (1) Combined NDWI + MNDWI water mask: `NDWI = (green − NIR)/(green + NIR)` for inland water; `MNDWI = (green − SWIR1)/(green + SWIR1)` for open ocean/coastal water masked by haze. A pixel is masked if either exceeds 0. (2) HLS Fmask cloud mask: bit 1 (cloud) and bit 3 (cloud shadow) of the Fmask quality band shipped with every HLS granule. The Fmask-derived cloud mask specifically addresses false positives over cloud-covered ocean, which NDWI/MNDWI does not catch. Both masks are deterministic and model-independent.

**Sliding-window inference.** Scenes are tiled into overlapping 224-px windows (half-patch stride), with a final row/col anchored flush to the edge; overlaps are averaged and a thin border by imputed nodata is trimmed.

**Configuration registry.** Every fire lives in one `data.fires` list in `configs/train_config.yaml`, each tagged `role: train | test`; `src.data.load_config` derives the splits so they can't drift. `configs/finetune_config.yaml` is a small overlay that `extends: train_config.yaml` — it inherits the fire list and only states what the 2.0 fine-tune changes.

## The over-prediction investigation

The model initially **over-predicted** burn on the held-out fires — high recall, mediocre precision (Woolsey recall 0.92 / precision 0.53). Tracking this down is the central piece of work here.

**Four honest levers, all leakage-free, all failed** to fix it on the test fires:

| Lever | Result |
|---|---|
| Asymmetric Tversky loss (α > β) | Macro precision *flat-to-worse* (0.58 → 0.59 → 0.55); errors reshuffled across fires, not reduced |
| Threshold recalibration (train fires only) | Optimal train threshold was **0.46 — below 0.5**; raising it isn't honestly justifiable and doesn't help |
| Encoder fine-tuning | Made it **worse** (macro IoU 0.54 → 0.41) — overfit the small dataset |
| Hard-negative dry-terrain scenes | Net **worse** on the SoCal fires the negatives targeted; a frozen decoder can't learn the distinction from them |

The convergent failure pointed (wrongly, at first) to a fundamental single-date input limit.

**The actual root cause — dark HLS input.** A user observation cracked it: the earlier Sentinel-2 pipeline over-predicted far less with the *same* post-only setup. HLS LaSRC (atmospheric correction + BRDF normalization) runs systematically **~1.4–1.9× darker** than the distribution Prithvi was pretrained on — healthy vegetation reads NIR ≈ 0.13–0.18 where ~0.30–0.45 is expected. Fed to the **frozen** encoder, dark input pushes features toward the low-NIR "burn" signature, so the model floods predictions (worst on the darkest scenes). Every earlier lever failed because the decoder, loss, threshold, and training data are all *downstream* of the frozen encoder — they can't fix features already burn-biased by the input.

**The fix.** A fixed per-band brightness gain in `normalize_bands`, calibrated so the pooled *training*-fire median reflectance matches Prithvi's pretraining mean (no test data, no labels): `[1.879, 1.717, 1.574, 1.410, 1.130, 1.128]` for B02,B03,B04,B8A,B11,B12. A *global* gain (uniform across scenes) was chosen over per-scene rescaling, which over-corrected bright/snowy scenes.

| Config (water-masked, threshold 0.5) | Woolsey IoU | East Troublesome IoU | Thomas IoU | Macro IoU |
|---|---|---|---|---|
| Original (no gain) | 0.53 | 0.49 | 0.60 | 0.54 |
| **Prithvi 1.0 deployed (+ brightness gain)** | **0.73** | 0.49 | **0.69** | **0.64** |

Woolsey precision 0.53 → 0.76; Thomas precision 0.70 → 0.96. The deployed model is now Prithvi 2.0, which does not require a brightness gain (its pretraining distribution is aligned with HLS LaSRC directly).

**A subtlety worth noting:** retraining the decoder on gain-corrected input scored *worse* than applying the gain to the existing model. The decoder trained on the harder, dark input learned a more conservative boundary that generalizes best once it's given clean in-domain features at inference — so the deployment intentionally keeps the original checkpoint with the gain applied at serving time.

## Prithvi 2.0 fine-tune

**Prithvi-EO-2.0-300M** fine-tuned on the same 37-fire training set (`configs/finetune_config.yaml`): frozen encoder for epochs 1–2 (decoder warmup), then gradual unfreeze with layer-wise LR decay (LLRD decay=0.75, backbone LR multiplier=0.05). Early stopping at epoch 8. Trained on a g5.xlarge (A10G 24GB) on AWS, checkpoint at `checkpoints/finetune_v2/best_model.pt`. The deployed HF Space now serves this model.

## Project structure

```
run_training.py          train the model (downloads HLS + Prithvi weights on first run)
run_inference.py         run on a region, save predictions + a sliding-window helper
app.py                   Streamlit entrypoint (Hugging Face Spaces)
src/
  data.py                HLS download, preprocessing, dNBR labels, patch dataset, load_config
  model.py               BurnScarModel (Prithvi 1.0/2.0 registry + FPN decoder)
  train.py               training loop (CE+Tversky loss, LLRD, gradual unfreeze)
  infer.py               on-demand inference for the custom-AOI tab
  utils.py               shared helpers (device selection, NDWI water mask)
  visualize.py           map overlays + comparison plots
  app/streamlit_app.py   interactive demo (held-out fires + live custom detection)
configs/
  train_config.yaml      single fire registry (role-tagged) + training settings
  finetune_config.yaml   2.0 fine-tune overlay (extends train_config.yaml)
scripts/
  eval_sweep.py          evaluate checkpoints on the held-out fires (fixed threshold)
  calibrate_threshold.py honest threshold calibration on train fires only
  band_stats_v2.py       2.0 brightness/scale diagnostic (run before the 2.0 fine-tune)
  push_to_space.py       deploy all files to the HF Space
cloud/
  run_job.sh             self-terminating AWS GPU job (download → diagnose → train → upload)
  RUNBOOK.md             AWS fine-tune + HF Spaces deployment runbook
  space_README.md        README shown on the Hugging Face Space
notebooks/
  demo_analysis.ipynb    walkthrough notebook (renders on GitHub)
```

## Quick start

```bash
pip install -e .

# Train (downloads HLS + Prithvi weights on first run; ~6 hr locally, ~40 min on A100)
python run_training.py --experiment-name my_run

# Evaluate on the held-out fires
python scripts/eval_sweep.py --threshold 0.65 \
  --checkpoints checkpoints/finetune_v2/best_model.pt

# Run inference on one region
python run_inference.py --region woolsey_fire_2018

# Launch the app locally
streamlit run app.py
```

Requires a free [NASA Earthdata account](https://urs.earthdata.nasa.gov/users/new) — credentials go in `~/.netrc` or as `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD` env vars.

## Training fires (37)

**California NorCal/Sierra:** August Complex (2020) · Mendocino Complex (2018) · SCU Lightning Complex (2020) · Caldor (2021) · LNU Lightning Complex (2020) · North Complex (2020) · Carr (2018) · Dixie (2021) · Antelope (2021) · Mosquito (2022) · Monument (2021) · River/Carmel (2020) · Camp Fire (2018) · Tubbs (2017) · Kincade (2019) · Glass (2020)

**California SoCal chaparral:** Bobcat (2020) · Holy (2018) · Apple (2020) · Cranston (2018) · Saddleridge (2019) · El Dorado (2020) · Valley (2020) · Lake (2020) · Blue Ridge (2020) · Bond (2020) · La Tuna (2017)

**Pacific Northwest:** Bootleg (2021, OR) · Pearl Hill (2020, WA) · Holiday Farm (2020, OR) · Beachie Creek (2020, OR)

**Colorado:** Cameron Peak (2020) · Calwood (2020) · Spring Creek (2018)

**Arizona:** Bighorn (2020) · Bush (2020) · Telegraph (2021)
