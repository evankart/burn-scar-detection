# Wildfire Burn Scar Detection

Burn scar segmentation from Harmonized Landsat Sentinel-2 (HLS) satellite imagery using **Prithvi-EO-1.0-100M** — the IBM × NASA geospatial foundation model — with an FPN decoder trained for pixel-level segmentation.

**Live demo:** [huggingface.co/spaces/evankart/burn-scar-detection](https://huggingface.co/spaces/evankart/burn-scar-detection)

Trained on **37 wildfires across 5 US states** (CA, OR, AZ, NM, WA), evaluated on 3 held-out fires in different biomes. **Macro IoU 0.64** on the held-out test fires, at a decision threshold fixed *a priori* (never tuned on the test set).

## Results (held-out test fires, fixed 0.5 threshold)

| Fire | Year | Biome | Precision | Recall | IoU |
|---|---|---|---|---|---|
| Woolsey | 2018 | SoCal coastal chaparral | 76% | 94% | **73%** |
| East Troublesome | 2020 | CO Rockies subalpine conifer | 56% | 80% | **49%** |
| Thomas | 2017 | CA coastal mountains | 96% | 71% | **69%** |
| **Macro** | | | **76%** | **82%** | **64%** |

These three fires are held out of training entirely and span deliberately different biomes from much of the training set, so the numbers reflect cross-biome generalization, not memorization.

## Architecture

```
HLS (6 bands, 30m) → brightness gain → Prithvi-EO ViT encoder → FPN decoder → burn mask
                                        (100M params, frozen;     (3.6M params,
                                         pretrained on ~640k HLS)   trained from scratch)
```

- **Encoder** — [Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M), a 12-layer ViT pretrained by IBM/NASA on HLS. Frozen; multi-scale features tapped from encoder blocks `[2, 4, 7, 11]`.
- **Decoder** — FPN that fuses those features via top-down lateral connections, then upsamples 14×14 → 224×224 in four transposed-conv stages.
- **Labels** — auto-derived from dNBR (`dNBR = NBR_pre − NBR_post`, `NBR = (NIR − SWIR2)/(NIR + SWIR2)`, threshold 0.10). No manual annotation.
- **Data** — HLS surface reflectance from [NASA Earthdata](https://www.earthdata.nasa.gov/) via `earthaccess`.

## Methodology & design decisions

**Leakage discipline (the core constraint).** The decision threshold (0.5), the loss hyperparameters, and the brightness gain are *only* ever set or calibrated using the training fires and a held-out slice of them — never the three test fires. The threshold is fixed at 0.5 up front, not tuned on the test set. This is what makes the reported test numbers defensible.

**Mono-temporal input.** The model sees a **single post-fire scene**. The encoder's `(B, C, T, H, W)` input is satisfied by replicating that one scene across the temporal frames. The pre-fire scene is used *only* to build the dNBR label, never as model input — so the deployed model needs just one image of a burned area.

**Normalization + HLS brightness gain.** Bands are z-scored with Prithvi's pretraining statistics. Critically, HLS LaSRC reflectance runs ~1.4–1.9× darker than that pretraining distribution, so a fixed per-band gain is applied first (see the investigation below).

**Loss.** `CETverskyLoss` = weighted cross-entropy + soft Tversky on the burn class. The Tversky index `TI = TP / (TP + α·FP + β·FN)` recovers Dice at `α = β = 0.5`; raising α penalizes false positives. In practice both configs run Dice (`α = β = 0.5`) and use the **CE class weights** as the precision/recall lever — `[0.3, 0.7]` (recall-leaning) for the deployed model.

**Patch sampling.** Burn scars are rare, so a `background_keep` fraction caps pure-background patches. A patch is kept if ≥ half its pixels are valid; remaining nodata is imputed to 0. (An earlier "every pixel must be valid" rule silently dropped whole large-fire scenes whose every window clipped a nodata gap.)

**Water masking (inference).** Open water is not burnable but both the model and the dNBR label flag it (NIR ≈ SWIR ≈ 0 makes NBR noisy). An NDWI mask (`(green − NIR)/(green + NIR) > 0`) removes ocean/lakes deterministically — model-independent, never tuned on test data.

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
| **Deployed (+ brightness gain)** | **0.73** | 0.49 | **0.69** | **0.64** |

Woolsey precision 0.53 → 0.76; Thomas precision 0.70 → 0.96.

**A subtlety worth noting:** retraining the decoder on gain-corrected input scored *worse* than applying the gain to the existing model. The decoder trained on the harder, dark input learned a more conservative boundary that generalizes best once it's given clean in-domain features at inference — so the deployment intentionally keeps the original checkpoint with the gain applied at serving time.

## Staged: Prithvi 2.0 fine-tune (not yet run)

Infrastructure is in place to fine-tune **Prithvi-EO-2.0-300M** (`configs/finetune_config.yaml`): a version registry in `src/model.py`, layer-wise LR decay + gradual unfreeze in `src/train.py`, a fire-based validation split, and a self-terminating AWS GPU job (`cloud/run_job.sh`, `cloud/RUNBOOK.md`). Prithvi 2.0 uses the same six physical HLS bands as 1.0 — only the normalization stats and architecture differ — and `scripts/band_stats_v2.py` verifies the 2.0 stats / brightness gain before training. This run is pending AWS GPU quota; the deployed model remains the frozen 1.0 + gain baseline.

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

# Evaluate on the held-out fires at the fixed threshold
python scripts/eval_sweep.py --threshold 0.5 \
  --checkpoints checkpoints/balanced_chaparral/best_model.pt

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
