# Methodology & Design Rationale

Design decisions and the reasoning behind them. Code comments stay minimal and
point here; this file (and `results/over_prediction_analysis.md`) hold the
"why". Empirical results are in `results/`.

## Leakage discipline (non-negotiable)

The decision threshold, loss hyperparameters (Tversky alpha/beta, class
weights), and the brightness gain are **only ever calibrated on the held-out
*training* fires' validation split** — never on the three test fires
(`woolsey_fire_2018`, `east_troublesome_2020`, `thomas_fire_2017`). This keeps
the reported test-fire numbers defensible.

## Configuration registry

`configs/train_config.yaml` is the single source of truth: every fire lives in
one `data.fires` list, each tagged with `role: train | test | negative`.
`src.data.load_config` derives the `train_regions` / `test_regions` /
`negative_regions` lists from those roles so the splits can never drift.
`configs/finetune_config.yaml` is generated from it by
`scripts/make_finetune_config.py` and uses the explicit derived lists (also
accepted by `load_config`, for backward compatibility).

## Prithvi encoder versions

| | 1.0-100M | 2.0-300M |
|---|---|---|
| Arch | ViT-Base, embed 768, depth 12 | ViT-Large, embed 1024, depth 24 |
| Frames | 3 | 4 |
| Pretrain scenes | ~640k HLS | ~4.2M HLS |
| Bands | B02,B03,B04,B8A,B11,B12 | B02,B03,B04,B05,B06,B07 |
| FPN feature layers | [2,4,7,11] | [5,11,17,23] |
| Norm scale | 0–1 reflectance | raw DN / 10000 |

Both share the same FPN decoder API (lateral convs sized from `embed_dim`).
IBM's `Prithvi-EO-2.0-300M-BurnScars` checkpoint is not used directly (different
framework + UNet decoder); we fine-tune the base 2.0 encoder with our FPN.

- 1.0: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- 2.0: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M

## HLS brightness gain (Prithvi 1.0)

HLS LaSRC surface reflectance runs ~1.4–1.9× darker than Prithvi 1.0's
pretraining distribution, so the frozen encoder over-predicts burn. A per-band
gain (`GAIN_1` in `src.data.normalize_bands`), calibrated so the training-fire
median reflectance matches the pretraining mean, is applied before z-scoring.
This raised Woolsey IoU 0.53 → 0.73. Full investigation:
`results/over_prediction_analysis.md`.

No gain is applied for 2.0 yet — whether one is needed should be verified
empirically (`scripts/band_stats_v2.py`).

## Loss: CE + Tversky

`CETverskyLoss` = weighted cross-entropy + soft Tversky on the burn class.
Tversky index `TI = TP / (TP + alpha*FP + beta*FN)`; `alpha=beta=0.5` recovers
soft Dice. Raising alpha above beta penalizes false positives (curbs
over-prediction) at the cost of recall. alpha/beta and class weights are
calibrated on the training-fire val split only (see leakage discipline).

## Labels: dNBR (post-only model)

Burn labels come from `dNBR = NBR_pre − NBR_post`, `NBR = (NIR−SWIR2)/(NIR+SWIR2)`,
thresholded at 0.10. The model input is **post-fire only** (mono-temporal); the
pre-fire scene is used solely to build the label, never as model input.

## Patch sampling

Burn scars are rare, so keeping every background patch would swamp the positive
class — `background_keep` caps the background fraction. A patch is kept when at
least `min_valid_fraction` of its pixels are valid (not nodata); remaining
nodata is imputed to 0 (≈ per-band mean in z-scored space). Requiring *every*
pixel valid (an earlier approach) silently dropped whole large-fire scenes whose
every window clipped a nodata gap. `max_patches` stops one mega-fire dominating.

## Fire-based validation split

Validation holds out **whole fires** (`val_fires`), not random patches. A random
patch split lets spatially adjacent, near-duplicate patches from one fire land in
both train and val, giving an optimistic val IoU that hides overfitting — which
matters most when fine-tuning the full ViT.

## Hard-negative regions

`negative_regions` are unburned dry SoCal terrain (≈all-background masks),
patched through the same pipeline. They teach the post-only model that dark/dry
land is not burned, curbing over-prediction.

## NDWI water mask

Open water is not burnable, but both the post-only model and the dNBR label flag
it (where NIR ≈ SWIR ≈ 0, NBR is noise). `src.utils.water_mask` removes it
deterministically: `NDWI = (green−NIR)/(green+NIR)`, mask `NDWI > 0`. Burn scars
sit at NDWI ≤ 0, so the 0 cutoff drops ocean/lakes without eating real burns.
Independent of the model, never tuned on the test fires.

## Sliding-window inference

Full scenes are tiled into overlapping `patch_size` windows (half-patch stride),
with a final row/col anchored flush to the scene edge so the bottom/right strips
a fixed stride would skip are still covered. Per-pixel validity is computed once;
nodata is imputed to 0 within a window so a single cloud pixel no longer discards
the whole window. A thin border of the valid region is trimmed because
predictions abutting imputed nodata are unreliable.

## LLRD + gradual unfreeze (fine-tune)

The encoder trains decoder-only for `unfreeze_after_epoch` epochs, then unfreezes
with layer-wise LR decay (`llrd_decay`): shallow Prithvi layers get the smallest
LR so their general pretrained features are preserved, deeper layers a bit more.
Encoder depth is inferred from the model's named parameters so this works for
both 1.0 (depth 12) and 2.0 (depth 24).
