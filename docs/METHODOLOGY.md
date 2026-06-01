# Methodology & Design Rationale

Design decisions and the reasoning behind them. Code comments stay minimal and
point here; this file (and `results/over_prediction_analysis.md`) hold the
"why". Empirical results are in `results/`.

## Two configurations

The repo carries two configs that share one fire registry but differ in intent:

| | `train_config.yaml` (deployed baseline) | `finetune_config.yaml` (staged) |
|---|---|---|
| Encoder | Prithvi 1.0-100M, **frozen** | Prithvi 2.0-300M, gradual unfreeze |
| Bands | B02,B03,B04,B8A,B11,B12 | B02,B03,B04,B05,B06,B07 |
| Brightness gain | yes (1.0) | none yet (verify first) |
| Val split | random 90/10 patch split | fire-based (hold out carr + holy) |
| Loss | CE weights [0.3, 0.7] + Dice | CE [0.5, 0.5] + Dice |
| Epochs | 8 | 16 |
| LLRD / unfreeze | n/a (frozen) | llrd_decay 0.75, unfreeze @ epoch 2 |

The deployed model is the **frozen Prithvi 1.0 encoder + FPN decoder + brightness
gain** (`checkpoints/balanced_chaparral`). The 2.0 fine-tune is staged for the
AWS run and not yet trained/evaluated.

## Leakage discipline (non-negotiable)

The decision threshold (0.5, `pred_threshold`), loss hyperparameters (Tversky
alpha/beta, class weights), and the brightness gain are **only ever set or
calibrated using the training fires (and their held-out val split)** — never the
three test fires (`woolsey_fire_2018`, `east_troublesome_2020`,
`thomas_fire_2017`). The threshold is fixed at 0.5, not tuned on the test fires.
This keeps the reported test-fire numbers defensible.

## Configuration registry

`configs/train_config.yaml` is the single source of truth: every fire lives in
one `data.fires` list, each tagged `role: train | test` (a `negative` role is
also supported by the loader; none are configured currently — see Hard
negatives below). `src.data.load_config` derives the `train_regions` /
`test_regions` / `negative_regions` lists from those roles so the splits can
never drift. `configs/finetune_config.yaml` is generated from it by
`scripts/make_finetune_config.py` and uses the explicit derived lists (also
accepted by `load_config`).

Current split: 37 train fires, 3 held-out test fires.

## Prithvi encoder versions

| | 1.0-100M | 2.0-300M |
|---|---|---|
| Arch | ViT-Base, embed 768, depth 12 | ViT-Large, embed 1024, depth 24 |
| Frames | 3 | 4 |
| Pretrain corpus | ~640k HLS scenes | ~4.2M HLS scenes (larger/more diverse) |
| Bands | B02,B03,B04,B8A,B11,B12 | B02,B03,B04,B05,B06,B07 |
| FPN feature layers | [2,4,7,11] | [5,11,17,23] |
| Norm scale | 0–1 reflectance | raw DN / 10000 |

Both share the same FPN decoder API (lateral convs sized from `embed_dim`).
IBM's `Prithvi-EO-2.0-300M-BurnScars` checkpoint is not used directly (different
framework + UNet decoder); we fine-tune the base 2.0 encoder with our FPN.

- 1.0: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
- 2.0: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M

## Mono-temporal input

The model input is **post-fire only** (a single HLS scene). The encoder expects
`(B, C, T, H, W)`; the one post-fire scene is replicated across the T temporal
frames. The pre-fire scene is used solely to build the dNBR label, never as
model input.

## Labels: dNBR

Burn labels come from `dNBR = NBR_pre − NBR_post`, where
`NBR = (NIR − SWIR2)/(NIR + SWIR2)` (NIR = B8A, SWIR2 = B12), thresholded at
0.10 (`dnbr_threshold`).

## HLS brightness gain (Prithvi 1.0 only)

HLS LaSRC surface reflectance runs ~1.4–1.9× darker than Prithvi 1.0's
pretraining distribution, so the frozen encoder over-predicts burn. A per-band
gain (`GAIN_1` in `src.data.normalize_bands`), calibrated so the training-fire
median reflectance matches the pretraining mean, is applied before z-scoring.
This raised Woolsey IoU 0.53 → 0.73. Full investigation:
`results/over_prediction_analysis.md`.

No gain is applied for 2.0 — whether one is needed should be verified
empirically first (`scripts/band_stats_v2.py`).

## Loss: CE + Tversky

`CETverskyLoss` = weighted cross-entropy + soft Tversky on the burn class.
Tversky index `TI = TP / (TP + alpha·FP + beta·FN)`; `alpha = beta = 0.5`
recovers soft Dice. Raising alpha above beta penalizes false positives (curbs
over-prediction) at the cost of recall.

In practice both configs run with `alpha = beta = 0.5` (Dice); the precision
lever comes from the CE class weights instead — `[0.3, 0.7]` (favor recall) for
the frozen baseline, neutral `[0.5, 0.5]` for the fine-tune (which has the
capacity to learn precision directly). The asymmetric Tversky knob is available
via CLI overrides and, if used, would be calibrated on the training-fire val
split only (see leakage discipline).

## Patch sampling

Burn scars are rare, so keeping every background patch would swamp the positive
class — `background_keep` (default 0.3) caps the background fraction. A patch is
kept when at least `min_valid_fraction` of its pixels are valid (not nodata);
remaining nodata is imputed to 0 (≈ per-band mean in z-scored space). Requiring
*every* pixel valid (an earlier approach) silently dropped whole large-fire
scenes whose every window clipped a nodata gap. `max_patches` (optional) caps the
per-fire count so one mega-fire can't dominate.

## Validation split

- **Frozen baseline** (`train_config`): a random 90/10 patch split
  (`train_split: 0.9`). Adequate because the encoder is frozen and the decoder
  is small, so cross-patch leakage has limited effect.
- **Fine-tune** (`finetune_config`): a **fire-based** split via `val_fires` —
  whole fires (carr + holy) are held out for validation. A random patch split
  lets spatially adjacent, near-duplicate patches from one fire land in both
  train and val, giving an optimistic val IoU that hides overfitting, which
  matters when the full ViT is unfrozen.

## Hard-negative regions (explored, not currently used)

`load_config` supports a `negative` role for unburned hard-negative regions
(dry terrain, ≈all-background masks) intended to teach the post-only model that
dark/dry land is not burned. These were trialed but did not beat the
frozen-1.0 + gain baseline on the held-out fires, so **no negative regions are
configured currently** (`negative_regions` is empty). Documented as a negative
result in `results/over_prediction_analysis.md`.

## NDWI water mask

Applied at inference (not training). Open water is not burnable, but both the
post-only model and the dNBR label flag it (where NIR ≈ SWIR ≈ 0, NBR is noise).
`src.utils.water_mask` removes it deterministically: `NDWI = (green − NIR)/(green
+ NIR)` (green = B03, NIR = B8A), mask `NDWI > 0`. Burn scars sit at NDWI ≤ 0, so
the 0 cutoff drops ocean/lakes without eating real burns. Independent of the
model, never tuned on the test fires.

## Sliding-window inference

Full scenes are tiled into overlapping `patch_size` (224) windows at a half-patch
stride, with a final row/col anchored flush to the scene edge so the bottom/right
strips a fixed stride would skip are still covered. Per-pixel validity is
computed once; nodata is imputed to 0 within a window so a single cloud pixel no
longer discards the whole window. Overlapping predictions are averaged, then a
thin border of the valid region is trimmed (10-iteration erosion) because
predictions abutting imputed nodata are unreliable.

## LLRD + gradual unfreeze (fine-tune only)

Applies to `finetune_config`; the deployed baseline keeps the encoder frozen
throughout (`freeze_backbone: true`, `unfreeze_after_epoch: 999`). In the
fine-tune, the encoder trains decoder-only for `unfreeze_after_epoch` (2) epochs,
then unfreezes with layer-wise LR decay (`llrd_decay` 0.75): shallow Prithvi
layers get the smallest LR so their general pretrained features are preserved,
deeper layers a bit more. Encoder depth is inferred from the model's named
parameters so this works for both 1.0 (depth 12) and 2.0 (depth 24).
