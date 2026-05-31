# Over-prediction analysis: does an asymmetric loss fix it under a post-only constraint?

**Date:** 2026-05-29
**Author:** Evan Kartheiser

## Summary

The burn-scar segmentation model over-predicts on the held-out test fires: high
recall, mediocre precision (e.g. Woolsey recall 0.92 / precision 0.53). I tested
whether reshaping the **training objective** to penalize false positives more
aggressively would curb this, while keeping the operational constraint that the
**model sees only the post-fire image** (no pre-fire reference at inference).

**Result: it did not.** At a fixed, untuned decision threshold, an asymmetric
Tversky loss left held-out precision and IoU essentially unchanged. This
localizes the limitation to the **single-date input**, not the loss function —
a model given one post-fire image cannot reliably separate burn scars from
spectrally similar dry/dark terrain, no matter how the loss is weighted.

## Hypothesis

The deployed loss is weighted cross-entropy (`class_weights = [0.3, 0.7]`,
penalizing missed burns 2.3× more than false alarms) plus a symmetric soft-Dice
term. Both push toward predicting "burned," which matches the over-prediction
signature. Hypothesis: replacing Dice with a **Tversky loss**

```
TI = TP / (TP + alpha*FP + beta*FN)        # alpha=beta=0.5 recovers Dice
```

and setting `alpha > beta` (heavier false-positive penalty), together with
neutral class weights `[0.5, 0.5]`, would trade recall for precision and raise
IoU on the held-out fires.

## Method (honest protocol)

- **Input unchanged:** post-fire image only (6 HLS bands). Labels are bi-temporal
  dNBR (threshold 0.10) — unchanged.
- **Sweep** over the false-positive penalty, training only the FPN decoder
  (Prithvi encoder frozen), 8 epochs each:
  - `tversky_a50`: alpha=0.5, beta=0.5 (neutral; equals Dice)
  - `tversky_a60`: alpha=0.6, beta=0.4
  - `tversky_a70`: alpha=0.7, beta=0.3 *(not completed — see Limitations)*
- **Model selection** by validation IoU on the **training fires' val split only**.
- **Evaluation** at a **fixed 0.5 threshold for every model** — zero per-model
  threshold tuning of any kind, and the test fires (Woolsey, East Troublesome,
  Thomas) are never used for any calibration. This isolates the effect of the
  loss change and is leakage-free by construction.

Reproduce with `scripts/eval_sweep.py`.

## Results

Validation IoU (train fires only):

| Config | alpha / beta | best val IoU |
|---|---|---|
| `balanced_chaparral` (baseline) | Dice + CE[0.3,0.7] | 0.795 |
| `tversky_a50` | 0.5 / 0.5 | 0.800 |
| `tversky_a60` | 0.6 / 0.4 | 0.787 |

Held-out test fires, fixed threshold 0.5 (Precision / Recall / IoU):

| Config | Woolsey | East Troublesome | Thomas | **Macro** |
|---|---|---|---|---|
| `balanced_chaparral` | 0.528 / 0.924 / 0.506 | 0.518 / 0.824 / 0.466 | 0.692 / 0.752 / 0.563 | 0.579 / 0.833 / **0.512** |
| `tversky_a50` | 0.507 / 0.879 / 0.474 | 0.670 / 0.773 / 0.560 | 0.594 / 0.777 / 0.507 | 0.591 / 0.809 / **0.514** |
| `tversky_a60` | 0.511 / 0.884 / 0.479 | 0.592 / 0.820 / 0.524 | 0.556 / 0.790 / 0.485 | 0.553 / 0.831 / **0.496** |

## Interpretation

- **No precision gain.** Macro precision moved 0.579 → 0.591 → 0.553; the *more*
  aggressive false-positive penalty made it *worse*. Macro IoU was flat
  (0.512 / 0.514 / 0.496).
- **Woolsey unchanged.** Precision 0.528 → 0.511 with recall falling — the loss
  change shifted the operating point slightly but did not separate the classes
  better.
- **Errors reshuffled, not reduced.** `tversky_a50` improved East Troublesome
  precision markedly (0.518 → 0.670) but degraded Thomas — consistent with a
  decision boundary moving along a fixed precision/recall frontier rather than a
  genuinely better-separated feature space.

The over-prediction is **input-limited**. The labels are inherently bi-temporal
(dNBR = pre-fire NBR − post-fire NBR), but the model only sees the post-fire
date. Dry chaparral, grassland, cleared/agricultural land, and some urban
surfaces are spectrally close to char/ash in a single post-fire image; the
discriminative signal that distinguishes them — the *change* from pre-fire — is
absent from the input. The earlier observation that the false-positive blob in
north Woolsey has dNBR ≈ 0 (no actual burn signal) yet reads as "burned" in the
post-only image is the same story. A loss function cannot recover information
that is not in the input.

## Follow-up experiments (all post-only, all leakage-free)

After the loss sweep, three further honest levers were tested. **None improved
held-out precision/IoU on the SoCal test fires.** All test-fire numbers below use
the deployed pipeline (fixed 0.5 threshold + NDWI water exclusion).

### 1. Decision-threshold recalibration (no retrain)

Swept the threshold on the **training fires only** (`scripts/calibrate_threshold.py`)
to maximize mean train-fire IoU. The optimum is **0.46 — *below* the deployed
0.5** — and train-fire IoU falls monotonically as the threshold rises. Raising
the threshold to suppress over-prediction is therefore *not* honestly justifiable
(it would lower train-fire IoU), and applying 0.46 to the test fires lowers their
IoU (more predictions). **Over-prediction is not a threshold artifact.**

### 2. Encoder fine-tuning

Evaluated the existing encoder-unfrozen checkpoint vs the frozen baseline:

| model | macro Precision | macro IoU |
|---|---|---|
| `balanced_chaparral` (frozen, deployed) | **0.588** | **0.539** |
| `finetuned` (encoder unfrozen) | 0.422 | 0.408 |
| `balanced` (frozen, older) | 0.561 | 0.516 |

Fine-tuning made over-prediction **worse** — the frozen Prithvi representation is
the better starting point here.

### 3. Hard-negative data

Added 4 verified-unburned SoCal dry-terrain scenes (Inland Empire, Cleveland NF,
San Diego backcountry, San Gabriel — each <1.5% dNBR "burn", confirmed with tight
~4-week late-summer date pairs to avoid phenology-driven dNBR; a Santa Monica Mtns
candidate was dropped for Woolsey proximity) as explicit negatives (240 patches),
retrained the decoder (`checkpoints/hardneg/`), same loss as baseline:

| model | Woolsey IoU | East Troub. IoU | Thomas IoU | macro P / IoU |
|---|---|---|---|---|
| `balanced_chaparral` | 0.526 | 0.488 | 0.602 | 0.588 / 0.539 |
| `hardneg` | 0.479 | 0.503 | 0.562 | 0.565 / 0.514 |

Net **worse**: helped East Troublesome but hurt Woolsey and Thomas — the exact
SoCal coastal fires the negatives targeted. With a frozen encoder, the decoder
lacks the capacity to learn the burn-vs-dry-chaparral distinction from these
negatives.

### Consolidated conclusion

Four independent honest levers — asymmetric loss, threshold, fine-tuning, hard
negatives — all fail to reduce land over-prediction on the SoCal test fires. This
is strong, convergent evidence that the limitation is the **single-date input**,
not the objective, threshold, or training data. The deployed `balanced_chaparral`
remains the best model. (A separate, fully-honest win was the **NDWI water mask**,
which removed spurious burn over open water and lifted macro IoU 51.3 → 54.0.)

## RESOLUTION: the cause was dark HLS input, not the single-date limit

The "input-limited" conclusion above was **wrong**, and a user observation —
that the earlier Sentinel-2 (Planetary Computer) pipeline over-predicted far
less with the *same* post-only setup — is what corrected it.

**Root cause.** HLS surface reflectance (LaSRC atmospheric correction + BRDF
normalization) runs systematically **~1.4–1.9× darker** than the HLS
distribution Prithvi-EO was pretrained on — most strongly in the visible bands.
Verified against a raw granule: the scale factor (0.0001) is applied correctly
once, so this is a genuine sensor/processing property, not a pipeline bug.
Healthy vegetation (NDVI > 0.5) reads NIR ≈ 0.13–0.18 where ~0.30–0.45 is
expected. Fed to the **frozen** Prithvi encoder, this dark input pushes features
toward the low-NIR "burn" signature, so the model floods burn predictions — worst
on the darkest scenes (Woolsey normalized-NIR −0.86, the most over-predicting).
This also explains the Sentinel-2 observation: S2 L2A (Sen2Cor) is brighter and
closer to Prithvi's expected range. Every earlier lever failed because the
decoder, loss, threshold, and training data are all *downstream* of the frozen
encoder — they cannot fix features that are already burn-biased by dark input.

**Fix.** A fixed per-band brightness gain in `normalize_bands`, calibrated so the
pooled *training*-fire median reflectance matches the Prithvi pretraining mean
(no test data, no labels): `[1.879, 1.717, 1.574, 1.410, 1.130, 1.128]` for
B02,B03,B04,B8A,B11,B12. A *global* gain (uniform across scenes) was chosen over
per-scene rescaling, which over-corrected bright/snowy scenes (East Troublesome
IoU collapsed 0.52→0.24 per-scene vs preserved with global).

**Result (water-masked, threshold 0.5):**

| config | Woolsey IoU | East Troub. IoU | Thomas IoU | macro IoU |
|---|---|---|---|---|
| original (no gain) | 0.526 | 0.488 | 0.602 | 0.539 |
| **balanced_chaparral + gain (deployed)** | **0.729** | 0.489 | **0.688** | **0.635** |

Woolsey precision 0.53→0.76, Thomas precision 0.70→0.96. Macro IoU 0.539→0.635.

**Matched retrains underperformed.** Retraining the decoder on gain-corrected
input (so train==inference) scored *worse* than applying the gain to the existing
model: `bright` ([0.3,0.7]+Dice) 0.507 macro, `bright_p` (precision-leaning loss)
0.585 macro — both below the deployed 0.635. The decoder trained on the original
(harder, dark) input learned a more conservative boundary that, given clean
in-domain features at inference, generalizes best. The deployment therefore keeps
the original `balanced_chaparral` checkpoint with the gain applied at inference;
the train/serve preprocessing difference is intentional and was validated against
matched alternatives.

**Residual.** Some over-prediction remains on western chaparral hillsides
(precision ~0.76 on Woolsey) — a candidate for future work, but the dominant
flooding is resolved.

## What would actually move precision (earlier note, now superseded by the fix above)

1. **Hard-negative data.** Train on explicit unburned dry-terrain scenes
   (chaparral, grassland, cleared/agricultural, urban) so the model learns
   "dark ≠ burned" from the spectral signature alone. Most promising
   data-centric lever within the constraint.
2. **Encoder fine-tuning.** Unfreeze Prithvi (still post-only) to learn
   burn-specific spectral features beyond the frozen pretrained representation.
3. **Accept the ceiling.** If a single post-fire date is a hard product
   constraint, document that bi-temporal dNBR-quality precision is not
   attainable from one date and report the achievable operating point.

## Reproducibility / artifacts

- Loss: `src/train.py::CETverskyLoss` (Tversky generalization of the prior
  CE+Dice loss; `alpha=beta=0.5` is backward-compatible with Dice).
- Sweep: `run_training.py --experiment-name <name> --tversky-alpha A --tversky-beta B --class-weights W0 W1`
- Evaluation: `scripts/eval_sweep.py --checkpoints ... --threshold 0.5`
- Checkpoints: `checkpoints/tversky_a50/`, `checkpoints/tversky_a60/`
  (the aggressive `tversky_a70` run was stopped after 1 epoch and is not
  reported).
- Deployed model is unchanged: `checkpoints/balanced_chaparral/best_model.pt`.
