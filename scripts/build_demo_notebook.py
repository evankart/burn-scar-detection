"""Build notebooks/demo_analysis.ipynb — a narrative results notebook that reads
the precomputed prediction .npz files (no model loading) and renders RGB / dNBR /
ground-truth / prediction panels plus a held-out metrics table."""
import sys
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
def md(t): cells.append(new_markdown_cell(t))
def code(t): cells.append(new_code_cell(t))

md("""# Wildfire Burn-Scar Segmentation with Prithvi-EO

**Fine-tuning NASA/IBM's Prithvi-EO-1.0-100M geospatial foundation model to map wildfire burn scars from satellite imagery.**

This notebook walks through the held-out results and the key modeling findings. It reads precomputed predictions (no GPU needed) — the training/inference code lives in `src/` and `run_*.py`.

- **Encoder:** Prithvi-EO-1.0-100M (ViT, pretrained by IBM/NASA on HLS), frozen, with an FPN decoder.
- **Imagery:** Harmonized Landsat-Sentinel (HLS) surface reflectance, 30 m, via NASA `earthaccess`.
- **Labels:** bi-temporal dNBR (pre vs post-fire NBR; burn = dNBR > 0.10).
- **Evaluation:** three held-out fires the model never trained on — Woolsey, Thomas (SoCal coastal chaparral) and East Troublesome (Colorado Rockies).
- **Honest methods:** the decision threshold is fixed at 0.5 and was *never* tuned on the test fires; the brightness correction and water mask use only training-fire statistics / physics — no test-set leakage.""")

code("""import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Resolve repo root whether run from repo root or notebooks/
ROOT = Path.cwd()
if not (ROOT / "data/predictions").exists() and (ROOT.parent / "data/predictions").exists():
    ROOT = ROOT.parent
PRED_DIR = ROOT / "data/predictions"

FIRES = {
    "woolsey_fire_2018":     "Woolsey Fire (2018) — Malibu / Thousand Oaks, CA",
    "thomas_fire_2017":      "Thomas Fire (2017) — Ventura / Santa Barbara, CA",
    "east_troublesome_2020": "East Troublesome Fire (2020) — Colorado Rockies",
}

def load(name):
    d = np.load(PRED_DIR / f"{name}.npz")
    return {k: d[k] for k in d.files}

def metrics(pred, true):
    p, t = pred.astype(bool), true.astype(bool)
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    iou  = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return prec, rec, iou

def rgb(image):
    \"\"\"Build a display RGB from the normalized HLS bands [B02,B03,B04,B8A,B11,B12].\"\"\"
    out = np.dstack([image[2], image[1], image[0]]).astype(np.float32)  # R=B04,G=B03,B=B02
    for i in range(3):
        ch = out[..., i]
        lo, hi = np.nanpercentile(ch, 2), np.nanpercentile(ch, 98)
        out[..., i] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
    return np.nan_to_num(out, nan=0.0)

print("predictions:", [f.name for f in sorted(PRED_DIR.glob('*.npz'))])""")

md("""## Held-out performance

Precision / Recall / IoU on the three fires the model never saw, at the deployed 0.5 threshold (open water excluded via NDWI — see below).""")

code("""rows, raw = [], []
for name, label in FIRES.items():
    d = load(name)
    p, r, i = metrics(d["pred_mask"], d["true_mask"])
    raw.append((p, r, i))
    rows.append({"Fire": label.split(" — ")[0], "Precision": f"{p:.0%}", "Recall": f"{r:.0%}", "IoU": f"{i:.0%}"})
macro = np.mean(raw, axis=0)
rows.append({"Fire": "— Macro average —", "Precision": f"{macro[0]:.0%}", "Recall": f"{macro[1]:.0%}", "IoU": f"{macro[2]:.0%}"})
pd.DataFrame(rows)""")

md("""## Visual results

For each fire: **HLS post-fire RGB** · **dNBR burn severity** · **ground truth** (dNBR > 0.10) · **model prediction**. All panels share the same pixel grid, so overlays line up exactly.""")

code("""def show_fire(name):
    d = load(name)
    img, pred, true, dnbr = d["image"], d["pred_mask"], d["true_mask"], d["dnbr"]
    p, r, i = metrics(pred, true)
    base = rgb(img)
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(base); ax[0].set_title("HLS post-fire (RGB)")
    ax[1].imshow(np.ma.masked_invalid(dnbr), cmap="RdYlGn_r", vmin=-0.2, vmax=0.6)
    ax[1].set_title("dNBR (burn severity)")
    ax[2].imshow(base); ax[2].imshow(np.ma.masked_where(true == 0, true), cmap="autumn", alpha=0.55)
    ax[2].set_title("Ground truth (dNBR > 0.10)")
    ax[3].imshow(base); ax[3].imshow(np.ma.masked_where(pred == 0, pred), cmap="autumn", alpha=0.55)
    ax[3].set_title("Model prediction")
    for a in ax: a.axis("off")
    fig.suptitle(f"{FIRES[name]}    |    Precision {p:.0%}   Recall {r:.0%}   IoU {i:.0%}", fontsize=15)
    plt.tight_layout(); plt.show()

show_fire("woolsey_fire_2018")""")

code('show_fire("thomas_fire_2017")')
code('show_fire("east_troublesome_2020")')

md("""## Key finding: HLS reflectance is darker than the encoder expects

The model originally **flooded** burn predictions across each scene (Woolsey precision ~0.53, IoU ~0.53). Investigating why led to the central result of this project:

**HLS surface reflectance (LaSRC atmospheric correction + BRDF) runs ~1.4–1.9× darker than the HLS distribution Prithvi-EO was pretrained on** — most strongly in the visible bands. Healthy vegetation reads NIR ≈ 0.13–0.18 where ≈ 0.30–0.45 is expected. Fed to the *frozen* encoder, this dark input mimics the low-NIR signature of char, so the model labels too much as burned.

The fix is a fixed per-band **brightness gain** (calibrated so the pooled *training*-fire median reflectance matches the Prithvi pretraining mean — no test data, no labels), applied before normalization. Effect on held-out fires:

| | Woolsey IoU | East Troublesome IoU | Thomas IoU | **Macro IoU** |
|---|---|---|---|---|
| original (no gain) | 0.526 | 0.488 | 0.602 | **0.539** |
| **+ brightness gain (deployed)** | **0.729** | 0.489 | **0.688** | **0.635** |

Woolsey precision 0.53 → 0.76; Thomas precision 0.70 → 0.96. Full investigation in [`results/over_prediction_analysis.md`](../results/over_prediction_analysis.md).""")

md("""## Methods & honesty notes

- **No test-set leakage.** The 0.5 decision threshold was calibrated on training fires only; the brightness gain uses pooled training-fire medians; the water mask is a physics-based spectral index (NDWI). None touch the held-out fires.
- **Water exclusion.** Open water makes NBR pure noise (NIR ≈ SWIR ≈ 0), so both the model and the dNBR label spuriously flag it as burned. An NDWI mask removes it deterministically.
- **What didn't work (documented honestly).** Asymmetric loss, threshold re-tuning, encoder fine-tuning, and hard-negative data all failed to beat the frozen+gain model — the limitation was the *input domain*, not the objective. See the analysis writeup.

## Limitations & future work

- Residual over-prediction remains on dry chaparral hillsides (Woolsey precision ~0.76).
- Encoder fine-tuning overfit on a small (~19-fire) training set; the set has since been expanded to **37 geographically diverse fires**, and a GPU fine-tune (with layer-wise LR decay + a fire-based validation split) is set up to run on cloud hardware.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "bsc-venv", "display_name": "burn-scar venv", "language": "python"}
out = Path("notebooks"); out.mkdir(exist_ok=True)
path = out / "demo_analysis.ipynb"
nbf.write(nb, path)
print("wrote", path, "with", len(cells), "cells")
