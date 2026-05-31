---
title: Wildfire Burn Scar Detection
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: streamlit
app_file: app.py
pinned: false
license: mit
---

# Wildfire Burn Scar Detection

Fine-tuned NASA/IBM **Prithvi-EO-1.0-100M** geospatial foundation model mapping
wildfire burn scars from Harmonized Landsat-Sentinel (HLS) imagery.

- **Held-out fires** tab: precomputed results on three fires the model never trained on.
- **Detect on custom area** tab: draw an AOI, pick a post-fire date, and run the model live.

Built with a frozen Prithvi encoder + FPN decoder, a calibrated HLS brightness
correction, and an NDWI water mask. See the repo for the full methodology.
