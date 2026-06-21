---
title: Wildfire Burn Scar Detection
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Wildfire Burn Scar Detection

Fine-tuned NASA/IBM **Prithvi-EO-2.0-300M** geospatial foundation model mapping
wildfire burn scars from Harmonized Landsat-Sentinel (HLS) imagery.

Trained on **100 wildfires** (37 US + 55 global across 6 biomes). Macro IoU **0.73** on
4 held-out test fires at a fixed threshold of 0.5.

- **Held-out fires** tab: precomputed results on fires the model never trained on.
- **Detect on custom area** tab: draw an AOI, pick a post-fire date, and run the model live.

See the [GitHub repo](https://github.com/evankart/burn-scar-detection) for the full methodology.
