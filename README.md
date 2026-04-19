# Wildfire Burn Scar Detection

Burn scar mapping from Sentinel-2 satellite imagery using **Prithvi-EO-1.0-100M** — the IBM × NASA geospatial foundation model — fine-tuned for pixel-level segmentation.

The model is trained on 8 large California wildfires and evaluated on the **Woolsey Fire (2018)**.

## Architecture

```
Sentinel-2 (6 bands) → Prithvi-EO ViT encoder → CNN decoder → burn scar mask
                         (100M params,               (8M params,
                          pretrained on               trained from
                          640k HLS scenes)            scratch)
```

**Encoder**: [Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M) — a 12-layer Vision Transformer pretrained by IBM and NASA on Harmonized Landsat Sentinel-2 (HLS) imagery. The encoder is frozen; only the decoder is fine-tuned.

**Decoder**: Four-stage transposed-convolution upsampling network that maps 14×14 patch embeddings back to 224×224 pixel-level predictions.

**Labels**: Derived automatically from the differenced Normalized Burn Ratio (dNBR = NBR_pre − NBR_post). Pixels with dNBR > 0.27 are classified as burned — no manual annotation.

## Project structure

```
run_training.py              train the model on 8 CA fires
run_inference.py             run on Woolsey Fire, save predictions
scripts/extract_embeddings.py  PCA of Prithvi patch embeddings
src/
  data.py      download, preprocess, patch dataset
  model.py     BurnScarModel (Prithvi encoder + CNN decoder)
  train.py     training loop
  visualize.py map overlays + comparison plots
  app/
    streamlit_app.py   interactive demo
configs/train_config.yaml
```

## Quick start

```bash
pip install -e .

# 1. Train (downloads Prithvi weights ~450 MB on first run; ~45 min on M1)
python run_training.py

# 2. Run inference on the held-out Woolsey Fire
python run_inference.py --region woolsey_fire_2018

# 3. Extract Prithvi embeddings for the embedding space visualization
python scripts/extract_embeddings.py

# 4. Launch the app
streamlit run src/app/streamlit_app.py
```

## Training fires

August Complex (2020) · Mendocino Complex (2018) · SCU Lightning Complex (2020) · Creek Fire (2020) · LNU Lightning Complex (2020) · Thomas Fire (2017) · Caldor Fire (2021) · Antelope Fire (2021)

## Results on Woolsey Fire (held out)

The model was trained on 8 Northern/Central California wildfires and has never seen the Woolsey Fire.

| Metric | Value |
|---|---|
| Recall | 75% |
| Precision | 77% |
| IoU | 61% |

Evaluated against dNBR ground truth (pixels with dNBR > 0.10 classified as burned). Water pixels are masked using NDWI before evaluation. The Woolsey Fire burned through chaparral in the Santa Monica Mountains — chaparral recovers faster spectrally than the conifer forests in the training set, so lower dNBR thresholds are appropriate. The model's predictions align visually with the official CAL FIRE perimeter.
