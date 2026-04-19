"""
Streamlit app for interactive burn scar detection visualization.
"""

import json
import sys
from pathlib import Path

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualize import create_folium_overlay, create_sentinel_overlay

st.set_page_config(
    page_title="Wildfire Burn Scar Detection",
    layout="wide",
)

REGION = {
    "display_name": "Woolsey Fire (2018)",
    "name": "woolsey_fire_2018",
    "center": [34.16, -118.83],
    "zoom": 11,
    "description": (
        "Burned 96,949 acres across Ventura and Los Angeles counties, "
        "destroying over 1,600 structures and threatening Malibu and Calabasas. "
        "November 2018."
    ),
    "acres": "96,949",
    "date": "November 2018",
}

TRAIN_FIRES = [
    "August Complex (2020)",
    "Mendocino Complex (2018)",
    "SCU Lightning Complex (2020)",
    "Creek Fire (2020)",
    "LNU Lightning Complex (2020)",
    "Thomas Fire (2017)",
    "Caldor Fire (2021)",
    "Antelope Fire (2021)",
]


def load_prediction() -> dict | None:
    pred_path = Path(f"data/predictions/{REGION['name']}.npz")
    if pred_path.exists():
        data = np.load(pred_path)
        return {
            "pred_mask": data["pred_mask"],
            "true_mask": data["true_mask"],
            "image": data["image"],
            "bounds": data["bounds"].tolist(),
        }
    return None


def create_map(pred_data: dict | None, overlay_opacity: float) -> folium.Map:
    m = folium.Map(location=REGION["center"], zoom_start=REGION["zoom"], tiles=None)
    if pred_data is not None:
        m.fit_bounds(pred_data["bounds"])

    folium.TileLayer(tiles="OpenStreetMap", name="Street Map", show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    perimeter_path = Path("data/perimeters/woolsey_fire_2018.geojson")
    if perimeter_path.exists():
        folium.GeoJson(
            json.loads(perimeter_path.read_text()),
            name="CAL FIRE Perimeter",
            style_function=lambda _: {"color": "#ff6600", "weight": 2.5, "fillOpacity": 0},
            tooltip="Official CAL FIRE perimeter",
        ).add_to(m)

    if pred_data is not None:
        create_sentinel_overlay(pred_data["image"], pred_data["bounds"]).add_to(m)
        create_folium_overlay(
            pred_data["pred_mask"], pred_data["bounds"], opacity=overlay_opacity
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def main():
    st.markdown(
        "<style>[data-testid='stMetricValue'] { font-size: 1.6rem; }</style>",
        unsafe_allow_html=True,
    )
    st.title("Wildfire Burn Scar Detection")
    st.caption(
        "Prithvi-EO-1.0-100M geospatial foundation model (IBM × NASA) "
        "fine-tuned on 8 California wildfires, evaluated on the Woolsey Fire."
    )

    pred_data = load_prediction()

    overlay_opacity = 0.6

    # --- Sidebar ---
    with st.sidebar:
        st.subheader(REGION["display_name"])
        st.caption(REGION["date"])
        st.markdown(REGION["description"])
        st.markdown(f"**Area burned:** {REGION['acres']} acres")

        st.divider()

        if pred_data is not None:
            pred = pred_data["pred_mask"]
            true = pred_data["true_mask"]
            total = pred.size

            tp = int(((pred == 1) & (true == 1)).sum())
            fp = int(((pred == 1) & (true == 0)).sum())
            fn = int(((pred == 0) & (true == 1)).sum())
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            st.subheader("Results")
            st.caption(
                "The model was trained on 8 other California wildfires and has never "
                "seen the Woolsey Fire. These numbers reflect how well it generalises."
            )

            c1, c2, c3 = st.columns(3)
            c1.metric(
                "Recall",
                f"{recall * 100:.0f}%",
                help="Of all pixels the dNBR map marks as burned, what fraction did the model detect.",
            )
            c2.metric(
                "Precision",
                f"{precision * 100:.0f}%",
                help="Of all pixels the model flagged as burned, what fraction actually appear burned in the dNBR map.",
            )
            c3.metric(
                "IoU",
                f"{iou * 100:.0f}%",
                help=(
                    "Intersection over Union: what fraction of burn pixels were identified by "
                    "both the model and the dNBR map."
                ),
            )

            st.divider()
            st.markdown(f"**Model detected:** {pred.sum() / total * 100:.1f}% of scene burned")
            st.markdown(f"**dNBR reference:** {true.sum() / total * 100:.1f}% of scene burned")

        else:
            st.info(
                "Predictions not yet available.\n\n"
                "```bash\npython run_inference.py --region woolsey_fire_2018\n```"
            )

        st.divider()
        st.caption("Encoder: Prithvi-EO-1.0-100M · 100M params · pretrained on 640k HLS scenes")

    # --- Map ---
    with st.spinner("Loading satellite imagery..."):
        m = create_map(pred_data, overlay_opacity)
    st_folium(m, use_container_width=True, height=560, returned_objects=[])

    # --- Comparison plot ---
    st.subheader("Prediction Comparison")
    st.caption(
        "Sentinel-2 RGB  ·  dNBR ground truth  ·  model prediction  ·  overlay. "
        "The model sees only post-fire imagery — no before/after comparison is made at inference time."
    )

    if pred_data is not None:
        from src.visualize import plot_predictions
        import matplotlib.pyplot as plt

        fig = plot_predictions(
            pred_data["image"],
            pred_data["true_mask"],
            pred_data["pred_mask"],
            title=REGION["display_name"],
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Run inference to generate the comparison plot.")

    # --- Methodology ---
    with st.expander("How it works"):
        st.markdown(f"""
        #### Data

        Sentinel-2 L2A satellite imagery with 6 spectral bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
        downloaded from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/).
        For each fire, two scenes are fetched: one before the fire and one after, selecting the
        lowest cloud-cover acquisition within the relevant date window.

        #### Labels

        There are no hand-drawn masks. Instead, labels are computed automatically from the
        Differenced Normalized Burn Ratio (dNBR):

        > `NBR = (NIR − SWIR2) / (NIR + SWIR2)`
        > `dNBR = NBR_before − NBR_after`

        Burned vegetation absorbs less NIR and reflects more SWIR than healthy plants, so dNBR spikes
        sharply where fire passed. Any pixel where `dNBR > 0.27` is classified as burned — the USGS
        threshold for moderate-to-high severity burns.

        #### Model

        **[Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M)**
        is a Vision Transformer pretrained by IBM and NASA on 640,000 Harmonized Landsat Sentinel-2
        (HLS) scenes — a globally representative dataset spanning all seasons and land cover types.
        The encoder learns rich spectral-spatial representations of Earth's surface.

        For burn scar detection, the Prithvi encoder (frozen) is paired with a lightweight CNN
        decoder. Each 224×224 image patch is:
        1. Passed through the 12-layer ViT encoder → 588 patch tokens of 768 dimensions
        2. Mean-pooled over the spatial grid → 14×14 feature map
        3. Upsampled by the CNN decoder (4× bilinear stages) → per-pixel burn probability

        Prithvi's patch embeddings naturally separate burned from unburned land in its
        representation space — evidence the foundation model has learned spectral features
        that transfer to fire detection without task-specific supervision.
        Reference: [arXiv:2310.18660](https://arxiv.org/abs/2310.18660)

        #### Train / test split

        Trained on 8 large California wildfires:
        {", ".join(TRAIN_FIRES)}.

        The Woolsey Fire is held out entirely — no Woolsey patches appear during training or
        validation. This gives a realistic measure of generalisation to an unseen fire.

        Full scenes are sliced into 224×224 px patches (50% overlap). Patches with at least 1%
        burned pixels are always kept; background-only patches are sampled at 30% to reduce class
        imbalance.
        """)


if __name__ == "__main__":
    main()
