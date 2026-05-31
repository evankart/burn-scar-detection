"""
Streamlit app for interactive burn scar detection visualization.
"""

import json
import sys
from pathlib import Path

import math
from datetime import date

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw, Geocoder
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualize import (
    SEVERITY_CLASSES,
    create_folium_overlay,
    create_sentinel_overlay,
    create_severity_overlay,
)

st.set_page_config(
    page_title="Wildfire Burn Scar Detection",
    layout="wide",
)

REGIONS = {
    "woolsey_fire_2018": {
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
        "setting": "Southern California chaparral",
    },
    "east_troublesome_2020": {
        "display_name": "East Troublesome Fire (2020)",
        "name": "east_troublesome_2020",
        "center": [40.25, -105.85],
        "zoom": 10,
        "description": (
            "Burned 193,812 acres in Grand County, Colorado — among the largest fires "
            "in state history. Grew explosively in October 2020, forcing evacuations "
            "near Grand Lake and Rocky Mountain National Park."
        ),
        "acres": "193,812",
        "date": "October 2020",
        "setting": "Colorado subalpine forest",
    },
    "thomas_fire_2017": {
        "display_name": "Thomas Fire (2017)",
        "name": "thomas_fire_2017",
        "center": [34.48, -119.25],
        "zoom": 10,
        "description": (
            "Burned 281,893 acres across Ventura and Santa Barbara counties over "
            "several weeks — among the largest California wildfires on record at the "
            "time. December 2017."
        ),
        "acres": "281,893",
        "date": "December 2017",
        "setting": "California coastal sage & chaparral",
    },
}

DEFAULT_REGION = "woolsey_fire_2018"

TRAIN_FIRES = [
    "August Complex (2020)",
    "Mendocino Complex (2018)",
    "SCU Lightning Complex (2020)",
    "Caldor (2021)",
    "LNU Lightning Complex (2020)",
    "North Complex (2020)",
    "Carr (2018)",
    "Dixie (2021)",
    "Antelope (2021)",
    "Holy (2018)",
    "Bobcat (2020)",
    "Bootleg (2021, OR)",
    "Hermits Peak (2022, NM)",
    "Pearl Hill (2020, WA)",
]


HF_REPO = "evankart/burn-scar-detection-data"


@st.cache_data(show_spinner=False)
def load_prediction(region_name: str) -> dict | None:
    pred_path = Path(f"data/predictions/{region_name}.npz")
    if not pred_path.exists():
        with st.spinner("Downloading predictions from HuggingFace Hub…"):
            try:
                from huggingface_hub import hf_hub_download
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=f"predictions/{region_name}.npz",
                    local_dir="data",
                )
            except Exception as e:
                st.error(f"Could not download predictions: {e}")
                return None
    data = np.load(pred_path)
    return {
        "pred_mask": data["pred_mask"],
        "true_mask": data["true_mask"],
        "image": data["image"],
        "dnbr": data["dnbr"] if "dnbr" in data else None,
        "bounds": data["bounds"].tolist(),
    }


def create_map(region: dict, pred_data: dict | None, overlay_opacity: float) -> folium.Map:
    m = folium.Map(location=region["center"], zoom_start=region["zoom"], tiles=None)
    if pred_data is not None:
        m.fit_bounds(pred_data["bounds"])

    folium.TileLayer(tiles="OpenStreetMap", name="Street Map", show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
    ).add_to(m)

    perimeter_path = Path(f"data/perimeters/{region['name']}.geojson")
    if perimeter_path.exists():
        folium.GeoJson(
            json.loads(perimeter_path.read_text()),
            name="CAL FIRE Perimeter",
            style_function=lambda _: {"color": "#ff6600", "weight": 2.5, "fillOpacity": 0},
            tooltip="Official CAL FIRE perimeter",
        ).add_to(m)

    if pred_data is not None:
        create_sentinel_overlay(pred_data["image"], pred_data["bounds"]).add_to(m)
        if pred_data.get("dnbr") is not None:
            # Ground-truth burn-severity gradient; off by default so it doesn't
            # stack on the model prediction — toggle it on via the layer control.
            create_severity_overlay(
                pred_data["dnbr"], pred_data["bounds"], show=False
            ).add_to(m)
        create_folium_overlay(
            pred_data["pred_mask"], pred_data["bounds"], opacity=overlay_opacity
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ----------------------------------------------------------------------------
# Custom-AOI live detection (post-fire only)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model():
    from src.infer import load_model
    return load_model()


@st.cache_data(show_spinner=False)
def run_detection(bbox: tuple, post_date: str) -> dict:
    """Cached so re-running the same AOI/date is instant."""
    from src.infer import detect_burn_scar
    model, device, cfg = _load_model()
    return detect_burn_scar(bbox, post_date, model, device, cfg)


def _aoi_area_km2(bbox: tuple) -> float:
    minlon, minlat, maxlon, maxlat = bbox
    midlat = math.radians((minlat + maxlat) / 2)
    w = (maxlon - minlon) * 111.32 * math.cos(midlat)
    h = (maxlat - minlat) * 110.57
    return abs(w * h)


def custom_detection_view():
    st.subheader("Detect burn scars on a custom area")
    st.markdown(
        "**1.** Search or zoom to a location · **2.** Draw a polygon or rectangle with the "
        "tools on the left edge of the map · **3.** Pick a post-fire date · **4.** Run detection.  \n"
        "_The model runs on a single post-fire HLS scene — no before/after comparison._"
    )
    post_date = st.date_input(
        "Post-fire date",
        value=date(2020, 9, 15), min_value=date(2015, 7, 1), max_value=date.today(),
        help="The app finds the least-cloudy HLS scene within ~30 days after this date.",
    )

    draw_map = folium.Map(location=[39.0, -120.5], zoom_start=6, tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(draw_map)
    Geocoder(collapsed=False, add_marker=False).add_to(draw_map)
    Draw(
        draw_options={"polyline": False, "circle": False, "marker": False,
                      "circlemarker": False, "polygon": True, "rectangle": True},
        edit_options={"edit": False, "remove": True},
    ).add_to(draw_map)
    out = st_folium(draw_map, key="draw_map", use_container_width=True, height=480,
                    returned_objects=["last_active_drawing"])

    # Capture the drawn polygon -> bbox; persist so it survives the button rerun.
    draw = (out or {}).get("last_active_drawing")
    if draw and draw.get("geometry", {}).get("type") == "Polygon":
        ring = draw["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
        st.session_state["aoi_bbox"] = (min(lons), min(lats), max(lons), max(lats))

    bbox = st.session_state.get("aoi_bbox")
    if not bbox:
        st.info("Draw a polygon or rectangle on the map to define your area of interest.")
        return

    area = _aoi_area_km2(bbox)
    st.caption(f"AOI ≈ {area:,.0f} km²  ·  bbox {tuple(round(b, 3) for b in bbox)}")
    if area > 10000:
        st.warning("Large area — the download and inference will be slow. A smaller AOI "
                   "(under ~10,000 km²) is recommended.")

    if not st.button("🔥 Detect burn scars", type="primary"):
        return
    try:
        with st.spinner("Finding and downloading the HLS scene, then running the model (~1–3 min)…"):
            res = run_detection(tuple(round(b, 4) for b in bbox), post_date.isoformat())
    except Exception as e:
        st.error(f"Could not run detection: {e}")
        return

    st.success(f"Detected burn scar over ~{res['burned_frac'] * 100:.1f}% of the area.")
    st.caption(f"Imagery: least-cloudy HLS acquired **{res['scene_date']}** "
               f"(searched ≤30 days after your date; composited from {res['n_scenes']} scene(s)).")
    s, w = res["bounds"][0]; n, e = res["bounds"][1]
    rm = folium.Map(location=[(s + n) / 2, (w + e) / 2], zoom_start=10, tiles=None)
    rm.fit_bounds(res["bounds"])
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(rm)
    create_sentinel_overlay(res["image"], res["bounds"]).add_to(rm)
    create_folium_overlay(res["pred_mask"], res["bounds"], opacity=0.6).add_to(rm)
    folium.LayerControl(collapsed=False).add_to(rm)
    st_folium(rm, key="result_map", use_container_width=True, height=560, returned_objects=[])


def main():
    st.markdown(
        "<style>[data-testid='stMetricValue'] { font-size: 1.6rem; }</style>",
        unsafe_allow_html=True,
    )
    st.title("Wildfire Burn Scar Detection")
    st.caption(
        "Prithvi-EO-1.0-100M geospatial foundation model (IBM × NASA) "
        "fine-tuned on 12 wildfires across 4 US states, evaluated on 3 held-out fires."
    )

    overlay_opacity = 0.6

    with st.sidebar:
        mode = st.radio(
            "Mode",
            ["📍 Held-out test fires", "✏️ Detect on custom area"],
            help="Browse the held-out evaluation fires, or draw your own area and run the model live.",
        )

    if mode.endswith("custom area"):
        custom_detection_view()
        return

    # --- Sidebar ---
    with st.sidebar:
        region_key = st.selectbox(
            "Held-out test fire",
            options=list(REGIONS.keys()),
            index=list(REGIONS.keys()).index(DEFAULT_REGION),
            format_func=lambda k: REGIONS[k]["display_name"],
        )
        region = REGIONS[region_key]
        pred_data = load_prediction(region["name"])

        st.subheader(region["display_name"])
        st.caption(f"{region['date']} · {region['setting']}")
        st.markdown(region["description"])
        st.markdown(f"**Area burned:** {region['acres']} acres")

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
                f"The model was trained on 12 wildfires across 4 US states and has "
                f"never seen the {region['display_name']}. These numbers reflect how "
                f"well it generalises to unseen terrain."
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

            if pred_data.get("dnbr") is not None:
                st.divider()
                st.markdown("**dNBR burn severity**")
                st.caption("Ground-truth severity gradient — toggle the layer on the map.")
                swatches = []
                for label, lo, hi, color in SEVERITY_CLASSES:
                    hi_txt = "∞" if hi == float("inf") else f"{hi:.2f}"
                    swatches.append(
                        f"<div style='display:flex;align-items:center;margin:2px 0;'>"
                        f"<span style='width:14px;height:14px;background:{color};"
                        f"display:inline-block;margin-right:8px;border:1px solid #999;'></span>"
                        f"<span style='font-size:0.85rem;'>{label} ({lo:.2f}–{hi_txt})</span>"
                        f"</div>"
                    )
                st.markdown("".join(swatches), unsafe_allow_html=True)

        else:
            st.info(
                "Predictions not yet available.\n\n"
                f"```bash\npython run_inference.py --region {region['name']}\n```"
            )

        st.divider()
        st.caption("Encoder: Prithvi-EO-1.0-100M · 100M params · pretrained on 640k HLS scenes")

    # --- Map ---
    with st.spinner("Loading satellite imagery..."):
        m = create_map(region, pred_data, overlay_opacity)
    st_folium(m, use_container_width=True, height=560, returned_objects=[])

    # --- Comparison plot ---
    st.subheader("Prediction Comparison")
    st.caption(
        "HLS RGB  ·  dNBR ground truth  ·  model prediction  ·  overlay. "
        "The model sees only post-fire imagery — no before/after comparison is made at inference time."
    )

    if pred_data is not None:
        from src.visualize import plot_predictions
        import matplotlib.pyplot as plt

        fig = plot_predictions(
            pred_data["image"],
            pred_data["true_mask"],
            pred_data["pred_mask"],
            title=region["display_name"],
        )
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    else:
        st.info("Run inference to generate the comparison plot.")

    # --- Methodology ---
    with st.expander("How it works"):
        st.markdown(f"""
        #### Data

        Harmonized Landsat Sentinel-2 (HLS) imagery with 6 spectral bands (Blue, Green, Red, NIR, SWIR1, SWIR2)
        downloaded from [NASA Earthdata](https://www.earthdata.nasa.gov/).
        For each fire, two scenes are fetched: one before the fire and one after, selecting the
        lowest cloud-cover acquisition within the relevant date window.

        #### Labels

        There are no hand-drawn masks. Instead, labels are computed automatically from the
        Differenced Normalized Burn Ratio (dNBR):

        > `NBR = (NIR − SWIR2) / (NIR + SWIR2)`
        > `dNBR = NBR_before − NBR_after`

        Burned vegetation absorbs less NIR and reflects more SWIR than healthy plants, so dNBR spikes
        sharply where fire passed. Any pixel where `dNBR > 0.10` is classified as burned — the USGS
        low-severity threshold, capturing light scorching through to complete burns.

        #### Model

        **[Prithvi-EO-1.0-100M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M)**
        is a Vision Transformer pretrained by IBM and NASA on 640,000 Harmonized Landsat Sentinel-2
        (HLS) scenes — a globally representative dataset spanning all seasons and land cover types.
        The encoder learns rich spectral-spatial representations of Earth's surface.

        For burn scar detection, the Prithvi encoder is paired with an FPN (Feature Pyramid
        Network) decoder. Each 224×224 image patch is:
        1. Passed through the 12-layer ViT encoder
        2. Features extracted from layers 3, 5, 8, and 12 (evenly spaced to capture both
           fine spectral detail and high-level semantic patterns)
        3. Fused via top-down lateral connections (FPN) into a single 14×14 feature map
        4. Upsampled by the decoder (4× transposed-conv stages) → per-pixel burn probability

        Tapping multiple encoder layers improves boundary precision — early layers retain
        spectral/texture detail that deeper layers abstract away.
        Reference: [arXiv:2310.18660](https://arxiv.org/abs/2310.18660)

        #### Train / test split

        Trained on 12 wildfires across 4 US states (CA, OR, NM, WA):
        {", ".join(TRAIN_FIRES)}.

        Three fires are held out entirely for evaluation: Woolsey (2018, CA chaparral),
        East Troublesome (2020, CO subalpine), and Thomas (2017, CA coast). No held-out
        patches appear during training or validation.

        Full scenes are sliced into 224×224 px patches (75% overlap). Patches with at least 1%
        burned pixels are always kept; background-only patches are sampled at 60% to reduce class
        imbalance.
        """)


if __name__ == "__main__":
    main()
