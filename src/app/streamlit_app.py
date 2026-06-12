import math
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw, Geocoder
from streamlit_folium import st_folium

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
    "palisades_fire_2025": {
        "display_name": "Palisades Fire (2025)",
        "name": "palisades_fire_2025",
        "center": [34.05, -118.55],
        "zoom": 12,
        "description": (
            "Burned ~23,400 acres across Pacific Palisades and Malibu, destroying "
            "thousands of structures in one of the most destructive fires in Los Angeles "
            "history. January 2025."
        ),
        "acres": "23,400",
        "date": "January 2025",
        "setting": "Southern California coastal sage scrub & WUI",
    },
    "eaton_fire_2025": {
        "display_name": "Eaton Fire (2025)",
        "name": "eaton_fire_2025",
        "center": [34.19, -118.07],
        "zoom": 12,
        "description": (
            "Burned ~14,100 acres in Altadena and the Pasadena foothills, causing "
            "widespread destruction in a densely populated urban-wildland interface. "
            "January 2025."
        ),
        "acres": "14,100",
        "date": "January 2025",
        "setting": "Southern California foothill chaparral & WUI",
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
    # NorCal / Sierra
    "August Complex (2020)", "Mendocino Complex (2018)", "SCU Lightning Complex (2020)",
    "Caldor (2021)", "LNU Lightning Complex (2020)", "North Complex (2020)",
    "Carr (2018)", "Dixie (2021)", "Antelope (2021)", "Bootleg (2021, OR)",
    "Pearl Hill (2020, WA)", "Mosquito (2022)", "Monument (2021)",
    "River/Carmel (2020)", "Camp Fire (2018)", "Tubbs (2017)",
    "Kincade (2019)", "Glass (2020)",
    # SoCal chaparral (domain-matching the test fires)
    "Bobcat (2020)", "Holy (2018)", "Apple (2020)", "Cranston (2018)",
    "Saddleridge (2019)", "El Dorado (2020)", "Valley (2020)", "Lake (2020)",
    "Blue Ridge (2020)", "Bond (2020)", "La Tuna (2017)",
    # Colorado Rockies
    "Cameron Peak (2020)", "Calwood (2020)", "Spring Creek (2018)",
    # Arizona
    "Bighorn (2020)", "Bush (2020)", "Telegraph (2021)",
    # PNW
    "Holiday Farm (2020)", "Beachie Creek (2020)",
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

    if pred_data is not None:
        create_sentinel_overlay(pred_data["image"], pred_data["bounds"]).add_to(m)
        if pred_data.get("dnbr") is not None:
            create_severity_overlay(
                pred_data["dnbr"], pred_data["bounds"], show=False
            ).add_to(m)
        create_folium_overlay(
            pred_data["pred_mask"], pred_data["bounds"], opacity=overlay_opacity
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# --- Custom AOI live detection (post-fire only) ---
@st.cache_resource(show_spinner=False)
def _load_model():
    from src.infer import load_model
    return load_model()


@st.cache_data(show_spinner=False)
def _fetch_scene_cached(bbox: tuple, post_date: str) -> dict:
    """Find the best Sentinel-2 scene and return a tile URL — no data download."""
    from src.infer import fetch_preview_tiles
    return fetch_preview_tiles(bbox, post_date)


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
        "**1.** Search or zoom to a location · **2.** Draw a rectangle on the map · "
        "**3.** Pick a post-fire date · **4.** Preview the satellite scene · **5.** Run detection.  \n"
        "_The model runs on a single post-fire HLS scene — no before/after comparison._"
    )
    post_date = st.date_input(
        "Post-fire date",
        value=date.today(), min_value=date(2015, 7, 1), max_value=date.today(),
        format="YYYY-MM-DD",
        help="The app finds the least-cloudy Sentinel-2 scene within ~30 days after this date.",
    )

    # --- Base map (never reloads — preview/detection layers injected dynamically) ---
    preview = st.session_state.get("scene_preview")
    bbox = st.session_state.get("aoi_bbox")

    m = folium.Map(location=[39.0, -120.5], zoom_start=6, tiles=None, control_scale=True)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite",
    ).add_to(m)
    Geocoder(collapsed=False, add_marker=False).add_to(m)
    Draw(
        draw_options={"polyline": False, "circle": False, "marker": False,
                      "circlemarker": False, "polygon": True, "rectangle": True},
        edit_options={"edit": False, "remove": True},
    ).add_to(m)

    # Build feature group to inject without reloading the map
    detection_early = st.session_state.get("detection_result")
    fg = folium.FeatureGroup(name="overlay")
    map_center = None
    map_zoom = None
    import math

    active_bounds = None
    if detection_early and bbox:
        active_bounds = detection_early["bounds"]
    elif preview and bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        active_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    if active_bounds:
        s, w = active_bounds[0]; n, e = active_bounds[1]
        map_center = ((s + n) / 2, (w + e) / 2)
        span = max(n - s, e - w)
        map_zoom = max(1, min(14, int(math.log2(360 / span)) - 1))

    if detection_early:
        create_sentinel_overlay(detection_early["image"], detection_early["bounds"]).add_to(fg)
        create_folium_overlay(detection_early["pred_mask"], detection_early["bounds"], opacity=0.6).add_to(fg)
    elif preview and bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{preview['image_b64']}",
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=1.0,
        ).add_to(fg)

    out = st_folium(
        m, key="draw_map",
        feature_group_to_add=fg,
        center=map_center,
        zoom=map_zoom,
        use_container_width=True, height=500,
        returned_objects=["last_active_drawing"],
    )

    # Capture drawn shape → bbox; clear preview if AOI/date changed.
    draw = (out or {}).get("last_active_drawing")
    if draw and draw.get("geometry", {}).get("type") == "Polygon":
        ring = draw["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
        new_bbox = (min(lons), min(lats), max(lons), max(lats))
        if new_bbox != st.session_state.get("aoi_bbox"):
            st.session_state["aoi_bbox"] = new_bbox
            st.session_state.pop("scene_preview", None)
            st.session_state.pop("detection_result", None)
            preview = None

    bbox = st.session_state.get("aoi_bbox")
    if not bbox:
        st.info("Draw a polygon or rectangle on the map to define your area of interest.")
        return

    area = _aoi_area_km2(bbox)
    rounded = (tuple(round(b, 4) for b in bbox), post_date.isoformat())
    preview = st.session_state.get("scene_preview")

    # Invalidate preview if bbox or date changed.
    if preview and preview.get("_key") != rounded:
        preview = None
        st.session_state.pop("scene_preview", None)
        st.session_state.pop("detection_result", None)

    if area > 10000:
        st.warning("Large area — download and inference will be slow. A smaller AOI "
                   "(under ~10,000 km²) is recommended.")

    # --- Stage 1: Preview scene ---
    if preview is None:
        st.caption(f"AOI ≈ {area:,.0f} km²")
        if st.button("🛰 Preview satellite scene", type="secondary"):
            try:
                with st.spinner("Finding best Sentinel-2 scene…"):
                    p = _fetch_scene_cached(*rounded)
                    p["_key"] = rounded
                    st.session_state["scene_preview"] = p
                    st.session_state.pop("detection_result", None)
                st.rerun()
            except Exception as e:
                st.error(f"Could not find a scene: {e}")
        return

    cc = preview.get("cloud_cover", "?")
    st.caption(
        f"Scene acquired **{preview['scene_date']}** · {cc}% cloud cover · AOI ≈ {area:,.0f} km²"
    )
    if isinstance(cc, (int, float)) and cc > 30:
        st.warning(f"Cloud cover is {cc}% — image may be partially obscured. Try a different date.")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Change area / date"):
            st.session_state.pop("scene_preview", None)
            st.session_state.pop("detection_result", None)
            st.rerun()

    # --- Stage 2: Run detection ---
    detection = st.session_state.get("detection_result")

    if detection is None:
        with col2:
            if st.button("🔥 Run burn scar detection", type="primary"):
                try:
                    with st.spinner("Running the model (~1–2 min)…"):
                        res = run_detection(*rounded)
                        st.session_state["detection_result"] = res
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection failed: {e}")
        return

    st.success(
        f"Detected burn scar over ~{detection['burned_frac'] * 100:.1f}% of the area. "
        f"HLS acquired **{detection['scene_date']}** · {detection['n_scenes']} granule(s)."
    )


def main():
    st.markdown(
        "<style>[data-testid='stMetricValue'] { font-size: 1.6rem; }</style>",
        unsafe_allow_html=True,
    )
    st.title("Wildfire Burn Scar Detection")
    st.caption(
        "Prithvi-EO-1.0-100M geospatial foundation model (IBM × NASA) "
        "fine-tuned on 37 wildfires across 5 US states, evaluated on 4 held-out California fires."
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
                f"The model was trained on 37 wildfires across 5 US states and has "
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
        For each training fire, two scenes are fetched: one before the fire and one after, selecting the
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

        Trained on 37 wildfires across 5 US states (CA, OR, AZ, NM, WA):
        {", ".join(TRAIN_FIRES)}.

        Four California fires are held out entirely for evaluation: Palisades (2025, LA coastal WUI),
        Eaton (2025, Pasadena foothill WUI), Woolsey (2018, SoCal chaparral), and Thomas (2017, CA coast).
        No held-out patches appear during training or validation.

        Full scenes are sliced into 224×224 px patches (75% overlap). Patches with at least 1%
        burned pixels are always kept; background-only patches are sampled at 60% to reduce class
        imbalance.

        #### Brightness correction

        HLS surface reflectance (LaSRC atmospheric correction + BRDF normalization) runs
        ~1.4–1.9× darker than the HLS distribution Prithvi-EO was pretrained on. Fed to the
        frozen encoder, this dark input biases features toward the low-NIR burn signature and
        floods burn predictions. A fixed per-band **brightness gain** — calibrated so the pooled
        training-fire median reflectance matches the Prithvi pretraining mean — corrects this
        before z-scoring. This lifted Woolsey IoU from 0.53 → 0.73 and macro IoU from 0.54 → 0.64.
        A deterministic **NDWI water mask** (green > NIR) removes spurious burn predictions over
        open water. Both corrections use only training-fire statistics — no test-fire leakage.
        """)


if __name__ == "__main__":
    main()
