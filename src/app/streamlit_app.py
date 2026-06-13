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
def _fetch_available_dates(bbox: tuple, center_date: str,
                           before_days: int = 14, after_days: int = 14) -> list[dict]:
    """Return S2 scene dates within before_days/after_days of center_date, with cloud cover."""
    import pystac_client
    import planetary_computer
    from datetime import datetime, timedelta
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    dt = datetime.strptime(center_date, "%Y-%m-%d")
    start = (dt - timedelta(days=before_days)).strftime("%Y-%m-%d")
    end = min(dt + timedelta(days=after_days), datetime.utcnow() - timedelta(days=3))
    if end.strftime("%Y-%m-%d") <= start:
        return []
    items = list(catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end.strftime('%Y-%m-%d')}",
        max_items=200,
    ).items())
    by_date: dict = {}
    for item in items:
        d = item.datetime.strftime("%Y-%m-%d")
        cc = round(item.properties.get("eo:cloud_cover", 100), 1)
        if d not in by_date or cc < by_date[d]:
            by_date[d] = cc
    return sorted(
        [{"date": d, "cloud_cover": cc} for d, cc in by_date.items()],
        key=lambda x: x["date"], reverse=True,
    )


@st.cache_data(show_spinner=False)
def _fetch_scene_cached(bbox: tuple, post_date: str) -> dict:
    from src.infer import fetch_preview_tiles
    return fetch_preview_tiles(bbox, post_date)


@st.cache_data(show_spinner=False)
def run_detection(bbox: tuple, post_date: str) -> dict:
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
    today = date.today()
    months = ["January","February","March","April","May","June",
              "July","August","September","October","November","December"]
    col_m, col_d, col_y = st.columns(3)
    with col_m:
        month = st.selectbox("Month", months, index=today.month - 1, label_visibility="collapsed",
                             placeholder="Month", key="date_month")
    with col_d:
        day = st.selectbox("Day", list(range(1, 32)), index=today.day - 1, label_visibility="collapsed",
                           placeholder="Day", key="date_day")
    with col_y:
        years = list(range(2015, today.year + 1))
        year = st.selectbox("Year", years, index=len(years) - 1, label_visibility="collapsed",
                            placeholder="Year", key="date_year")
    import calendar
    max_day = calendar.monthrange(year, months.index(month) + 1)[1]
    day = min(day, max_day)
    try:
        post_date = date(year, months.index(month) + 1, day)
    except ValueError:
        post_date = today
    st.caption(f"Search window end: {post_date.isoformat()}")

    days_ago = (today - post_date).days
    if days_ago < 3:
        st.warning(
            f"Date is {days_ago} day{'s' if days_ago != 1 else ''} ago — "
            "HLS data takes 1–3 days to process and reach the archive. "
            "Try an earlier date."
        )

    # Reset scene selection when window date changes
    if post_date.isoformat() != st.session_state.get("aoi_post_date"):
        st.session_state["aoi_post_date"] = post_date.isoformat()
        st.session_state.pop("aoi_scene_date", None)
        st.session_state.pop("aoi_show_more_dates", None)
        st.session_state.pop("aoi_zoom_count", None)
        st.session_state.pop("scene_preview", None)
        st.session_state.pop("detection_result", None)

    zoom_pending = st.session_state.pop("aoi_zoom_pending", False)

    # --- Base map ---
    preview = st.session_state.get("scene_preview")
    bbox = st.session_state.get("aoi_bbox")

    # Remove fill from drawn shapes so the satellite image shows through.
    st.markdown(
        "<style>.leaflet-pane.leaflet-overlay-pane svg path.leaflet-interactive"
        "{fill:none!important}</style>",
        unsafe_allow_html=True,
    )

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

    detection_early = st.session_state.get("detection_result")
    fg = folium.FeatureGroup(name="overlay")

    # Increment a counter each time zoom is needed so the map key changes,
    # forcing a fresh component render that picks up m.fit_bounds().
    zoom_count = st.session_state.get("aoi_zoom_count", 0)
    if bbox and (preview or detection_early):
        min_lon, min_lat, max_lon, max_lat = bbox
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[20, 20])
    if zoom_pending and bbox:
        zoom_count += 1
        st.session_state["aoi_zoom_count"] = zoom_count

    if detection_early:
        create_sentinel_overlay(detection_early["image"], detection_early["bounds"]).add_to(fg)
        create_folium_overlay(detection_early["pred_mask"], detection_early["bounds"], opacity=0.6).add_to(fg)
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            folium.Rectangle(
                bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                color="#3388ff", weight=2, fill=False,
            ).add_to(fg)
    elif preview and bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{preview['image_b64']}",
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            opacity=1.0,
        ).add_to(fg)
        folium.Rectangle(
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            color="#3388ff", weight=2, fill=False,
        ).add_to(fg)

    out = st_folium(
        m, key=f"draw_map_{zoom_count}",
        feature_group_to_add=fg,
        use_container_width=True, height=460,
        returned_objects=["last_active_drawing"],
    )

    draw = (out or {}).get("last_active_drawing")
    if draw and draw.get("geometry", {}).get("type") == "Polygon":
        ring = draw["geometry"]["coordinates"][0]
        lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
        new_bbox = (min(lons), min(lats), max(lons), max(lats))
        if new_bbox != st.session_state.get("aoi_bbox"):
            st.session_state["aoi_bbox"] = new_bbox
            st.session_state.pop("aoi_scene_date", None)
            st.session_state.pop("aoi_show_more_dates", None)
            st.session_state.pop("aoi_zoom_count", None)
            st.session_state.pop("scene_preview", None)
            st.session_state.pop("detection_result", None)
            preview = None

    bbox = st.session_state.get("aoi_bbox")
    if not bbox:
        st.info("Draw a polygon or rectangle on the map to define your area of interest.")
        return

    area = _aoi_area_km2(bbox)
    if area > 10000:
        st.warning("Large area — download and inference will be slow. A smaller AOI (under ~10,000 km²) is recommended.")

    # --- Scene date selection ---
    show_more = st.session_state.get("aoi_show_more_dates", False)
    before, after = (60, 14) if show_more else (14, 14)
    with st.spinner("Checking available scenes…"):
        available = _fetch_available_dates(bbox, post_date.isoformat(), before, after)

    if not available:
        st.warning("No Sentinel-2 scenes found within 2 weeks of this date for this area.")
        if not show_more:
            if st.button("Search wider window (±60 days)"):
                st.session_state["aoi_show_more_dates"] = True
                st.rerun()
        return

    def _scene_label(s):
        cc = s["cloud_cover"]
        icon = "🟢" if cc <= 20 else ("🟡" if cc <= 50 else "🔴")
        return f"{s['date']}  {icon}  {cc:.0f}% cloud cover"

    scene_dates = [s["date"] for s in available]
    best_idx = min(range(len(available)), key=lambda i: available[i]["cloud_cover"])
    stored = st.session_state.get("aoi_scene_date")
    default_idx = scene_dates.index(stored) if stored in scene_dates else best_idx

    selected_scene = st.selectbox(
        "Available Sentinel-2 scenes",
        options=scene_dates,
        index=default_idx,
        format_func=lambda d: _scene_label(next(s for s in available if s["date"] == d)),
        help="Scenes from the 60 days before your window date. 🟢 ≤20% cloud · 🟡 20–50% · 🔴 >50%",
    )

    if selected_scene != st.session_state.get("aoi_scene_date"):
        st.session_state["aoi_scene_date"] = selected_scene
        st.session_state.pop("scene_preview", None)
        st.session_state.pop("detection_result", None)

    if not show_more:
        if st.button("Show more dates (±60 days)", use_container_width=False):
            st.session_state["aoi_show_more_dates"] = True
            st.rerun()
    else:
        if st.button("Show fewer dates (±2 weeks)", use_container_width=False):
            st.session_state["aoi_show_more_dates"] = False
            st.rerun()

    rounded = (tuple(round(b, 4) for b in bbox), selected_scene)
    preview = st.session_state.get("scene_preview")

    if preview and preview.get("_key") != rounded:
        preview = None
        st.session_state.pop("scene_preview", None)
        st.session_state.pop("detection_result", None)

    # --- Stage 1: Preview ---
    if preview is None:
        st.caption(f"AOI ≈ {area:,.0f} km²")
        if st.button("🛰 Preview satellite scene", type="secondary"):
            try:
                with st.spinner("Fetching scene…"):
                    p = _fetch_scene_cached(*rounded)
                    p["_key"] = rounded
                    st.session_state["scene_preview"] = p
                    st.session_state["aoi_zoom_pending"] = True
                    st.session_state.pop("detection_result", None)
                st.rerun()
            except Exception as e:
                st.error(f"Could not load scene: {e}")
        return

    cc = preview.get("cloud_cover", "?")
    st.caption(
        f"Scene acquired **{preview['scene_date']}** · {cc}% cloud cover · AOI ≈ {area:,.0f} km²"
    )
    if isinstance(cc, (int, float)) and cc > 20:
        st.warning(f"Cloud cover is {cc}% — image may be partially obscured.")

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("← Change area / date"):
            st.session_state.pop("scene_preview", None)
            st.session_state.pop("detection_result", None)
            st.rerun()

    # --- Stage 2: Detection ---
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
        """
        <style>
        header[data-testid="stHeader"] { display: none; }
        footer { display: none; }
        .block-container {
            padding-top: 0.4rem !important;
            padding-bottom: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .stVerticalBlock { gap: 0.4rem !important; }
        [data-testid='stMetricValue'] { font-size: 1.6rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    overlay_opacity = 0.6

    # ── Sidebar: app description + metrics (filled after fire selection) ──────
    with st.sidebar:
        st.markdown(
            "<h3 style='margin:0 0 6px 0'>Wildfire Burn Scar Detection</h3>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Fine-tuned [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) "
            "(IBM × NASA ViT-Large, 300M params) on 37 wildfires across 5 US states. "
            "Evaluated on 4 held-out California fires never seen during training."
        )
    metrics_placeholder = st.sidebar.empty()

    # ── Main area: mode + fire selector above map ─────────────────────────────
    col_mode, col_fire = st.columns([1, 2])
    with col_mode:
        mode = st.radio(
            "Mode",
            ["📍 Held-out test fires", "✏️ Detect on custom area"],
            label_visibility="collapsed",
        )
    with col_fire:
        if mode.endswith("test fires"):
            region_key = st.selectbox(
                "Fire",
                options=list(REGIONS.keys()),
                index=list(REGIONS.keys()).index(DEFAULT_REGION),
                format_func=lambda k: REGIONS[k]["display_name"],
                label_visibility="collapsed",
            )

    if mode.endswith("custom area"):
        custom_detection_view()
        return

    region = REGIONS[region_key]
    pred_data = load_prediction(region["name"])

    # ── Fill sidebar metrics now that we have pred_data ───────────────────────
    with metrics_placeholder.container():
        st.divider()
        if pred_data is not None:
            pred = pred_data["pred_mask"]
            true = pred_data["true_mask"]
            total = pred.size
            tp = int(((pred == 1) & (true == 1)).sum())
            fp = int(((pred == 1) & (true == 0)).sum())
            fn = int(((pred == 0) & (true == 1)).sum())
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

            st.caption(f"**{region['display_name']}** — held-out, never seen during training")
            c1, c2, c3 = st.columns(3)
            c1.metric("Recall",    f"{recall    * 100:.0f}%",
                      help="Of all dNBR-burned pixels, what fraction the model detected.")
            c2.metric("Precision", f"{precision * 100:.0f}%",
                      help="Of all model-flagged pixels, what fraction were truly burned.")
            c3.metric("IoU",       f"{iou       * 100:.0f}%",
                      help="Intersection over Union — overlap between prediction and dNBR reference.")
            st.caption(
                f"Model: {pred.sum() / total * 100:.1f}% burned · "
                f"dNBR ref: {true.sum() / total * 100:.1f}% burned"
            )
        else:
            st.info(
                "Predictions not available.\n\n"
                f"```bash\npython run_inference.py --region {region['name']}\n```"
            )

    # ── Map ───────────────────────────────────────────────────────────────────
    with st.spinner("Loading satellite imagery..."):
        m = create_map(region, pred_data, overlay_opacity)
    st_folium(m, use_container_width=True, height=660, returned_objects=[])

    # --- Comparison plot (collapsed by default) ---
    with st.expander("Prediction comparison"):
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

        **[Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)**
        is a ViT-Large pretrained by IBM and NASA on 640,000 Harmonized Landsat Sentinel-2
        (HLS) scenes — a globally representative dataset spanning all seasons and land cover types.
        The encoder learns rich spectral-spatial representations of Earth's surface.

        For burn scar detection, the Prithvi encoder is paired with an FPN (Feature Pyramid
        Network) decoder. Each 224×224 image patch is:
        1. Passed through the 24-layer ViT-Large encoder
        2. Features extracted from layers 6, 12, 18, and 24 (evenly spaced to capture both
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
