import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
import numpy as np
import streamlit as st
from folium.plugins import Draw, Geocoder
from streamlit_folium import st_folium

from src.limits import MAX_AOI_KM2, SLOW_AOI_KM2  # hard cap / slow-run warning (see src/limits.py)
from src.visualize import (
    SEVERITY_CLASSES,
    create_folium_overlay,
    create_sentinel_overlay,
    create_severity_overlay,
)

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, .stApp, p, h1, h2, h3, h4, h5, h6, label, li, a, button, input, select, textarea,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"],
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
/* Never override Streamlit's icon font (expander arrows, etc.) */
[data-testid="stIconMaterial"], span[class*="material"] {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
header[data-testid="stHeader"], footer { display: none; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 100% !important;
    padding: 1.25rem 2rem 1rem 2rem !important;
}
.stVerticalBlock { gap: 0.6rem !important; }
[data-testid="stSidebar"] {
    background: #f6f7f9;
    border-right: 1px solid #e3e6ea;
}
[data-testid="stSidebar"] h3 {
    font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em; margin: 0 0 6px 0;
}
.fire-title { font-size: 0.95rem; font-weight: 600; color: #1a1d21; line-height: 1.3; }
.fire-sub { font-size: 0.78rem; color: #5b6470; line-height: 1.3; margin: 1px 0 10px 0; }
.fact-list { padding-bottom: 14px; }
[data-testid="stSidebar"] hr { margin: 0.5rem 0 1.25rem 0 !important; }
.fact-row { display: grid; grid-template-columns: 5.2rem 1fr; column-gap: 8px;
            font-size: 0.86rem; line-height: 1.35; padding: 3px 0; }
.fact-key { color: #5b6470; }
.fact-val { color: #1a1d21; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.metric-row { display: grid; grid-template-columns: 5.2rem 1fr; column-gap: 8px;
              align-items: center; padding: 11px 0; border-top: 1px solid #e3e6ea; }
.metric-label { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 4px; }
.metric-name { font-size: 0.8rem; font-weight: 500; line-height: 1.2; color: #5b6470; }
.metric-value { font-size: 1.3rem; font-weight: 600; line-height: 1.2;
                font-variant-numeric: tabular-nums; color: #1a1d21; }
.metric-desc { font-size: 0.86rem; line-height: 1.35; color: #5b6470; }
.metric-foot { font-size: 0.76rem; color: #5b6470; padding: 10px 0 14px 0; line-height: 1.5;
               border-top: 1px solid #e3e6ea; }
.stButton > button {
    border-radius: 4px; border: 1px solid #c9ced6; background: #fff;
    font-weight: 500; box-shadow: none;
}
.stButton > button:hover { border-color: #1f4e79; color: #1f4e79; }
.stButton > button[kind="primary"] { background: #1f4e79; border-color: #1f4e79; color: #fff; }
.stButton > button[kind="primary"]:hover { background: #173b5c; color: #fff; }
[data-testid="stAlert"] {
    background: #f6f7f9; border: 1px solid #e3e6ea; border-radius: 4px; color: #1a1d21;
}
[data-testid="stExpander"] { border: 1px solid #e3e6ea; border-radius: 4px; box-shadow: none; }
.leaflet-pane.leaflet-overlay-pane svg path.leaflet-interactive { fill: none !important; }
iframe[title="streamlit_folium.st_folium"] { width: 100% !important; }
[data-testid="stSidebar"] { min-width: 380px; }
</style>
"""
REGIONS = {
    "woolsey_fire_2018": {
        "display_name": "Woolsey Fire (2018)",
        "location": "Ventura and Los Angeles counties, CA",
        "dates": "Nov 8 – Nov 20, 2018",
        "name": "woolsey_fire_2018",
        "center": [34.16, -118.83],
        "zoom": 11,
        "description": (
            "Burned 96,949 acres across Ventura and Los Angeles counties, "
            "destroying over 1,600 structures and threatening Malibu and Calabasas. "
            "November 2018."
        ),
        "acres": "89,551",
        "date": "November 2018",
        "setting": "Southern California chaparral",
    },
    "palisades_fire_2025": {
        "display_name": "Palisades Fire (2025)",
        "location": "Pacific Palisades, Los Angeles, CA",
        "dates": "Jan 7 – Jan 31, 2025",
        "name": "palisades_fire_2025",
        "center": [34.05, -118.55],
        "zoom": 12,
        "description": (
            "Burned ~23,400 acres across Pacific Palisades and Malibu, destroying "
            "thousands of structures in one of the most destructive fires in Los Angeles "
            "history. January 2025."
        ),
        "acres": "23,449",
        "date": "January 2025",
        "setting": "Southern California coastal sage scrub & WUI",
    },
    "eaton_fire_2025": {
        "display_name": "Eaton Fire (2025)",
        "location": "Altadena, Los Angeles County, CA",
        "dates": "Jan 8 – Jan 31, 2025",
        "name": "eaton_fire_2025",
        "center": [34.19, -118.07],
        "zoom": 12,
        "description": (
            "Burned ~14,100 acres in Altadena and the Pasadena foothills, causing "
            "widespread destruction in a densely populated urban-wildland interface. "
            "January 2025."
        ),
        "acres": "14,056",
        "date": "January 2025",
        "setting": "Southern California foothill chaparral & WUI",
    },
    "thomas_fire_2017": {
        "display_name": "Thomas Fire (2017)",
        "location": "Ventura and Santa Barbara counties, CA",
        "dates": "Dec 4, 2017 – Jan 12, 2018",
        "name": "thomas_fire_2017",
        "center": [34.48, -119.25],
        "zoom": 10,
        "description": (
            "Burned 281,893 acres across Ventura and Santa Barbara counties over "
            "several weeks — among the largest California wildfires on record at the "
            "time. December 2017."
        ),
        "acres": "281,791",
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
PERIMETER_DIR = Path(__file__).resolve().parents[2] / "assets" / "perimeters"


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


def _perimeter_bounds(region_name: str):
    """[[south, west], [north, east]] of the CAL FIRE perimeter, or None."""
    path = PERIMETER_DIR / f"{region_name}.geojson"
    if not path.exists():
        return None
    lons, lats = [], []

    def walk(c):
        if c and isinstance(c[0], (int, float)):
            lons.append(c[0]); lats.append(c[1])
        else:
            for x in c:
                walk(x)

    for feat in json.loads(path.read_text()).get("features", []):
        walk(feat["geometry"]["coordinates"])
    if not lons:
        return None
    return [[min(lats), min(lons)], [max(lats), max(lons)]]


def _focus_bounds(region: dict, pred_data):
    """What the map frames: the extent of the HLS tile that was run."""
    return pred_data["bounds"] if pred_data else _perimeter_bounds(region["name"])


def create_map(region: dict, pred_data: dict | None, overlay_opacity: float) -> folium.Map:
    m = folium.Map(location=region["center"], zoom_start=region["zoom"], tiles=None,
                   zoom_snap=0.25)
    focus = _focus_bounds(region, pred_data)
    if focus is not None:
        m.fit_bounds(focus, padding=(10, 10))

    # Satellite is the only basemap, so it is kept out of the layer control.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=False,
    ).add_to(m)

    if pred_data is not None:
        # Layer-control order follows add order: HLS and predictions first (on),
        # then the reference layers (off by default).
        create_sentinel_overlay(pred_data["image"], pred_data["bounds"]).add_to(m)
        create_folium_overlay(
            pred_data["pred_mask"], pred_data["bounds"], opacity=overlay_opacity
        ).add_to(m)
        if pred_data.get("dnbr") is not None:
            create_severity_overlay(
                pred_data["dnbr"], pred_data["bounds"], show=False
            ).add_to(m)

    perimeter = PERIMETER_DIR / f"{region['name']}.geojson"
    if perimeter.exists():
        folium.GeoJson(
            json.loads(perimeter.read_text()),
            name="CAL FIRE perimeter",
            style_function=lambda _f: {"color": "#ffd24a", "weight": 2.5, "fillOpacity": 0},
            highlight_function=None,
            show=False,
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


def _fit_map_to_window(bounds=None):
    """Stretch the st_folium iframe, and the Leaflet map inside it, to the bottom of the
    browser window. st_folium takes a fixed pixel height, so this measures the real window
    from the page and resizes live (also on window resize). If `bounds` is given, the map
    is re-framed to them once it has its final size, so zoom matches the real window."""
    script = """
        <script>
        const BOUNDS = __BOUNDS__;
        // On Hugging Face the whole app sits inside another iframe that grows to fit its
        // content. Sizing the map from this window's height then feeds back on itself
        // (taller map -> taller frame -> taller map...), so cap it by the physical screen.
        const embedded = (() => { try { return window.parent.parent !== window.parent; } catch (e) { return true; } })();
        const fit = () => {
          const pw = window.parent;
          const winH = embedded ? Math.min(pw.innerHeight, screen.availHeight - 170) : pw.innerHeight;
          pw.document.querySelectorAll('iframe').forEach(f => {
            try {
              const d = f.contentDocument;
              const m = d && (d.getElementById('map_div') || d.querySelector('.folium-map'));
              if (!m) return;
              const h = Math.max(300, winH - f.getBoundingClientRect().top - 12);
              f.style.height = h + 'px';
              m.style.height = h + 'px';
              d.documentElement.style.height = h + 'px';
              d.body.style.margin = '0';
              if (f._fitH !== h) {
                f._fitH = h;
                const lm = f.contentWindow.map;
                if (lm && lm.invalidateSize) lm.invalidateSize();
                if (lm && BOUNDS && lm.fitBounds) lm.fitBounds(BOUNDS, {padding: [10, 10]});
              }
            } catch (e) {}
          });
        };
        fit();
        setInterval(fit, 300);
        window.parent.addEventListener('resize', fit);
        </script>
        """
    script = script.replace("__BOUNDS__", json.dumps(bounds))
    if hasattr(st, "iframe"):
        st.iframe(script, height=1)
    else:  # older Streamlit
        import streamlit.components.v1 as components
        components.html(script, height=0)


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def _fmt_seconds(sec: float) -> str:
    sec = max(0, int(round(sec)))
    if sec < 90:
        return f"{sec} s"
    return f"{sec // 60} min {sec % 60:02d} s"


def _run_with_progress(bbox: tuple, post_date: str, estimate: dict) -> dict:
    """Run detection in a worker thread while this thread redraws a live progress bar:
    stage, bytes received out of the exact planned total, speed, and a smoothed
    time-remaining several times a second."""
    import collections
    import threading
    import time

    from src import netcount
    from src.infer import detect_burn_scar

    netcount.start()
    with st.spinner("Loading model..."):
        model, device, cfg = _load_model()

    state = {"stage": "search", "done": 0, "total": 0, "infer_t0": None,
             "planned_bytes": None, "result": None, "error": None}

    def cb(stage, done, total, info=None):
        if info and info.get("planned_bytes"):
            state["planned_bytes"] = info["planned_bytes"]
        if stage == "infer" and state["infer_t0"] is None:
            state["infer_t0"] = time.time()
        state.update(stage=stage, done=done, total=total)

    def work():
        try:
            state["result"] = detect_burn_scar(bbox, post_date, model, device, cfg, progress=cb)
        except Exception as e:
            state["error"] = e

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    bar = st.progress(0.0, text="Starting")
    detail = st.empty()

    short = collections.deque()       # last ~3 s of (time, bytes): shown speed
    long = collections.deque()        # last ~20 s: speed used for the time estimate
    dl_base = None                    # network counter when the download began
    dl_t0 = None
    dl_bytes = 0.0
    last_data_t = time.time()
    smooth = None                     # smoothed seconds remaining
    last_tick = time.time()

    def rate(window):
        if len(window) < 2:
            return 0.0
        return (window[-1][1] - window[0][1]) / max(window[-1][0] - window[0][0], 1e-6)

    def raw_remaining(stage, done, total, speed_long):
        """Unsmoothed seconds left. The download part is bytes left / recent speed;
        everything not yet measured falls back to the stored per-device averages."""
        infer_est = estimate["infer_s"]
        planned = state["planned_bytes"] and state["planned_bytes"] * 1.05
        if stage in ("search", "plan"):
            return estimate["seconds"]
        if stage == "download":
            if planned and dl_t0 is not None and time.time() - dl_t0 > 6 and speed_long > 1e5:
                left = max(planned - dl_bytes, 0) / speed_long
            elif planned and dl_bytes < planned:
                left = estimate["download_s"] * (1 - dl_bytes / planned)
            else:
                left = estimate["download_s"] * max(total - done, 0) / max(total, 1)
            return left + 4 + infer_est
        if stage == "prepare":
            return infer_est
        if done == 0:
            return infer_est
        return (time.time() - state["infer_t0"]) / done * (total - done)

    while thread.is_alive():
        now = time.time()
        stage, done, total = state["stage"], state["done"], state["total"]
        cur = netcount.total()  # bytes this app received (not the whole machine)
        for dq, span in ((short, 3), (long, 20)):
            dq.append((now, cur))
            while dq and now - dq[0][0] > span:
                dq.popleft()
        speed = rate(short)
        if speed > 50_000:
            last_data_t = now

        amount = ""
        line = []
        if stage == "search":
            frac, what = 0.02, "Searching for scenes"
        elif stage == "plan":
            frac = 0.03 + 0.02 * (done / total if total else 0)
            what = f"Planning download: scene {min(done + 1, total)} of {total}"
        elif stage == "download":
            if dl_base is None:
                dl_base, dl_t0, last_data_t = cur, now, now
                long.clear()
            dl_bytes = cur - dl_base
            planned = state["planned_bytes"] and state["planned_bytes"] * 1.05  # ~5% HTTP overhead, measured
            if planned:
                frac = 0.05 + 0.20 * min(dl_bytes / planned, 1.0)
                amount = f"{_fmt_bytes(dl_bytes)} / {_fmt_bytes(max(planned, dl_bytes))}"
            else:
                frac = 0.05 + 0.20 * (done / total if total else 0)
                amount = f"{_fmt_bytes(dl_bytes)} received"
            what = (f"Downloading imagery, scene {min(done + 1, total)} of {total}"
                    if total and done < total else "Merging scenes")
            line.append(f"{_fmt_bytes(speed)}/s")
            idle = now - last_data_t
            if done < total and idle > 8:
                line.append(f"no data for {_fmt_seconds(idle)}, may be stalled")
        elif stage == "prepare":
            frac, what = 0.27, "Preparing image"
        else:
            frac = 0.30 + 0.70 * (done / total if total else 0)
            what = "Running model"

        # Smooth the time remaining: count down in real time and only nudge toward each
        # new raw estimate, so the number falls steadily and doesn't jump around.
        raw = raw_remaining(stage, done, total, rate(long))
        dt = now - last_tick
        last_tick = now
        if smooth is None:
            smooth = raw
        else:
            smooth = max(smooth - dt, 0.0)
            smooth += 0.04 * (raw - smooth)
        shown = smooth if smooth < 60 else round(smooth / 5) * 5
        line.append(f"about {_fmt_seconds(shown)} remaining")

        bar.progress(min(frac, 1.0), text=what)
        detail.caption((amount + "  \n" if amount else "") + "  ·  ".join(line))
        time.sleep(0.25)
    detail.empty()
    bar.empty()
    if state["error"] is not None:
        raise state["error"]
    return state["result"]


def _aoi_area_km2(bbox: tuple) -> float:
    minlon, minlat, maxlon, maxlat = bbox
    midlat = math.radians((minlat + maxlat) / 2)
    w = (maxlon - minlon) * 111.32 * math.cos(midlat)
    h = (maxlat - minlat) * 110.57
    return abs(w * h)


MAP_CONTROL_CSS = """
<style>
.leaflet-control-scale-line {
    background: rgba(255,255,255,0.9); border: 1px solid #5b6470; border-top: none;
    color: #1a1d21; font: 11px/1.2 Inter, -apple-system, 'Segoe UI', sans-serif;
    padding: 1px 6px; margin: 0 0 10px 10px; box-shadow: none; text-shadow: none;
}
.leaflet-control-geocoder { border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,.25); }
.leaflet-control-geocoder-form input {
    font: 13px Inter, -apple-system, 'Segoe UI', sans-serif; padding: 6px 8px;
}
.leaflet-control-geocoder-alternatives li { font: 13px Inter, sans-serif; }
.leaflet-bar a { border-radius: 0; }
</style>
"""


def _style_map_controls(m: folium.Map):
    """Metric-only minimal scale bar and cleaner control styling."""
    from branca.element import Element, MacroElement, Template
    m.get_root().header.add_child(Element(MAP_CONTROL_CSS))
    scale = MacroElement()
    scale._template = Template(
        "{% macro script(this, kwargs) %}"
        "L.control.scale({imperial: false, position: 'bottomleft', maxWidth: 120})"
        ".addTo({{ this._parent.get_name() }});"
        "{% endmacro %}"
    )
    m.add_child(scale)


def custom_detection_view(sidebar_slot):
    # The map owns the main panel; every control and status message lives in the
    # sidebar so the main area is exactly one full-height map.
    main_slot = st.container()
    with sidebar_slot:
        today = date.today()
        default_date = today - timedelta(days=3)   # HLS needs ~3 days to reach the archive
        months = ["January","February","March","April","May","June",
                  "July","August","September","October","November","December"]
        col_m, col_d, col_y = st.columns(3)
        with col_m:
            month = st.selectbox("Month", months, index=default_date.month - 1, label_visibility="collapsed",
                                 placeholder="Month", key="date_month")
        with col_d:
            day = st.selectbox("Day", list(range(1, 32)), index=default_date.day - 1, label_visibility="collapsed",
                               placeholder="Day", key="date_day")
        with col_y:
            years = list(range(2015, today.year + 1))
            year = st.selectbox("Year", years, index=years.index(default_date.year), label_visibility="collapsed",
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

        m = folium.Map(location=[39.0, -120.5], zoom_start=6, tiles=None)
        _style_map_controls(m)
        folium.TileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri", name="Satellite", control=False,
        ).add_to(m)
        # Photon: much faster and typo-tolerant than the default Nominatim, with live suggestions.
        Geocoder(collapsed=False, add_marker=False, provider="photon", zoom=11,
                 placeholder="Search for a place", suggestMinLength=3,
                 suggestTimeout=200, errorMessage="No results").add_to(m)
        Draw(
            draw_options={"polyline": False, "circle": False, "marker": False,
                          "circlemarker": False, "polygon": True, "rectangle": True},
            edit_options={"edit": False, "remove": True},
        ).add_to(m)

        detection_early = st.session_state.get("detection_result")
        fg = folium.FeatureGroup(name="overlay", control=False)

        # Increment a counter each time zoom is needed so the map key changes,
        # forcing a fresh component render that picks up m.fit_bounds().
        zoom_count = st.session_state.get("aoi_zoom_count", 0)
        if bbox and (preview or detection_early):
            min_lon, min_lat, max_lon, max_lat = bbox
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=[20, 20])
        if zoom_pending and bbox:
            zoom_count += 1
            st.session_state["aoi_zoom_count"] = zoom_count

        groups = [fg]
        layer_control = None
        if detection_early:
            # Separate groups so the prediction can be toggled independently of the imagery.
            fg_hls = folium.FeatureGroup(name="HLS RGB")
            fg_pred = folium.FeatureGroup(name="Burn Scar Predictions")
            create_sentinel_overlay(detection_early["image"], detection_early["bounds"], fixed_scale=True).add_to(fg_hls)
            create_folium_overlay(detection_early["pred_mask"], detection_early["bounds"], opacity=0.6).add_to(fg_pred)
            groups = [fg_hls, fg_pred, fg]
            layer_control = folium.LayerControl(collapsed=False)
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

        with main_slot:
            out = st_folium(
                m, key=f"draw_map_{zoom_count}",
                feature_group_to_add=groups,
                layer_control=layer_control,
                use_container_width=True, height=700,
                returned_objects=["last_active_drawing"],
            )
            _fit_map_to_window()

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
            st.info("Draw a polygon or rectangle on the map to define your area of interest, or choose a test fire from the list below.")
            return

        area = _aoi_area_km2(bbox)
        if area > MAX_AOI_KM2:
            st.error(f"Area too large ({area:,.0f} km²). The limit here is ~{MAX_AOI_KM2:,} km² "
                     "to stay within memory — please draw a smaller box.")
            return
        if area > SLOW_AOI_KM2:
            st.warning(f"Large area ({area:,.0f} km²). Download and inference will be slow, and very "
                       "large boxes may only be partially covered by the available scenes.")

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
            return f"{s['date']}  ·  {cc:.0f}% cloud cover"

        scene_dates = [s["date"] for s in available]
        best_idx = min(range(len(available)), key=lambda i: available[i]["cloud_cover"])
        stored = st.session_state.get("aoi_scene_date")
        default_idx = scene_dates.index(stored) if stored in scene_dates else best_idx

        selected_scene = st.selectbox(
            "Available Sentinel-2 scenes",
            options=scene_dates,
            index=default_idx,
            format_func=lambda d: _scene_label(next(s for s in available if s["date"] == d)),
            help="Scenes from the window before your selected date, with cloud cover per scene.",
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
            st.caption(f"Area: **{area:,.0f} km²**")
            if st.button("Preview satellite scene", type="secondary", use_container_width=True):
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
            f"Scene acquired: **{preview['scene_date']}**  \n"
            f"Cloud cover: **{cc}%**  \n"
            f"Area: **{area:,.0f} km²**"
        )
        if isinstance(cc, (int, float)) and cc > 20:
            st.warning(f"Cloud cover is {cc}%. The image may be partially obscured.")

        # Multi-tile transparency: the AOI may straddle several Sentinel-2 MGRS tiles,
        # which are mosaicked together (not dropped). Warn — and flag date mismatches.
        tiles = preview.get("tiles", [])
        if len(tiles) > 1:
            uniq_dates = sorted(set(preview.get("tile_dates", {}).values()))
            msg = (f"Your area spans {len(tiles)} Sentinel-2 tiles ({', '.join(tiles)}), "
                   "mosaicked into one image before detection.")
            if len(uniq_dates) > 1:
                msg += (f"  \n\nTiles were acquired on different dates "
                        f"({', '.join(uniq_dates)}), so seams may show along tile boundaries.")
            st.warning(msg)

        # --- Stage 2: Detection ---
        detection = st.session_state.get("detection_result")

        if detection is None:
            from src.infer import estimate_runtime
            from src.utils import get_device
            est = estimate_runtime(bbox, get_device())
            st.caption(f"Estimated run time: **about {_fmt_seconds(est['seconds'])}**")
            if st.button("Run burn scar detection", type="primary", use_container_width=True):
                try:
                    res = _run_with_progress(*rounded, est)
                    st.session_state["detection_result"] = res
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection failed: {e}")
            if st.button("Change area or date", use_container_width=True):
                st.session_state.pop("scene_preview", None)
                st.session_state.pop("detection_result", None)
                st.rerun()
            return

        st.success(
            f"Burn scar detected over about {detection['burned_frac'] * 100:.1f}% of the area.  \n\n"
            f"HLS acquired: {detection['scene_date']}  \n"
            f"Granules used: {detection['n_scenes']}"
        )
        if st.button("Change area or date", use_container_width=True):
            st.session_state.pop("scene_preview", None)
            st.session_state.pop("detection_result", None)
            st.rerun()


def main():
    # Must run on every script rerun, so it lives here rather than at import time.
    # Python caches the imported module, so module-level st.* calls only execute on
    # the first run; later reruns would fall back to the default centered layout.
    st.set_page_config(page_title="Wildfire Burn Scar Detection", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    overlay_opacity = 0.6

    # ── Sidebar: app description + metrics (filled after fire selection) ──────
    with st.sidebar:
        st.markdown(
            "<h3>Wildfire Burn Scar Detection</h3>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Fine-tuned [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) "
            "(IBM × NASA ViT-Large, 300M params) on 92 wildfires — 37 US + 55 global across 6 biomes. "
            "Evaluated on 4 held-out California fires never seen during training."
        )
        # Dynamic content (custom-area controls, or fire metrics) goes in `top`; the
        # optional held-out fire picker sits below it in `bottom`.
        top = st.container()
        bottom = st.container()
        below = st.container()   # fire metrics and expanders, under the picker
    with bottom:
        st.divider()
        region_key = st.selectbox(
            "Test Fires",
            options=list(REGIONS.keys()),
            index=None,
            placeholder="Choose a fire",
            format_func=lambda k: REGIONS[k]["display_name"],
            key="test_fire",
        )
        st.caption("Select a held-out test fire to see how the model did. "
                   "Clear the selection to draw your own area.")
    metrics_placeholder = below.empty()

    if region_key is None:
        custom_detection_view(top)
        return

    region = REGIONS[region_key]
    pred_data = load_prediction(region["name"])

    # ── Fill sidebar metrics now that we have pred_data ───────────────────────
    with metrics_placeholder.container():
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

            rows = [
                ("Recall", recall, "Share of burned pixels the model found"),
                ("Precision", precision, "Share of flagged pixels that were burned"),
                ("IoU", iou, "Overlap of prediction and reference"),
            ]
            body = "".join(
                f'<div class="metric-row"><div class="metric-label">'
                f'<span class="metric-value">{val * 100:.0f}%</span>'
                f'<span class="metric-name">{name}</span></div>'
                f'<div class="metric-desc">{desc}</div></div>'
                for name, val, desc in rows
            )
            km2 = float(region["acres"].replace(",", "")) * 0.00404686
            facts_html = "".join(
                f'<div class="fact-row"><span class="fact-key">{k}</span>'
                f'<span class="fact-val">{v}</span></div>'
                for k, v in [
                    ("Location", region["location"]),
                    ("Dates", region["dates"]),
                    ("Area burned", f'{region["acres"]} acres ({km2:,.0f} km²)'),
                ]
            )
            st.markdown(
                f'<div class="fire-title">{region["display_name"]}</div>'
                f'<div class="fire-sub">Held-out fire, never seen during training</div>'
                f'<div class="fact-list">{facts_html}</div>'
                f'{body}',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="metric-foot">Model: {pred.sum() / total * 100:.1f}% burned<br>'
                f'dNBR reference: {true.sum() / total * 100:.1f}% burned</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(
                "Predictions not available.\n\n"
                f"```bash\npython run_inference.py --region {region['name']}\n```"
            )

    # ── Map ───────────────────────────────────────────────────────────────────
    with st.spinner("Loading satellite imagery..."):
        m = create_map(region, pred_data, overlay_opacity)
    st_folium(m, use_container_width=True, height=700, returned_objects=[])
    _fit_map_to_window(_focus_bounds(region, pred_data))

    with below:
        # --- Comparison plot (collapsed by default) ---
        with st.expander("Error map"):
            st.caption(
                "Where the model agrees and disagrees with the dNBR reference. "
                "Green: burned in both. Orange: burned but missed by the model. "
                "Red: flagged by the model but not burned in the reference."
            )
            if pred_data is not None:
                from src.visualize import plot_error_map
                import matplotlib.pyplot as plt
                fig = plot_error_map(
                    pred_data["image"],
                    pred_data["true_mask"],
                    pred_data["pred_mask"],
                    title=region["display_name"],
                )
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            else:
                st.info("Run inference to generate the error map.")

        # --- Methodology ---
        with st.expander("How it works"):
            st.markdown("""
            **Data.** Harmonized Landsat Sentinel-2 (HLS) imagery from
            [NASA Earthdata](https://www.earthdata.nasa.gov/): 6 bands (blue, green, red,
            near-infrared, and two shortwave-infrared) at 30 m resolution. 

            **Labels.** For each training fire, a pre-fire and a post-fire
            image were compared. A pixel is labeled burned when
            `dNBR = NBR_before − NBR_after > 0.10` (USGS low-severity threshold),
            where `NBR = (NIR − SWIR) / (NIR + SWIR)`. Burned vegetation reflects less
            near-infrared and more shortwave-infrared, so NBR drops after a fire. The pre-fire
            image is used only to make labels, and the post-fire image only is the model input.

            The model is trained on the resulting burned / not burned mask, not on the dNBR values
            themselves. It learns a yes/no answer for each pixel and does not predict burn severity.

            **Model.** The model has two parts, an encoder and a decoder:

            1. **Encoder.** [Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)
               is a ViT-Large pretrained by NASA and IBM on 4.2 million HLS samples. For each of the 4
               time slots it splits the 224×224 patch into 16×16-pixel tokens (a 14×14 grid) and
               passes them through 24 transformer layers.
               Its output is an *embedding* for each token: a list of 1024 numbers describing that
               part of the image. Prithvi expects a time series, so the single post-fire scene is
               repeated across its 4 time slots, and the 4 sets of embeddings are averaged back
               into one 14×14 grid.

            2. **Decoder.** A small FPN network (3.8M parameters) added on top of the model. It takes the
               embeddings from encoder and processes them to produce the final output. The output is a burn 
               probability for every pixel. 

            **Training (fine-tuning).** The model was trained on 92 wildfires: 37 in the US and 55
            global events across 6 biomes. Each training example is a post-fire patch and its burn
            mask from the dNBR labels. The model predicts a mask, the prediction is compared to the
            label, and the weights are updated. The decoder is trained first while the encoder is
            held fixed, then the encoder is also updated with a small learning rate so it keeps
            what it learned in pretraining.

            The loss is cross-entropy plus Tversky loss, which handles the fact that burned pixels
            are a minority in most scenes. The learning rate and loss weights were chosen with an
            Optuna search on training fires only.

            **Test fires.** Four California fires are held out entirely: Palisades, Eaton, Woolsey
            and Thomas. 

            **Running on a scene.** Large scenes are split into overlapping 224×224 windows and
            overlapping predictions are averaged. Open water is masked out beforehand using NDWI
            water indices to avoid mistaking water for burn.

            **Custom areas.** You can also draw a box anywhere on the map and run the model on a
            post-fire scene from a date you choose:

            1. **Pick a date.** The scene list and the preview image on the map come from
               Sentinel-2 imagery on Microsoft Planetary Computer. They are only for choosing a
               date with low cloud cover. It is a different image
               product from the one the model uses.
            2. **Download.** When you click run, the app downloads the HLS images for that area
               and date from NASA Earthdata, which is the same data type the model was trained on.
               If the box spans several satellite tiles, they are merged into one image.
            3. **Predict.** The merged image is run through the model with overlapping windows
               and the water mask, as described above.
            4. **Display.** The map is redrawn with the HLS image and the predicted burn scar in
               red. Both views use the same fixed brightness scale, so
               colors are close to the preview. They can differ slightly because the two come
               from different image products. The scene date is the same.

            Large areas take longer, and tiles from different dates may show seams. The model is
            trained to find burn scars, so results over areas with no recent fire, or with
            clouds, snow or haze, may not be reliable.
            """)


if __name__ == "__main__":
    main()
