"""
Download + verify a large, geographically diverse batch of wildfires (2018-2022,
HLSS30 coverage) to expand the training set for encoder fine-tuning. A fire is
kept only if it shows real burn (dNBR-burn >= 3%) AND correct reflectance scaling
(NIR median < 1). None are adjacent to the held-out test fires (Woolsey 34.16/
-118.83, Thomas 34.5/-119.1, East Troublesome 40.15/-105.85). Prints YAML-ready
blocks for the fires that pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import logging; logging.basicConfig(level=logging.WARNING)
import numpy as np, xarray as xr
from src.data import HLSDownloader, _restore_crs, generate_burn_mask

# name, lat, lon, buffer_km, pre_date, post_date
CANDIDATES = [
    # --- California Sierra / NorCal ---
    ("camp_fire_2018", 39.81, -121.44, 20, "2018-10-01", "2018-12-05"),
    ("tubbs_2017", 38.55, -122.62, 16, "2017-09-01", "2017-11-10"),
    ("kincade_2019", 38.79, -122.78, 18, "2019-09-15", "2019-11-20"),
    ("glass_2020", 38.56, -122.50, 16, "2020-08-20", "2020-10-25"),
    ("creek_2020_sierra", 37.19, -119.26, 30, "2020-08-20", "2020-11-01"),
    ("castle_sqf_2020", 36.20, -118.55, 28, "2020-08-10", "2020-11-05"),
    ("monument_2021", 40.60, -123.30, 30, "2021-07-15", "2021-10-20"),
    ("mosquito_2022", 39.00, -120.73, 22, "2022-08-25", "2022-10-25"),
    ("river_carmel_2020", 36.30, -121.55, 20, "2020-08-20", "2020-10-20"),
    ("zogg_2020", 40.46, -122.55, 14, "2020-09-01", "2020-11-01"),
    # --- Pacific Northwest ---
    ("holiday_farm_2020", 44.13, -122.45, 22, "2020-08-25", "2020-10-25"),
    ("beachie_creek_2020", 44.75, -122.18, 25, "2020-08-25", "2020-10-25"),
    ("cedar_creek_2022_or", 44.10, -121.92, 25, "2022-08-25", "2022-10-25"),
    ("cold_springs_2020_wa", 48.20, -118.90, 30, "2020-08-25", "2020-10-15"),
    ("schneider_2021_wa", 46.60, -121.00, 20, "2021-08-01", "2021-10-10"),
    # --- Rockies / Colorado ---
    ("cameron_peak_2020", 40.60, -105.72, 28, "2020-08-01", "2020-11-05"),
    ("pine_gulch_2020", 39.42, -108.40, 22, "2020-07-20", "2020-09-20"),
    ("calwood_2020", 40.13, -105.38, 12, "2020-10-01", "2020-11-15"),
    ("spring_creek_2018", 37.50, -105.02, 18, "2018-06-01", "2018-08-10"),
    # --- Southwest (AZ/NM) ---
    ("bighorn_2020_az", 32.43, -110.78, 18, "2020-05-20", "2020-07-25"),
    ("bush_2020_az", 33.80, -111.30, 25, "2020-05-20", "2020-07-25"),
    ("telegraph_2021_az", 33.10, -110.90, 25, "2021-05-20", "2021-07-25"),
    ("cerro_pelado_2022_nm", 35.72, -106.55, 14, "2022-04-25", "2022-07-01"),
    # --- More SoCal chaparral (test domain; none near Woolsey/Thomas) ---
    ("blue_ridge_2020", 33.88, -117.68, 12, "2020-10-01", "2020-11-20"),
    ("bond_2020", 33.75, -117.62, 12, "2020-11-25", "2021-01-15"),
    ("tenaja_2019", 33.46, -117.27, 12, "2019-08-20", "2019-10-20"),
    ("sand_2016", 34.37, -118.36, 14, "2016-07-22", "2016-09-20"),
    ("la_tuna_2017", 34.23, -118.30, 12, "2017-09-01", "2017-10-25"),
]

dl = HLSDownloader("configs/train_config.yaml")
kept = []
for name, lat, lon, bk, pre, post in CANDIDATES:
    r = {"name": name, "lat": lat, "lon": lon, "buffer_km": bk,
         "pre_fire_date": pre, "post_fire_date": post}
    try:
        p = dl.download_region(r)
        pre_ds = _restore_crs(xr.open_dataset(p["pre"], engine="h5netcdf"))
        post_ds = _restore_crs(xr.open_dataset(p["post"], engine="h5netcdf")).rio.reproject_match(pre_ds)
        m = generate_burn_mask(pre_ds, post_ds, dnbr_threshold=0.10)
        nir = post_ds["B8A"].values.astype("float32"); med = float(np.median(nir[np.isfinite(nir)]))
        burn = 100 * float(m.mean())
        ok = burn >= 3.0 and med < 1.0
        print(f"{name:22s} burn={burn:5.1f}%  NIRmed={med:.3f}  {'KEEP' if ok else 'DROP'}")
        if ok:
            kept.append((name, lat, lon, bk, pre, post))
    except Exception as e:
        print(f"{name:22s} FAILED {str(e)[:60]}")

print("\n=== KEPT (%d) — YAML ===" % len(kept))
for name, lat, lon, bk, pre, post in kept:
    print(f'    - name: "{name}"\n      lat: {lat}\n      lon: {lon}\n      buffer_km: {bk}\n'
          f'      pre_fire_date: "{pre}"\n      post_fire_date: "{post}"')
