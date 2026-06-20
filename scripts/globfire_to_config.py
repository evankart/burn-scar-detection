"""
Convert a GlobFire (GWIS) CSV export into `data.fires` entries for the training
config.

The CSV is the one produced by the Earth Engine export in TODO.md / RUNBOOK
(columns: id, lon, lat, area_ha, initial_date, final_date). For each row this
derives the config fields the HLS downloader needs:

  name           gwis_<tag>_<year>_<id>     (GlobFire events have no names)
  role           train                      (NEVER test — see the leakage guard)
  lat, lon       fire centroid
  buffer_km      from burned area (equivalent radius × 1.5, clamped 10–70 km)
  pre_fire_date  initial_date − 40 d        (clean pre-fire scene window start)
  post_fire_date final_date + 5 d           (post-fire scene window start)

Selection (--min-ha / --max / --tag) is meant to be run once per biome CSV so
the final set is biome-balanced. Entries are appended to the config as raw text
(before the `model:` block) so the hand-maintained YAML formatting + comments are
preserved; existing YAML is only *read* (never re-dumped).

Usage:
    # review entries from one biome export:
    python scripts/globfire_to_config.py --csv au.csv --tag au --max 12

    # append them into the training config:
    python scripts/globfire_to_config.py --csv au.csv --tag au --max 12 \
        --append-to configs/train_config.yaml
"""

import argparse
import csv
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml

# Skip any candidate whose centroid is within this distance of a test-fire
# center — prevents a global fire from leaking spatially into a held-out region.
TEST_LEAKAGE_KM = 60.0


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _buffer_km(area_ha: float) -> int:
    """Half-width of the AOI box: equivalent circular radius × 1.5, clamped.

    Matches the existing registry's scaling (e.g. ~39k ha → ~20 km, ~280k ha →
    ~45 km)."""
    area_m2 = area_ha * 1e4
    radius_km = math.sqrt(area_m2 / math.pi) / 1000.0
    return int(max(10, min(70, round(radius_km * 1.5))))


def _read_existing(config_path: Path):
    """Return (existing_names, test_centers) from the target config (read-only)."""
    if not config_path.exists():
        return set(), []
    cfg = yaml.safe_load(config_path.read_text())
    fires = (cfg.get("data") or {}).get("fires") or []
    names = {f["name"] for f in fires if "name" in f}
    test_centers = [(f["lat"], f["lon"]) for f in fires
                    if f.get("role") == "test" and "lat" in f and "lon" in f]
    return names, test_centers


def _entry_dict(row: dict, tag: str) -> dict | None:
    """One CSV row → a fire entry dict, or None if the row is unusable."""
    try:
        lat = float(row["lat"]); lon = float(row["lon"])
        area_ha = float(row["area_ha"])
        idate = datetime.strptime(row["initial_date"], "%Y-%m-%d")
        fdate = datetime.strptime(row["final_date"], "%Y-%m-%d")
    except (KeyError, ValueError):
        return None
    fid = str(row.get("id", "")).strip() or "x"
    pre = (idate - timedelta(days=40)).strftime("%Y-%m-%d")
    post = (fdate + timedelta(days=5)).strftime("%Y-%m-%d")
    return {
        "name": f"gwis_{tag}_{idate.year}_{fid}",
        "role": "train",
        "lat": round(lat, 4),
        "lon": round(lon, 4),
        "buffer_km": _buffer_km(area_ha),
        "pre_fire_date": pre,
        "post_fire_date": post,
        "_area_ha": area_ha,  # for selection/sorting only; stripped on render
    }


def _render(entry: dict) -> str:
    """Render one entry as YAML list-item text at the registry's indentation
    (`  - name:` at 2 spaces, fields at 4)."""
    keys = ["name", "role", "lat", "lon", "buffer_km", "pre_fire_date", "post_fire_date"]
    lines = [f"  - name: {entry['name']}"]
    for k in keys[1:]:
        v = entry[k]
        v = f"'{v}'" if k.endswith("_date") else v
        lines.append(f"    {k}: {v}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True,
                    help="GlobFire CSV export (repeatable for multiple biomes).")
    ap.add_argument("--tag", default="g", help="Short biome/region tag for names, e.g. au, ca, med.")
    ap.add_argument("--bbox", default=None,
                    help="Optional sub-region filter 'minlon,minlat,maxlon,maxlat' (keep only "
                         "centroids inside it). Lets one continent export be split by biome, "
                         "e.g. Australian eucalyptus = SE forest, not northern savanna.")
    ap.add_argument("--min-ha", type=float, default=10000.0, help="Drop fires smaller than this.")
    ap.add_argument("--max", type=int, default=None, help="Keep only the N largest after filtering.")
    ap.add_argument("--min-sep-km", type=float, default=0.0,
                    help="Drop a fire if its centroid is within this distance of an already-"
                         "selected (larger) fire — thins overlapping AOIs. 0 = off.")
    ap.add_argument("--append-to", type=Path, default=None,
                    help="Config to append entries into (before the model: block). "
                         "Omit to print to stdout for review.")
    args = ap.parse_args()

    existing_names, test_centers = _read_existing(args.append_to) if args.append_to else (set(), [])

    bbox = None
    if args.bbox:
        minlon, minlat, maxlon, maxlat = (float(x) for x in args.bbox.split(","))
        bbox = (minlon, minlat, maxlon, maxlat)

    entries, skipped_leak, skipped_bbox = [], 0, 0
    for csv_path in args.csv:
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                e = _entry_dict(row, args.tag)
                if e is None or e["_area_ha"] < args.min_ha:
                    continue
                if bbox and not (bbox[0] <= e["lon"] <= bbox[2] and bbox[1] <= e["lat"] <= bbox[3]):
                    skipped_bbox += 1
                    continue
                if any(_haversine_km(e["lat"], e["lon"], tlat, tlon) < TEST_LEAKAGE_KM
                       for tlat, tlon in test_centers):
                    skipped_leak += 1
                    continue
                entries.append(e)

    # Largest first, dedupe by name (and against the target config).
    entries.sort(key=lambda e: e["_area_ha"], reverse=True)
    seen, deduped = set(existing_names), []
    skipped_sep = 0
    for e in entries:
        if e["name"] in seen:
            continue
        if args.min_sep_km and any(
            _haversine_km(e["lat"], e["lon"], k["lat"], k["lon"]) < args.min_sep_km
            for k in deduped
        ):
            skipped_sep += 1
            continue
        seen.add(e["name"]); deduped.append(e)
    if args.max:
        deduped = deduped[: args.max]

    if not deduped:
        print("No fires to add (after filtering/dedup/leakage guard).", file=sys.stderr)
        if skipped_leak:
            print(f"  ({skipped_leak} skipped for proximity to a test fire)", file=sys.stderr)
        return

    text = "".join(_render(e) for e in deduped)

    if skipped_leak:
        print(f"# skipped {skipped_leak} fire(s) within {TEST_LEAKAGE_KM:g} km of a test fire",
              file=sys.stderr)

    if args.append_to is None:
        print(text, end="")
        print(f"# {len(deduped)} fire(s) — re-run with --append-to to insert", file=sys.stderr)
        return

    # Insert before the first top-level `model:` line (end of the data.fires list).
    lines = args.append_to.read_text().splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.startswith("model:"):
            lines[i:i] = [text]
            break
    else:
        raise SystemExit(f"Could not find a top-level 'model:' block in {args.append_to}")
    args.append_to.write_text("".join(lines))
    print(f"Appended {len(deduped)} fire(s) to {args.append_to} "
          f"(tag={args.tag}, ≥{args.min_ha:g} ha).")


if __name__ == "__main__":
    main()
