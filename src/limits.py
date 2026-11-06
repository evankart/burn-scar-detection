"""Size limits for on-demand detection.

The hosted Hugging Face Space has ~16 GB of RAM and no GPU, so it gets tight limits.
Anywhere else (a local machine) gets larger ones. Each can be overridden with an
environment variable.
"""
import os

# Hugging Face sets SPACE_ID inside a running Space.
ON_SPACE = bool(os.environ.get("SPACE_ID"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Largest AOI the app accepts, in km2.
MAX_AOI_KM2 = _env_int("BURNSCAR_MAX_AOI_KM2", 10_000 if ON_SPACE else 40_000)
# Above this the app warns that download and inference will be slow.
SLOW_AOI_KM2 = _env_int("BURNSCAR_SLOW_AOI_KM2", 2_000 if ON_SPACE else 10_000)
# Max HLS granules mosaicked per run. An HLS tile is ~110 km across, so a 3x3 block
# (about 330 km square) needs 9.
MAX_SCENES = _env_int("BURNSCAR_MAX_SCENES", 3 if ON_SPACE else 9)
