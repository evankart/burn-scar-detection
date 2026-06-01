#!/usr/bin/env bash
# Self-contained training job for a us-west-2 GPU instance (AWS Deep Learning AMI).
# Does: install deps -> download HLS (direct S3 in-region) -> fine-tune -> upload
# the trained checkpoint to S3 -> (optionally) self-terminate the instance.
#
# The lightweight eval (vs the baseline checkpoint, which is not in git) is run
# locally after pulling the result back, so the cloud box doesn't need it.
#
# Usage on the instance:
#   export EARTHDATA_USER=... EARTHDATA_PASS=...
#   export S3_BUCKET=s3://burn-scar-detection
#   bash cloud/run_job.sh                 # train + upload, leave instance up
#   SELF_TERMINATE=1 bash cloud/run_job.sh  # train + upload + terminate
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/burn-scar-detection}"
S3_BUCKET="${S3_BUCKET:-s3://burn-scar-detection}"
EXP="${EXP:-finetune_v2}"
CONFIG="${CONFIG:-configs/finetune_config.yaml}"

cd "$REPO_DIR"

# --- NASA Earthdata credentials -> ~/.netrc (needed by earthaccess) ---
if [ ! -f "$HOME/.netrc" ]; then
  if [ -z "${EARTHDATA_USER:-}" ] || [ -z "${EARTHDATA_PASS:-}" ]; then
    echo "ERROR: set EARTHDATA_USER and EARTHDATA_PASS env vars (or create ~/.netrc) first" >&2
    exit 1
  fi
  printf 'machine urs.earthdata.nasa.gov login %s password %s\n' \
    "$EARTHDATA_USER" "$EARTHDATA_PASS" > "$HOME/.netrc"
  chmod 600 "$HOME/.netrc"
fi

# --- deps (DL AMI already has torch+CUDA; install the geospatial stack) ---
python -m pip install -q -r requirements.txt

# --- train ---
export MPLBACKEND=Agg

# Step 1: download/cache all regions (with the config's band set) WITHOUT training.
echo "=== Downloading HLS for $CONFIG (download-only) ==="
python -u run_training.py --config "$CONFIG" --experiment-name "$EXP" --download-only

# Step 2: brightness diagnostic — prints per-band train-fire medians vs the
# Prithvi pretraining mean so we can see if 2.0 needs a brightness gain before
# committing GPU hours. Non-fatal: never blocks the run.
echo "=== Band-brightness diagnostic ==="
python -u scripts/band_stats_v2.py --config "$CONFIG" || echo "(diagnostic failed; continuing)"

# Step 3: train (caches are warm, so this skips re-download).
echo "=== Training $EXP with $CONFIG ==="
python -u run_training.py --config "$CONFIG" --experiment-name "$EXP"

# --- upload results to S3 (checkpoint survives instance termination) ---
echo "=== Uploading checkpoints/$EXP to $S3_BUCKET/$EXP/ ==="
aws s3 cp "checkpoints/$EXP/" "$S3_BUCKET/$EXP/" --recursive

echo "=== DONE. Pull locally with: aws s3 cp $S3_BUCKET/$EXP/ ./checkpoints/$EXP/ --recursive ==="

if [ "${SELF_TERMINATE:-0}" = "1" ]; then
  echo "=== Self-terminating instance ==="
  sudo shutdown -h now
fi
