#!/usr/bin/env bash
# Optuna hyperparameter search on a us-west-2 GPU instance (AWS Deep Learning AMI).
# Mirrors run_job.sh: install deps -> cache HLS -> run the search -> upload the
# study + tuned config to S3 -> (optionally) self-terminate.
#
# After this finishes, retrain with the tuned config (TODO item 4):
#   EXP=finetune_v3 CONFIG=configs/finetune_optuna_config.yaml \
#     SELF_TERMINATE=1 bash cloud/run_job.sh
#
# Usage on the instance:
#   export EARTHDATA_USER=... EARTHDATA_PASS=...
#   export S3_BUCKET=s3://burn-scar-detection
#   bash cloud/run_optuna.sh                  # search + upload, leave instance up
#   SELF_TERMINATE=1 bash cloud/run_optuna.sh # search + upload + terminate
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/burn-scar-detection}"
S3_BUCKET="${S3_BUCKET:-s3://burn-scar-detection}"
EXP="${EXP:-optuna}"
CONFIG="${CONFIG:-configs/finetune_config.yaml}"
N_TRIALS="${N_TRIALS:-10}"
EPOCHS="${EPOCHS:-8}"

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

# --- deps (DL AMI already has torch+CUDA; install the geospatial + optuna stack) ---
python -m pip install -q -r requirements.txt

export MPLBACKEND=Agg

# Step 1: warm the HLS cache (download-only) so trials don't re-download.
echo "=== Caching HLS for $CONFIG (download-only) ==="
python -u run_training.py --config "$CONFIG" --experiment-name "$EXP" --download-only

# Step 2: run the Optuna search (data is built once, then N_TRIALS fine-tunes).
echo "=== Optuna search: $N_TRIALS trials x $EPOCHS epochs ==="
python -u scripts/optuna_search.py \
  --config "$CONFIG" --n-trials "$N_TRIALS" --epochs "$EPOCHS" --experiment-name "$EXP"

# --- upload study + tuned config to S3 ---
echo "=== Uploading checkpoints/$EXP to $S3_BUCKET/$EXP/ ==="
aws s3 cp "checkpoints/$EXP/" "$S3_BUCKET/$EXP/" --recursive
if [ -f configs/finetune_optuna_config.yaml ]; then
  aws s3 cp configs/finetune_optuna_config.yaml "$S3_BUCKET/$EXP/finetune_optuna_config.yaml"
fi

echo "=== DONE. Pull locally with: aws s3 cp $S3_BUCKET/$EXP/ ./checkpoints/$EXP/ --recursive ==="
echo "=== Then retrain: EXP=finetune_v3 CONFIG=configs/finetune_optuna_config.yaml bash cloud/run_job.sh ==="

if [ "${SELF_TERMINATE:-0}" = "1" ]; then
  echo "=== Self-terminating instance ==="
  sudo shutdown -h now
fi
