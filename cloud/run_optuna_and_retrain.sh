#!/bin/bash
# Complete Optuna + retrain pipeline. Runs entirely on EC2 instance.
# Usage: Pass as EC2 user data or run directly after SSH.
# This script is independent of local machine—instance handles everything.

set -euo pipefail

LOG_FILE="/tmp/training.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "[$(date)] ========== TRAINING PIPELINE START ==========="

REPO_DIR="${REPO_DIR:-$HOME/burn-scar-detection}"
S3_BUCKET="${S3_BUCKET:-s3://burn-scar-detection}"
EARTHDATA_USER="${EARTHDATA_USER:-ekarthei}"
EARTHDATA_PASS="${EARTHDATA_PASS:-}"

# Activate PyTorch environment
source /opt/pytorch/bin/activate
echo "[$(date)] PyTorch environment activated"

# Setup Earthdata credentials
if [ -z "$EARTHDATA_PASS" ]; then
    echo "[$(date)] ERROR: EARTHDATA_PASS not set. Aborting."
    exit 1
fi

printf 'machine urs.earthdata.nasa.gov login %s password %s\n' "$EARTHDATA_USER" "$EARTHDATA_PASS" > ~/.netrc
chmod 600 ~/.netrc
echo "[$(date)] Earthdata credentials configured"

# Clone repo if not already there
if [ ! -d "$REPO_DIR" ]; then
    echo "[$(date)] Cloning repo..."
    git clone -b cloud-deploy https://github.com/evankart/burn-scar-detection.git "$REPO_DIR"
fi

cd "$REPO_DIR"
echo "[$(date)] Working directory: $(pwd)"

# Install dependencies
echo "[$(date)] Installing dependencies..."
pip install -q optuna

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export S3_BUCKET="$S3_BUCKET"

# ===== OPTUNA SEARCH =====
echo "[$(date)] ========== OPTUNA SEARCH START =========="
python -u scripts/optuna_search.py \
  --config configs/finetune_50fires_config.yaml \
  --n-trials 3 --epochs 4 --experiment-name optuna 2>&1 | while IFS= read -r line; do
    echo "[$(date)] $line"
    # Periodic S3 upload every 100 lines (~5 min of logs)
    if (( RANDOM % 100 == 0 )); then
        aws s3 sync checkpoints/optuna/ "$S3_BUCKET/optuna/" --region us-west-2 --quiet 2>/dev/null || true
    fi
done

echo "[$(date)] Optuna search complete, uploading final results to S3..."
aws s3 sync checkpoints/optuna/ "$S3_BUCKET/optuna/" --region us-west-2 || {
    echo "[$(date)] ERROR: Failed to upload Optuna results to S3!"
    exit 1
}
echo "[$(date)] ✓ Optuna results in S3"

# ===== FINAL RETRAIN =====
echo "[$(date)] ========== FINAL RETRAIN START =========="
EXP=finetune_v4_50fires CONFIG=configs/finetune_50fires_config.yaml bash cloud/run_job.sh 2>&1 | while IFS= read -r line; do
    echo "[$(date)] $line"
    # Periodic S3 upload every 150 lines (~5-10 min of logs)
    if (( RANDOM % 150 == 0 )); then
        aws s3 sync "checkpoints/$EXP/" "$S3_BUCKET/$EXP/" --region us-west-2 --quiet 2>/dev/null || true
    fi
done

echo "[$(date)] Final retrain complete, uploading results to S3..."
aws s3 sync "checkpoints/finetune_v4_50fires/" "$S3_BUCKET/finetune_v4_50fires/" --region us-west-2 || {
    echo "[$(date)] ERROR: Failed to upload retrain results to S3!"
    exit 1
}
echo "[$(date)] ✓ Final retrain results in S3"

# Verify results
echo "[$(date)] Verifying S3 uploads..."
aws s3 ls "$S3_BUCKET/optuna/best_params.yaml" --region us-west-2 && echo "[$(date)] ✓ Optuna results verified"
aws s3 ls "$S3_BUCKET/finetune_v4_50fires/best_model.pt" --region us-west-2 && echo "[$(date)] ✓ Final checkpoint verified"

echo "[$(date)] ========== TRAINING PIPELINE COMPLETE =========="
echo "[$(date)] Results:"
echo "[$(date)]   - Optuna: s3://burn-scar-detection/optuna/"
echo "[$(date)]   - Checkpoint: s3://burn-scar-detection/finetune_v4_50fires/best_model.pt"
echo "[$(date)] Logs: $LOG_FILE"

# Upload logs to S3
aws s3 cp "$LOG_FILE" "$S3_BUCKET/logs/training_$(date +%s).log" --region us-west-2 || true

# Self-terminate
echo "[$(date)] Self-terminating instance..."
sudo shutdown -h now
