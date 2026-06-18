#!/bin/bash
# Production training launcher: Optuna search (10 trials x 8 epochs) then
# full retrain with best hyperparams. Runs entirely via EC2 user data — no
# SSH or local terminal required after launch.
#
# Usage:
#   export EARTHDATA_USER=ekarthei EARTHDATA_PASS=...
#   bash scripts/launch_training_bulletproof.sh
#
# Cost: ~$15-18 (g5.xlarge ~15-18 hrs). Results land in s3://burn-scar-detection/
# For a quick smoke test first: bash scripts/launch_cloud_test.sh (~$1-2)

set -euo pipefail

EARTHDATA_USER="${EARTHDATA_USER:-ekarthei}"
EARTHDATA_PASS="${EARTHDATA_PASS:-}"

if [ -z "$EARTHDATA_PASS" ]; then
    echo "ERROR: Set EARTHDATA_PASS environment variable"
    exit 1
fi

echo "[$(date)] Launching instance with embedded training script..."

# Create user data script (runs on instance startup)
USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -ex
exec 1> >(tee -a /tmp/training_full.log)
exec 2>&1

echo "[$(date)] ========== INSTANCE STARTUP =========="
apt-get update -qq
apt-get install -y -qq git curl wget > /dev/null

echo "[$(date)] Cloning repo..."
git clone -b cloud-deploy https://github.com/evankart/burn-scar-detection.git /home/ubuntu/burn-scar-detection 2>&1 | grep -E "Cloning|fatal|ERROR" || true

echo "[$(date)] Activating PyTorch environment..."
source /opt/pytorch/bin/activate

echo "[$(date)] Installing dependencies..."
cd /home/ubuntu/burn-scar-detection
pip install -q -r requirements.txt 2>&1 | grep -E "ERROR|Successfully" || true
pip install -q earthaccess optuna 2>&1 | tail -1 || true

echo "[$(date)] Configuring Earthdata (credentials embedded)..."
cat > /root/.netrc << 'NETRC'
machine urs.earthdata.nasa.gov login EARTHDATA_USER password EARTHDATA_PASS
NETRC
sed -i "s/EARTHDATA_USER/$EARTHDATA_USER/g" /root/.netrc
sed -i "s|EARTHDATA_PASS|$EARTHDATA_PASS|g" /root/.netrc
chmod 600 /root/.netrc
export PYTHONPATH=/home/ubuntu/burn-scar-detection:${PYTHONPATH:-}
export S3_BUCKET='s3://burn-scar-detection'

echo "[$(date)] ========== OPTUNA SEARCH START =========="
python -u scripts/optuna_search.py \
  --config configs/finetune_config.yaml \
  --n-trials 10 --epochs 8 --experiment-name optuna || {
    echo "[$(date)] ERROR: Optuna failed";
    exit 1;
}

echo "[$(date)] Uploading Optuna results..."
aws s3 sync checkpoints/optuna/ s3://burn-scar-detection/optuna/ --region us-west-2 || {
    echo "[$(date)] WARNING: S3 upload failed";
}

echo "[$(date)] ========== FINAL RETRAIN START =========="
EXP=finetune_v3 CONFIG=configs/finetune_optuna_config.yaml bash cloud/run_job.sh || {
    echo "[$(date)] ERROR: Retrain failed";
    exit 1;
}

echo "[$(date)] ========== TRAINING COMPLETE =========="
echo "[$(date)] Uploading final logs to S3..."
aws s3 cp /tmp/training_full.log s3://burn-scar-detection/logs/training_$(date +%s).log --region us-west-2 || true

echo "[$(date)] Auto-terminating instance..."
sleep 10
sudo shutdown -h now
USERDATA
)

# Base64 encode user data (AWS requirement)
USER_DATA_B64=$(echo "$USER_DATA" | sed "s/EARTHDATA_USER/$EARTHDATA_USER/g" | sed "s/EARTHDATA_PASS/$EARTHDATA_PASS/g" | base64 | tr -d '\n')

echo "[$(date)] Launching g5.xlarge with user data..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-05d96ec5b47d26b37 \
  --instance-type g5.xlarge \
  --key-name burn-scar-detection \
  --region us-west-2 \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data "$(echo "$USER_DATA" | sed "s/EARTHDATA_USER/$EARTHDATA_USER/g" | sed "s/EARTHDATA_PASS/$EARTHDATA_PASS/g")" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=burn-scar-training-final}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "[$(date)] Instance launched: $INSTANCE_ID"
echo ""
echo "════════════════════════════════════════════"
echo "✓ Training pipeline RUNNING ON AWS"
echo "════════════════════════════════════════════"
echo ""
echo "Instance: $INSTANCE_ID"
echo "Duration: ~12-16 hours"
echo "Cost: ~\$12-18"
echo ""
echo "Monitor progress:"
echo "  aws s3 ls s3://burn-scar-detection/optuna/ --region us-west-2"
echo "  aws s3 ls s3://burn-scar-detection/finetune_v3/ --region us-west-2"
echo "  aws s3 cp s3://burn-scar-detection/logs/ . --recursive --region us-west-2"
echo ""
echo "Logs will be uploaded to:"
echo "  s3://burn-scar-detection/logs/"
echo ""
echo "Instance will auto-terminate when complete."
echo "════════════════════════════════════════════"
