#!/bin/bash
# Safe Optuna + retrain launcher with monitoring and cost controls.
# Usage: bash scripts/launch_optuna_safe.sh
# Runs in background (survives terminal closure via nohup).
# Cost: ~$12-18 (g5.xlarge at $1.01/hr × 12-18 hours)

set -euo pipefail

# Run in background so terminal closure doesn't kill training
if [ "${NOHUP_LAUNCHED:-0}" != "1" ]; then
    export NOHUP_LAUNCHED=1
    exec nohup "$0" "$@" > training_$(date +%s).log 2>&1 &
    echo "Training started in background. Check logs with: tail -f training_*.log"
    exit 0
fi

INSTANCE_ID=""
INSTANCE_IP=""
TIMEOUT_TOTAL=72000  # 20 hour hard cap (catches true hangs, won't interrupt normal training)

trap cleanup EXIT

cleanup() {
    if [ -n "$INSTANCE_ID" ]; then
        echo "[$(date)] Terminating instance $INSTANCE_ID..."
        aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region us-west-2 2>/dev/null || true
    fi
}

# Launch instance
echo "[$(date)] Launching g5.xlarge instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-05d96ec5b47d26b37 \
  --instance-type g5.xlarge \
  --key-name burn-scar-detection \
  --region us-west-2 \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=burn-scar-optuna-safe}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "[$(date)] Instance launched: $INSTANCE_ID"

# Wait for public IP
echo "[$(date)] Waiting for public IP..."
for i in {1..30}; do
    INSTANCE_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region us-west-2 \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
    if [ "$INSTANCE_IP" != "None" ] && [ -n "$INSTANCE_IP" ]; then
        break
    fi
    sleep 2
done

if [ -z "$INSTANCE_IP" ] || [ "$INSTANCE_IP" = "None" ]; then
    echo "[$(date)] ERROR: Failed to get public IP after 60 seconds. Aborting."
    exit 1
fi

echo "[$(date)] Instance ready at $INSTANCE_IP"

# Authorize SSH key
AZ=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region us-west-2 \
    --query 'Reservations[0].Instances[0].Placement.AvailabilityZone' --output text)
aws ec2-instance-connect send-ssh-public-key \
  --instance-id "$INSTANCE_ID" \
  --instance-os-user ubuntu \
  --ssh-public-key "$(cat ~/.ssh/id_rsa.pub)" \
  --availability-zone "$AZ" \
  --region us-west-2 2>/dev/null || true

sleep 3

# Run Optuna with explicit heartbeat logging
echo "[$(date)] Starting Optuna search (ETA: 6-8 hours)..."
timeout $TIMEOUT_TOTAL ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no ubuntu@"$INSTANCE_IP" << 'ENDSSH' || {
    echo "[$(date)] ERROR: Optuna failed or hit 20-hour timeout!"
    exit 1
}
source /opt/pytorch/bin/activate
printf 'machine urs.earthdata.nasa.gov login ekarthei password "a!bGE28i@uH#hbi"\n' > ~/.netrc
chmod 600 ~/.netrc
git clone -b cloud-deploy https://github.com/evankart/burn-scar-detection.git 2>&1 | tail -3
cd burn-scar-detection
pip install -q optuna
export PYTHONPATH=/home/ubuntu/burn-scar-detection:${PYTHONPATH:-}
export S3_BUCKET=s3://burn-scar-detection
echo "[HEARTBEAT] $(date): Starting Optuna"
python -u scripts/optuna_search.py --config configs/finetune_50fires_config.yaml --n-trials 3 --epochs 4 --experiment-name optuna 2>&1 | tee /tmp/optuna.log &
OPTUNA_PID=$!
# Monitor and upload every 20 minutes in case of interruption
while kill -0 $OPTUNA_PID 2>/dev/null; do
    sleep 1200  # 20 minutes
    echo "[HEARTBEAT] $(date): Uploading Optuna checkpoints to S3..."
    aws s3 sync checkpoints/optuna/ s3://burn-scar-detection/optuna/ --region us-west-2 --quiet || true
done
wait $OPTUNA_PID
echo "[HEARTBEAT] $(date): Optuna complete, final upload to S3"
aws s3 sync checkpoints/optuna/ s3://burn-scar-detection/optuna/ --region us-west-2 || echo "WARNING: S3 upload failed"
ENDSSH

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Optuna search failed or timed out"
    exit 1
fi

echo "[$(date)] ✓ Optuna complete, results in S3"

# Run final retrain
echo "[$(date)] Starting final retrain (ETA: 6-8 hours)..."
timeout $TIMEOUT_TOTAL ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no ubuntu@"$INSTANCE_IP" << 'ENDSSH' || {
    echo "[$(date)] ERROR: Final retrain failed or hit 20-hour timeout!"
    exit 1
}
source /opt/pytorch/bin/activate
cd burn-scar-detection
export PYTHONPATH=/home/ubuntu/burn-scar-detection:${PYTHONPATH:-}
export S3_BUCKET=s3://burn-scar-detection
echo "[HEARTBEAT] $(date): Starting final retrain"
EXP=finetune_v4_50fires CONFIG=configs/finetune_50fires_config.yaml bash cloud/run_job.sh 2>&1 | tee /tmp/retrain.log &
RETRAIN_PID=$!
# Monitor and upload every 30 minutes in case of interruption
while kill -0 $RETRAIN_PID 2>/dev/null; do
    sleep 1800  # 30 minutes
    echo "[HEARTBEAT] $(date): Uploading retrain checkpoints to S3..."
    aws s3 sync checkpoints/finetune_v4_50fires/ s3://burn-scar-detection/finetune_v4_50fires/ --region us-west-2 --quiet || true
done
wait $RETRAIN_PID
echo "[HEARTBEAT] $(date): Final retrain complete, final upload to S3"
aws s3 sync checkpoints/finetune_v4_50fires/ s3://burn-scar-detection/finetune_v4_50fires/ --region us-west-2 || true
ENDSSH

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Final retrain failed or timed out"
    exit 1
fi

echo "[$(date)] ✓ Training complete, results in S3 (finetune_v4_50fires/)"
echo "[$(date)] Instance will auto-terminate. Pulling results..."

# Verify results in S3
sleep 5
aws s3 ls s3://burn-scar-detection/finetune_v4_50fires/ --region us-west-2 && echo "[$(date)] ✓ Results confirmed in S3"

echo "[$(date)] SUCCESS: Training pipeline complete in $(( SECONDS / 60 )) minutes"
