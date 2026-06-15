#!/bin/bash
# Simple launcher: starts AWS instance, kicks off training, returns immediately.
# Training runs fully on AWS—your machine can shutdown safely.
# Usage: bash scripts/launch_training_remote.sh

set -euo pipefail

EARTHDATA_USER="${EARTHDATA_USER:-ekarthei}"
EARTHDATA_PASS="${EARTHDATA_PASS:-}"

if [ -z "$EARTHDATA_PASS" ]; then
    echo "ERROR: Set EARTHDATA_PASS environment variable"
    exit 1
fi

echo "[$(date)] Launching g5.xlarge instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-05d96ec5b47d26b37 \
  --instance-type g5.xlarge \
  --key-name burn-scar-detection \
  --region us-west-2 \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=burn-scar-training}]' \
  --query 'Instances[0].InstanceId' --output text)

echo "[$(date)] Instance launched: $INSTANCE_ID"
echo "[$(date)] Waiting for public IP..."

IP=""
for i in {1..30}; do
    IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region us-west-2 \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null || echo "None")
    if [ "$IP" != "None" ] && [ -n "$IP" ]; then
        break
    fi
    sleep 2
done

if [ -z "$IP" ] || [ "$IP" = "None" ]; then
    echo "ERROR: Failed to get public IP"
    exit 1
fi

echo "[$(date)] Instance ready: $IP"

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

echo "[$(date)] Starting training on instance (logs in S3)..."

# SSH once to start training script, then detach
ssh -i ~/.ssh/id_rsa -o StrictHostKeyChecking=no ubuntu@"$IP" << ENDSSH &
nohup bash << 'TRAINING_SCRIPT' > /tmp/training.log 2>&1 &
source /opt/pytorch/bin/activate

git clone -b cloud-deploy https://github.com/evankart/burn-scar-detection.git ~/burn-scar-detection 2>/dev/null || true

export EARTHDATA_USER='$EARTHDATA_USER'
export EARTHDATA_PASS='$EARTHDATA_PASS'
export S3_BUCKET='s3://burn-scar-detection'

bash ~/burn-scar-detection/cloud/run_optuna_and_retrain.sh
TRAINING_SCRIPT
exit
ENDSSH

sleep 2

echo ""
echo "════════════════════════════════════════════"
echo "✓ Training pipeline started on AWS instance"
echo "════════════════════════════════════════════"
echo ""
echo "Instance: $INSTANCE_ID ($IP)"
echo "Duration: ~12-16 hours"
echo "Cost: ~\$12-18"
echo ""
echo "Monitor progress:"
echo "  aws s3 ls s3://burn-scar-detection/optuna/ --region us-west-2"
echo "  aws s3 ls s3://burn-scar-detection/finetune_v4_50fires/ --region us-west-2"
echo ""
echo "Results:"
echo "  - Best hyperparams: s3://burn-scar-detection/optuna/best_params.yaml"
echo "  - Final checkpoint: s3://burn-scar-detection/finetune_v4_50fires/best_model.pt"
echo "  - Training logs: s3://burn-scar-detection/logs/"
echo ""
echo "Instance will auto-terminate when training completes."
echo "You can safely close this terminal and shutdown your machine."
echo ""
