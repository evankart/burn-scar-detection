#!/bin/bash
# Minimal end-to-end cloud test: 3 fires, 2 Optuna trials x 2 epochs,
# then 1 training run (2 epochs). Verifies the cloud pipeline works before
# committing to the full 92-fire run (~$15).
# Cost: ~$1-2 (~45-60 min on g5.xlarge).
#
# Usage:
#   export EARTHDATA_USER=ekarthei
#   export EARTHDATA_PASS=...
#   bash scripts/launch_cloud_test.sh

set -euo pipefail

EARTHDATA_USER="${EARTHDATA_USER:-ekarthei}"
EARTHDATA_PASS="${EARTHDATA_PASS:-}"
REGION="us-west-2"
AMI="ami-05d96ec5b47d26b37"

if [ -z "$EARTHDATA_PASS" ]; then
    echo "ERROR: Set EARTHDATA_PASS environment variable" >&2
    exit 1
fi

USER_DATA=$(cat <<EOF
#!/bin/bash
set -ex
exec 1> >(tee -a /tmp/cloud_test.log)
exec 2>&1

echo "[$(date)] ===== CLOUD TEST START ====="

source /opt/pytorch/bin/activate
apt-get update -qq && apt-get install -y -qq git > /dev/null

git clone -b cloud-deploy https://github.com/evankart/burn-scar-detection.git /home/ubuntu/burn-scar-detection
cd /home/ubuntu/burn-scar-detection

echo "[$(date)] Installing deps..."
pip install -q -r requirements.txt
pip install -q optuna

echo "[$(date)] Configuring Earthdata..."
printf 'machine urs.earthdata.nasa.gov login ${EARTHDATA_USER} password ${EARTHDATA_PASS}\n' > /root/.netrc
chmod 600 /root/.netrc

export PYTHONPATH=/home/ubuntu/burn-scar-detection:\${PYTHONPATH:-}
export S3_BUCKET='s3://burn-scar-detection'
export MPLBACKEND=Agg

echo "[$(date)] ===== OPTUNA TEST (2 trials x 2 epochs) ====="
python -u scripts/optuna_search.py \
    --config configs/finetune_test_optuna.yaml \
    --n-trials 2 --epochs 2 \
    --experiment-name cloud_test_optuna

echo "[$(date)] ===== TRAINING TEST (2 epochs) ====="
python -u run_training.py \
    --config configs/finetune_test_optuna.yaml \
    --experiment-name cloud_test_train

echo "[$(date)] ===== UPLOADING RESULTS ====="
aws s3 sync checkpoints/cloud_test_optuna/ s3://burn-scar-detection/cloud_test/optuna/ --region ${REGION} || echo "WARNING: S3 upload failed"
aws s3 sync checkpoints/cloud_test_train/ s3://burn-scar-detection/cloud_test/train/ --region ${REGION} || echo "WARNING: S3 upload failed"
aws s3 cp /tmp/cloud_test.log s3://burn-scar-detection/cloud_test/cloud_test.log --region ${REGION} || true

echo "[$(date)] ===== CLOUD TEST COMPLETE ====="
sleep 10
sudo shutdown -h now
EOF
)

echo "[$(date)] Launching g5.xlarge cloud test..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI" \
    --instance-type g5.xlarge \
    --key-name burn-scar-detection \
    --region "$REGION" \
    --instance-initiated-shutdown-behavior terminate \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=80,VolumeType=gp3,DeleteOnTermination=true}' \
    --iam-instance-profile Name=burn-scar-ec2-profile \
    --user-data "$USER_DATA" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=burn-scar-cloud-test}]' \
    --query 'Instances[0].InstanceId' --output text)

echo ""
echo "Instance launched: $INSTANCE_ID"
echo "Expected duration: ~45-60 min | Cost: ~\$1-2"
echo ""
echo "Monitor logs:"
echo "  aws logs: watch 'aws s3 ls s3://burn-scar-detection/cloud_test/ --region $REGION'"
echo "  live log: aws s3 cp s3://burn-scar-detection/cloud_test/cloud_test.log - --region $REGION"
echo ""
echo "Check completion:"
echo "  aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0].State.Name'"
