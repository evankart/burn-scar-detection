#!/bin/bash
# Eval-only launcher: re-download 4 test fires (fresh Fmask), run eval_sweep + run_inference,
# upload .npz predictions + updated caches to S3. ~30-45 min, ~$0.75.
#
# Usage:
#   export EARTHDATA_PASS='<your earthdata password>'
#   bash scripts/launch_eval.sh

set -euo pipefail

EARTHDATA_USER="${EARTHDATA_USER:-ekarthei}"
EARTHDATA_PASS="${EARTHDATA_PASS:-}"

if [ -z "$EARTHDATA_PASS" ]; then
    echo "ERROR: Set EARTHDATA_PASS environment variable"
    exit 1
fi

echo "[$(date)] Launching eval instance..."

USER_DATA=$(cat <<'USERDATA'
#!/bin/bash
set -ex
exec 1> >(tee -a /tmp/eval.log)
exec 2>&1

echo "[$(date)] ========== EVAL INSTANCE STARTUP =========="
apt-get update -qq
apt-get install -y -qq git curl wget > /dev/null

git clone -b cloud-deploy https://github.com/evankart/burn-scar-detection.git /home/ubuntu/burn-scar-detection
chown -R ubuntu:ubuntu /home/ubuntu/burn-scar-detection
mkdir -p /home/ubuntu/burn-scar-detection/checkpoints

source /opt/pytorch/bin/activate
cd /home/ubuntu/burn-scar-detection
pip install -q -r requirements.txt 2>&1 | grep -E "ERROR|Successfully" || true
pip install -q earthaccess 2>&1 | tail -1 || true

export EARTHDATA_USERNAME=__ED_USER__
export EARTHDATA_PASSWORD=__ED_PASS__
export EARTHDATA_USER=__ED_USER__
export EARTHDATA_PASS=__ED_PASS__
export PYTHONPATH=/home/ubuntu/burn-scar-detection:${PYTHONPATH:-}

echo "[$(date)] ========== RESTORING ALL FIRE CACHES FROM S3 =========="
mkdir -p data/cache data/predictions
# Sync all 100 fire caches from S3 (training fires already cached, no re-download needed).
aws s3 sync s3://burn-scar-detection/hls-cache/ data/cache/ --region us-west-2 || echo "WARNING: cache sync partial"

echo "[$(date)] ========== DOWNLOADING BEST CHECKPOINT =========="
mkdir -p checkpoints/finetune_v3
aws s3 sync s3://burn-scar-detection/finetune_v3/ checkpoints/finetune_v3/ \
  --region us-west-2 --exclude "epoch_*.pt"

echo "[$(date)] ========== RE-DOWNLOADING TEST FIRES (fresh Fmask) =========="
# Delete stale test-fire caches so they're re-downloaded with current FMASK_BAD_BITS.
for fire in woolsey_fire_2018 thomas_fire_2017 palisades_fire_2025 eaton_fire_2025; do
    rm -f data/cache/${fire}_pre.nc data/cache/${fire}_post.nc
done

# Download only the 4 test fires via a minimal inline config subset.
python -u - <<'PYEOF'
import sys, os, yaml
sys.path.insert(0, '.')
import earthaccess
earthaccess.login(strategy='environment')

from src.data import HLSDownloader, load_config
cfg = load_config('configs/train_config.yaml')
dl = HLSDownloader('configs/train_config.yaml')

test_names = {'woolsey_fire_2018', 'thomas_fire_2017', 'palisades_fire_2025', 'eaton_fire_2025'}
all_regions = cfg['data'].get('test_regions', []) + cfg['data'].get('train_regions', [])
test_regions = [r for r in all_regions if r['name'] in test_names]
if len(test_regions) != len(test_names):
    found = {r['name'] for r in test_regions}
    missing = test_names - found
    print(f"WARNING: Could not find configs for: {missing}", flush=True)

for region in test_regions:
    print(f"Downloading {region['name']}...", flush=True)
    try:
        dl.download_region(region)
        print(f"  Done: {region['name']}", flush=True)
    except Exception as e:
        print(f"  ERROR {region['name']}: {e}", flush=True)
        sys.exit(1)
PYEOF

echo "[$(date)] ========== RUNNING EVAL SWEEP =========="
python -u scripts/eval_sweep.py --checkpoints checkpoints/finetune_v3/best_model.pt \
  2>&1 | tee /tmp/eval_results.txt

echo "[$(date)] ========== RUNNING INFERENCE (generating .npz predictions) =========="
for fire in woolsey_fire_2018 thomas_fire_2017 palisades_fire_2025 eaton_fire_2025; do
    echo "[$(date)] Inference: $fire"
    python -u run_inference.py --region $fire \
      --checkpoint checkpoints/finetune_v3/best_model.pt || {
        echo "ERROR: inference failed for $fire"; exit 1;
    }
done

echo "[$(date)] ========== UPLOADING RESULTS TO S3 =========="
# Updated test-fire caches (fresh Fmask)
for fire in woolsey_fire_2018 thomas_fire_2017 palisades_fire_2025 eaton_fire_2025; do
    aws s3 cp data/cache/${fire}_pre.nc  s3://burn-scar-detection/hls-cache/${fire}_pre.nc  --region us-west-2
    aws s3 cp data/cache/${fire}_post.nc s3://burn-scar-detection/hls-cache/${fire}_post.nc --region us-west-2
    aws s3 cp data/predictions/${fire}.npz s3://burn-scar-detection/predictions/${fire}.npz --region us-west-2
done

# Eval log
aws s3 cp /tmp/eval_results.txt s3://burn-scar-detection/logs/eval_$(date +%s).txt --region us-west-2 || true
aws s3 cp /tmp/eval.log s3://burn-scar-detection/logs/eval_full_$(date +%s).log --region us-west-2 || true

echo "[$(date)] ========== EVAL COMPLETE =========="
cat /tmp/eval_results.txt

echo "[$(date)] Auto-terminating..."
sleep 10
sudo shutdown -h now
USERDATA
)

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-05d96ec5b47d26b37 \
  --instance-type g5.xlarge \
  --key-name burn-scar-detection \
  --region us-west-2 \
  --instance-initiated-shutdown-behavior terminate \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3,DeleteOnTermination=true}' \
  --iam-instance-profile Name=burn-scar-ec2-profile \
  --user-data "$(echo "$USER_DATA" | sed "s/__ED_USER__/$EARTHDATA_USER/g" | sed "s|__ED_PASS__|$EARTHDATA_PASS|g")" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=burn-scar-eval}]' \
  --query 'Instances[0].InstanceId' --output text)

echo ""
echo "════════════════════════════════════════════"
echo "Eval instance launched: $INSTANCE_ID"
echo "Duration: ~30-45 min | Cost: ~\$0.75"
echo "════════════════════════════════════════════"
echo ""
echo "Monitor:"
echo "  aws ec2-instance-connect ssh --instance-id $INSTANCE_ID --region us-west-2 --os-user ubuntu"
echo "  tail -f /tmp/eval.log"
echo ""
echo "Results land at:"
echo "  s3://burn-scar-detection/logs/        (eval table + full log)"
echo "  s3://burn-scar-detection/predictions/ (.npz files for HF deploy)"
echo "  s3://burn-scar-detection/hls-cache/   (refreshed test-fire .nc caches)"
