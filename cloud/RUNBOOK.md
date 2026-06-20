# Cloud training runbook (AWS, us-west-2)

Run the **Prithvi-EO-2.0-300M** Optuna search + full retrain on a GPU instance
co-located with the NASA HLS archive (us-west-2 → direct-S3 reads, fast downloads).
The instance is disposable: it syncs the HLS cache from S3, runs Optuna (7 trials × 5 epochs
on a 14-fire fast subset), generates the best-HP config, trains the full model (100 fires),
uploads everything to S3, and self-terminates.

## One-time setup (already done)
- S3 bucket `burn-scar-detection` (us-west-2) — results land here.
- EC2 key pair `burn-scar-detection` (us-west-2) — `~/Downloads/burn-scar-detection.pem`.
- IAM instance profile `burn-scar-ec2-profile` with S3 read/write on the bucket.
- HLS cache at `s3://burn-scar-detection/hls-cache/` — synced to/from instance.

## Launch (fully automated)

```bash
export EARTHDATA_PASS='<your earthdata password>'   # single quotes: avoids ! expansion
bash scripts/launch_training.sh
```

This launches a `g5.xlarge` (A10G 24GB) via EC2 user data. The instance:
1. Clones `cloud-deploy` branch
2. Syncs HLS cache from S3 (fast — avoids re-downloading 100 fires)
3. Runs `optuna_search.py` (7 trials × 5 epochs, frozen encoder, 14-fire subset)
4. Uploads Optuna results to `s3://burn-scar-detection/optuna/`
5. Runs full retrain with best HPs (`finetune_optuna_config.yaml`, 100 fires, 16 epochs max)
6. Uploads checkpoint to `s3://burn-scar-detection/finetune_v3/`
7. Self-terminates

Monitor progress (SSM, no SSH key needed):
```bash
aws ssm start-session --target <INSTANCE_ID> --region us-west-2
# then: tail -f /tmp/training_full.log
```

Or SSH directly:
```bash
aws ec2-instance-connect ssh --instance-id <INSTANCE_ID> --region us-west-2 --os-user ubuntu
tail -f /tmp/training_full.log | grep -E "Epoch|val_iou|Trial|ERROR|COMPLETE"
```

## Get results back + evaluate locally

```bash
mkdir -p checkpoints/finetune_v3 checkpoints/optuna
aws s3 sync s3://burn-scar-detection/finetune_v3/ checkpoints/finetune_v3/ --region us-west-2 --exclude "epoch_*.pt"
aws s3 sync s3://burn-scar-detection/optuna/ checkpoints/optuna/ --region us-west-2

python scripts/eval_sweep.py --checkpoints checkpoints/finetune_v3/best_model.pt
```

## Cost / safety
- g5.xlarge on-demand ≈ $1.01/hr; full pipeline (Optuna + retrain) ≈ $3–4.
- The instance self-terminates — no idle charges. Budget alarm at $10 as a backstop.
- `SELF_TERMINATE=1` in the launch script; `shutdown -h now` at the end of user data.

---

# Deploy to Hugging Face Spaces

## Upload model checkpoint

```bash
# Upload new checkpoint to HF dataset repo
huggingface-cli upload evankart/burn-scar-detection-data \
  checkpoints/finetune_v3/best_model.pt \
  checkpoints/finetune_v3/best_model.pt \
  --repo-type dataset

# Upload precomputed predictions for held-out fires
for f in woolsey_fire_2018 thomas_fire_2017 palisades_fire_2025 eaton_fire_2025; do
  huggingface-cli upload evankart/burn-scar-detection-data \
    data/predictions/$f.npz predictions/$f.npz --repo-type dataset
done
```

## Push the app

```bash
python scripts/push_to_space.py
```

Uploads `app.py`, `src/**`, `configs/train_config.yaml`, `requirements.txt`,
`packages.txt`, and `cloud/space_README.md` → `README.md`.

## Account-side setup (already done)
- HF Write token: huggingface.co/settings/tokens
- Earthdata secrets on the Space: `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`

## Notes
- **Latency:** first live detection on CPU is ~2–4 min (download + model); cached after.
- **Memory:** Prithvi 2.0 fits the 16 GB CPU Space.
- **Cost:** free on the CPU tier.
