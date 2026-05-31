# Cloud fine-tune runbook (AWS, us-west-2)

Run the heavy encoder fine-tune on a GPU instance co-located with the NASA HLS
archive (us-west-2 → direct-S3 reads, fast in-region downloads). The instance is
disposable: it trains, uploads the checkpoint to S3, and self-terminates.

## One-time setup (already done)
- S3 bucket `burn-scar-detection` (us-west-2) — results land here.
- EC2 key pair `burn-scar-detection` (us-west-2) — `~/Downloads/burn-scar-detection.pem`.
- GPU quota: "Running On-Demand G instances" in us-west-2 ≥ 4 (request pending until approved).
- Code pushed to GitHub `main` (the instance clones it).

## Launch (once the GPU quota is approved)
A `g5.xlarge` (1× A10G, 24 GB) on the Deep Learning OSS PyTorch AMI. Either let
the assistant launch it via the AWS CLI, or in the console:
EC2 → Launch → AMI "Deep Learning OSS Nvidia Driver PyTorch (Ubuntu)" →
type `g5.xlarge` → region us-west-2 → key pair `burn-scar-detection` →
**Advanced → Shutdown behavior = Terminate** → Launch.

## Run the job
```bash
chmod 400 ~/Downloads/burn-scar-detection.pem            # once, locally
ssh -i ~/Downloads/burn-scar-detection.pem ubuntu@<INSTANCE_PUBLIC_IP>

# on the instance:
git clone https://github.com/evankart/burn-scar-detection.git
cd burn-scar-detection
export EARTHDATA_USER='<your earthdata login>'
export EARTHDATA_PASS='<your earthdata password>'
export S3_BUCKET=s3://burn-scar-detection
SELF_TERMINATE=1 bash cloud/run_job.sh
```
`SELF_TERMINATE=1` powers the box off when done; with "Shutdown behavior =
Terminate" that deletes it and stops billing. As a safety net you can also run
`sudo shutdown -h +300 &` first (hard 5-hour cap).

## Get results back + evaluate locally
```bash
# on your Mac, after the job finishes:
aws s3 cp s3://burn-scar-detection/finetune_big/ ./checkpoints/finetune_big/ --recursive
python scripts/eval_sweep.py --threshold 0.5 \
  --checkpoints checkpoints/balanced_chaparral/best_model.pt checkpoints/finetune_big/best_model.pt
```
(The eval runs locally because the baseline checkpoint isn't in git.)

## Cost / safety
- g5.xlarge on-demand ≈ $1.01/hr; a full fine-tune ≈ $1–2, covered by Free-tier credits.
- The instance self-terminates → no idle charges. Budget alarm set at $10 as a backstop.
- Spot (~$0.30/hr) would be cheaper but needs a separate quota + interruption handling; not used here.
