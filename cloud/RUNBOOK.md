# Cloud fine-tune runbook (AWS, us-west-2)

Run the heavy **Prithvi-EO-2.0-300M** encoder fine-tune (`configs/finetune_config.yaml`,
experiment `finetune_v2`) on a GPU instance co-located with the NASA HLS archive
(us-west-2 → direct-S3 reads, fast in-region downloads). The instance is
disposable: it downloads HLS, runs the brightness diagnostic, trains, uploads the
checkpoint to S3, and self-terminates. Prithvi 2.0 uses the same physical HLS
bands as 1.0, so existing caches are reused — `run_job.sh` only downloads fires
not already cached.

## One-time setup (already done)
- S3 bucket `burn-scar-detection` (us-west-2) — results land here.
- EC2 key pair `burn-scar-detection` (us-west-2) — `~/Downloads/burn-scar-detection.pem`.
- GPU quota: "Running On-Demand G instances" in us-west-2 ≥ 4 (request pending until approved).
- Code pushed to GitHub branch `cloud-deploy` (the instance clones this branch;
  `main` still holds the original code until this is merged).

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
git clone -b cloud-deploy https://github.com/evankart/burn-scar-detection.git
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
aws s3 cp s3://burn-scar-detection/finetune_v2/ ./checkpoints/finetune_v2/ --recursive
python scripts/eval_sweep.py --threshold 0.5 \
  --checkpoints checkpoints/balanced_chaparral/best_model.pt checkpoints/finetune_v2/best_model.pt
```
(The eval runs locally because the baseline checkpoint isn't in git.)

## Hyperparameter search (Optuna) — run before the final retrain
Same instance/AMI as the fine-tune. Searches LR, backbone-LR multiplier, Tversky
alpha, and burn-class weight over ~10 trials, maximizing val burn-class IoU on
the held-out *training* fires (carr + holy); test fires are never touched.
```bash
# on the instance, after cloning:
export EARTHDATA_USER=... EARTHDATA_PASS=... S3_BUCKET=s3://burn-scar-detection
SELF_TERMINATE=1 bash cloud/run_optuna.sh        # N_TRIALS=10 EPOCHS=8 by default
```
This uploads `checkpoints/optuna/` (study.pkl, best_params.yaml, plots) and
`configs/finetune_optuna_config.yaml` to S3. Then run the **final retrain** with the
tuned config:
```bash
EXP=finetune_v3 CONFIG=configs/finetune_optuna_config.yaml \
  SELF_TERMINATE=1 bash cloud/run_job.sh
```
Pull `checkpoints/finetune_v3/` back, run `run_inference.py` for the four test
fires, and update the README results table.

## Cost / safety
- g5.xlarge on-demand ≈ $1.01/hr; a full fine-tune ≈ $1–2, covered by Free-tier credits.
- The instance self-terminates → no idle charges. Budget alarm set at $10 as a backstop.
- Spot (~$0.30/hr) would be cheaper but needs a separate quota + interruption handling; not used here.

---

# Deploy to Hugging Face Spaces (Streamlit, free CPU)

Hosts both app tabs: precomputed held-out results, and live custom-area detection
(Prithvi runs on CPU — slow but works in the 16 GB Space).

## Account-side setup
1. **Create a Write token:** huggingface.co/settings/tokens → New token → type **Write**.
2. **Log in locally:** `huggingface-cli login` (paste the Write token).
3. **Set Earthdata secrets on the Space** (Settings → Variables and secrets → New secret):
   `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` (power `earthaccess` env-var auth).

## Upload model + predictions (the app fetches these at runtime)
```bash
huggingface-cli upload evankart/burn-scar-detection-data \
  checkpoints/balanced_chaparral/best_model.pt \
  checkpoints/balanced_chaparral/best_model.pt --repo-type dataset
for f in woolsey_fire_2018 thomas_fire_2017 palisades_fire_2025 eaton_fire_2025; do
  huggingface-cli upload evankart/burn-scar-detection-data \
    data/predictions/$f.npz predictions/$f.npz --repo-type dataset
done
```

## Push the app
`python scripts/push_to_space.py` uploads `app.py`, `src/**`, `configs/train_config.yaml`,
`requirements.txt`, `packages.txt`, and `cloud/space_README.md` → `README.md`. The
`app_file: app.py` front-matter makes Spaces run the top-level entrypoint, so
`from src.X import ...` resolves without sys.path hacks.

## Notes
- **Latency:** first live detection on CPU is ~2–4 min (download + model); cached after.
- **Memory:** Prithvi fits the 16 GB CPU Space. Switch to GPU/ZeroGPU for speed — no code change.
- **Cost:** free on the CPU tier.
