# Remaining work (cloud / external data)

The local code for all seven original items is implemented (Fmask masking,
multi-tile warning, Optuna script, notebook updates, BurnScars comparison
section). What's left runs on the cloud GPU box or needs external data.

## A. Run the Optuna search ☁️
```bash
SELF_TERMINATE=1 bash cloud/run_optuna.sh      # N_TRIALS=10 EPOCHS=8 default
```
Uploads `checkpoints/optuna/` (study.pkl, best_params.yaml, plots) and
`configs/finetune_optuna_config.yaml` to S3. Objective: val burn-class IoU on
carr + holy. See `cloud/RUNBOOK.md`.

## B. Retrain with the tuned config ☁️ (after A)
```bash
EXP=finetune_v3 CONFIG=configs/finetune_optuna_config.yaml \
  SELF_TERMINATE=1 bash cloud/run_job.sh
```
Then `run_inference.py` on the four test fires at threshold 0.5 and update the
README results table.

## C. Populate the deferred notebook cells ☁️ (after A/B)
- Optuna results cell: pull `checkpoints/optuna/` from S3, run the cell.
- BurnScars comparison cell: `pip install terratorch`, run on the GPU box.

## D. Global fire expansion ✅ (config done — imagery download still ☁️)
Done: 55 real GlobFire/GWIS events (year ≥ 2015, burned area > 10,000 ha) added
to `configs/train_config.yaml`, bringing the registry to **92 train + 4 test =
96 fires**. Source rows live in `data/globfire/*.csv` (one CSV per biome) and
were converted with `scripts/globfire_to_config.py`:

```bash
for b in south_america_cerrado:sa africa_savanna:af mediterranean_shrubland:med \
         australia_eucalyptus:au canada_boreal:ca siberia_taiga:ru; do
  csv=${b%:*}; tag=${b#*:}
  python scripts/globfire_to_config.py --csv data/globfire/$csv.csv --tag $tag \
    --min-sep-km 50 --max 10 --append-to configs/train_config.yaml
done
```

All entries are `role: train` (the script's 60 km leakage guard skips anything
near a test fire) and biome-balanced (cerrado, sub-Saharan savanna,
Mediterranean shrubland, Australian eucalyptus, Canadian boreal, Siberian
taiga). Fmask masking (the stated prerequisite) is already in place.

Remaining ☁️: download the imagery in-region on AWS
(`run_training.py --download-only`), then fold into the retrain (B).
