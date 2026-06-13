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

## D. Global fire expansion ⏳ (needs real GWIS data)
Target ~100 fires (currently 37). Source real fire metadata (coords + pre/post
dates) from GWIS (gwis.jrc.ec.europa.eu): year ≥ 2015, burned area > 10,000 ha.
Add each to `configs/train_config.yaml` under `data.fires` with `role: train`,
prioritizing biome diversity (Australia eucalyptus, Canada/Siberia boreal &
taiga, Mediterranean, South American cerrado, sub-Saharan savanna). Do not add
test fires. Fmask masking (the stated prerequisite) is already in place.
Download via `run_training.py --download-only` in-region on AWS, then fold into
the retrain (B).
