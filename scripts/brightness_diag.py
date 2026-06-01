import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, torch, xarray as xr
from scipy.ndimage import binary_erosion
from src.data import _restore_crs, generate_burn_mask, load_config
from src.model import BurnScarModel, PRITHVI_MEAN, PRITHVI_STD
from src.utils import get_device, water_mask

cfg = load_config("configs/train_config.yaml")
bands = cfg["data"]["bands"]; ps = 224
dev = get_device()
M = np.array(PRITHVI_MEAN, dtype=np.float32); S = np.array(PRITHVI_STD, dtype=np.float32)

model = BurnScarModel(num_classes=2, in_channels=6)
model.load_state_dict(torch.load("checkpoints/balanced_chaparral/best_model.pt",
                                 map_location=dev, weights_only=False)["model_state_dict"])
model = model.to(dev).eval()

FIRES = ["woolsey_fire_2018", "east_troublesome_2020", "thomas_fire_2017"]
raw = valid = water = true = None  # set per-fire in the loop below

# Global per-band gain calibrated on TRAINING fires only; see docs/METHODOLOGY.md.
TRAIN_SAMPLE = ["august_complex_2020", "mendocino_complex_2018", "caldor_fire_2021",
                "bobcat_2020", "holy_2018", "carr_fire_2018", "dixie_fire_2021", "bootleg_fire_2021"]
_meds = {b: [] for b in bands}
for nm in TRAIN_SAMPLE:
    try:
        d = _restore_crs(xr.open_dataset(f"data/cache/{nm}_post.nc", engine="h5netcdf"))
        for b in bands:
            a = np.clip(d[b].values.astype(np.float32), 0, 1); a = a[np.isfinite(a) & (a > 0)]
            if a.size: _meds[b].append(np.median(a))
    except Exception:
        pass
GLOBAL_GAIN = np.array([M[i] / (np.mean(_meds[b]) + 1e-6) for i, b in enumerate(bands)], dtype=np.float32)
print("Global per-band gains (train-calibrated):", {b: round(float(GLOBAL_GAIN[i]), 2) for i, b in enumerate(bands)})


def norm_global(r):
    return norm_prithvi(r * GLOBAL_GAIN[:, None, None])


def norm_prithvi(r):
    return (r - M[:, None, None]) / S[:, None, None]

def norm_perscene(r):  # standardize each band by its own valid mean/std
    out = np.empty_like(r)
    for i in range(r.shape[0]):
        v = r[i][valid]; out[i] = (r[i] - v.mean()) / (v.std() + 1e-6)
    return out

def norm_gain(r):  # rescale each band so its valid median == Prithvi mean, then Prithvi-normalize
    out = np.empty_like(r)
    for i in range(r.shape[0]):
        med = np.median(r[i][valid]) + 1e-6
        out[i] = r[i] * (M[i] / med)
    return norm_prithvi(out)


def predict(img):
    _, h, w = img.shape
    acc = np.zeros((h, w), np.float32); cnt = np.zeros((h, w), np.float32)
    ys = list(range(0, h - ps + 1, ps // 2)); xs = list(range(0, w - ps + 1, ps // 2))
    if ys and ys[-1] != h - ps: ys.append(h - ps)
    if xs and xs[-1] != w - ps: xs.append(w - ps)
    with torch.no_grad():
        for y in ys:
            for x in xs:
                pv = valid[y:y+ps, x:x+ps]
                if not pv.any(): continue
                patch = np.nan_to_num(img[:, y:y+ps, x:x+ps], nan=0.0)
                t = torch.from_numpy(patch).unsqueeze(0).float().to(dev)
                pr = torch.softmax(model(t), 1)[0, 1].cpu().numpy()
                acc[y:y+ps, x:x+ps][pv] += pr[pv]; cnt[y:y+ps, x:x+ps][pv] += 1
    cov = cnt > 0; acc[cov] /= cnt[cov]
    binp = (acc > 0.5).astype(np.uint8)
    binp[~binary_erosion(valid, iterations=10)] = 0
    binp[water] = 0
    return binp


def metrics(p):
    t = true.copy(); t[water] = 0
    v = valid & ~water
    pp = p[v].astype(bool); tt = t[v].astype(bool)
    tp = int((pp & tt).sum()); fp = int((pp & ~tt).sum()); fn = int((~pp & tt).sum())
    return (tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0,
            tp/(tp+fp+fn) if tp+fp+fn else 0, p[v].mean())

for fire in FIRES:
    pre = _restore_crs(xr.open_dataset(f"data/cache/{fire}_pre.nc", engine="h5netcdf"))
    post = _restore_crs(xr.open_dataset(f"data/cache/{fire}_post.nc", engine="h5netcdf")).rio.reproject_match(pre)
    true = generate_burn_mask(pre, post, dnbr_threshold=0.10)
    raw = np.stack([np.clip(post[b].values.astype(np.float32), 0, 1) for b in bands], 0)
    valid = ~(np.isnan(raw).any(0) | (np.nan_to_num(raw).max(0) == 0))
    water = water_mask(post)
    globals().update(raw=raw, valid=valid, water=water, true=true)
    tfrac = true[valid & ~water].mean()
    print(f"\n=== {fire}  (truth burned frac {tfrac:.3f}) ===")
    print(f"{'normalization':<22}{'Precision':>10}{'Recall':>9}{'IoU':>8}{'pred_frac':>11}")
    for label, fn_ in [("(a) Prithvi [current]", norm_prithvi),
                       ("(c) per-scene gain", norm_gain),
                       ("(d) global gain", norm_global)]:
        p, r, i, frac = metrics(predict(fn_(raw)))
        print(f"{label:<22}{p:>10.3f}{r:>9.3f}{i:>8.3f}{frac:>11.3f}")
