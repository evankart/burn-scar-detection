"""
Extract Prithvi encoder embeddings for Caldor Fire patches and reduce to 2D with PCA.

Run after training and inference:
    PYTHONPATH=. .venv/bin/python scripts/extract_embeddings.py

Output: data/embeddings/caldor_embeddings.npz
    embeddings_2d  — (N, 2) PCA-projected patch embeddings
    burn_fractions — (N,) fraction of burned pixels per patch (dNBR)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import numpy as np
import torch
import xarray as xr
import yaml
from sklearn.decomposition import PCA

from src.data import _restore_crs, normalize_bands, generate_burn_mask, create_patches
from src.model import BurnScarModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_patch_embeddings(model, patches, device, batch_size=32):
    """
    Run patches through the Prithvi encoder only (no decoder).
    Returns (N, 768) embeddings — one per patch.
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i : i + batch_size]
            images = torch.stack([
                torch.from_numpy(np.ascontiguousarray(p["image"])).float()
                for p in batch
            ]).to(device)

            B, C, H, W = images.shape

            # Replicate across T=3 temporal dim (matches Prithvi pretraining)
            x = images.unsqueeze(2).expand(-1, -1, 3, -1, -1)

            # Encoder forward — list of hidden states
            features = model.encoder.forward_features(x)

            # Last layer: (B, 589, 768) → remove CLS → (B, 588, 768)
            enc = features[-1][:, 1:]

            # Mean over temporal patches (3 × 14×14 = 588) → (B, 768)
            patch_emb = enc.mean(dim=1).cpu().numpy()
            embeddings.append(patch_emb)

            if (i // batch_size) % 5 == 0:
                logger.info(f"  {min(i + batch_size, len(patches))}/{len(patches)} patches")

    return np.concatenate(embeddings, axis=0)


def main():
    config_path = "configs/train_config.yaml"
    checkpoint_path = "checkpoints/best_model.pt"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Load model
    model = BurnScarModel(
        num_classes=config["model"]["num_classes"],
        in_channels=config["model"]["in_channels"],
        freeze_backbone=True,
    )

    if not Path(checkpoint_path).exists():
        logger.error(f"No checkpoint at {checkpoint_path} — run training first")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info("Loaded checkpoint")

    # Load Creek Fire scene and build patches
    cache_dir = Path(config["data"]["cache_dir"])
    pre_path = cache_dir / "woolsey_fire_2018_pre.nc"
    post_path = cache_dir / "woolsey_fire_2018_post.nc"

    if not pre_path.exists():
        logger.error("Woolsey Fire data not cached — run download first")
        sys.exit(1)

    logger.info("Loading Woolsey Fire scene...")
    pre_ds = _restore_crs(xr.open_dataset(pre_path, engine="h5netcdf"))
    post_ds = _restore_crs(xr.open_dataset(post_path, engine="h5netcdf"))
    post_ds = post_ds.rio.reproject_match(pre_ds)

    mask = generate_burn_mask(pre_ds, post_ds)
    image = normalize_bands(post_ds, config["data"]["bands"])

    patches = create_patches(
        image, mask,
        patch_size=config["data"]["patch_size"],
        min_burn_fraction=0.0,  # include all patches for embedding space coverage
    )

    # Cap at 600 patches for speed
    rng = np.random.default_rng(42)
    if len(patches) > 600:
        idx = rng.choice(len(patches), 600, replace=False)
        patches = [patches[i] for i in idx]

    logger.info(f"Extracting embeddings for {len(patches)} patches...")
    embeddings = extract_patch_embeddings(model, patches, device)

    burn_fractions = np.array([p["burn_fraction"] for p in patches])

    # PCA to 2D
    logger.info("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    logger.info(f"PCA explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%}")

    # Save
    out_dir = Path("data/embeddings")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "woolsey_embeddings.npz"
    np.savez_compressed(
        out_path,
        embeddings_2d=embeddings_2d,
        burn_fractions=burn_fractions,
        explained_variance=explained,
    )
    logger.info(f"Saved embeddings to {out_path}")
    logger.info(
        f"  {(burn_fractions > 0.1).sum()} burned patches, "
        f"{(burn_fractions <= 0.1).sum()} unburned patches"
    )


if __name__ == "__main__":
    main()
