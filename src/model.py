"""Burn scar segmentation on Prithvi-EO encoders (1.0-100M / 2.0-300M) + an FPN
decoder. Version is config-selectable; see docs/METHODOLOGY.md for the version
comparison and rationale."""

import importlib.util
import logging

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

# ── Version registry ────────────────────────────────────────────────────────

PRITHVI_VERSIONS = {
    "1.0": {
        "repo":        "ibm-nasa-geospatial/Prithvi-EO-1.0-100M",
        "weights":     "Prithvi_EO_V1_100M.pt",
        "mae_file":    "prithvi_mae.py",
        "embed_dim":   768,
        "depth":       12,
        "num_heads":   12,
        "num_frames":  3,
        # Pretraining stats, HLS reflectance (0–1). A brightness gain is applied
        # before z-scoring (see normalize_bands).
        "mean": [0.077523, 0.108099, 0.122859, 0.249720, 0.220421, 0.161083],
        "std":  [0.128153, 0.127003, 0.139948, 0.136834, 0.129168, 0.115451],
        "feature_layers": [2, 4, 7, 11],  # FPN taps, evenly spaced over depth 12
        "bands": ["B02", "B03", "B04", "B8A", "B11", "B12"],
    },
    "2.0": {
        "repo":        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
        "weights":     "Prithvi_EO_V2_300M.pt",
        "mae_file":    "prithvi_mae.py",
        "embed_dim":   1024,
        "depth":       24,
        "num_heads":   16,
        "num_frames":  4,
        # Pretraining stats (raw DN / 10000), from Prithvi-EO-2.0-300M config.
        "mean": [0.10870, 0.13420, 0.14330, 0.27340, 0.19580, 0.13630],
        "std":  [0.22480, 0.21790, 0.21780, 0.18500, 0.12420, 0.10490],
        "feature_layers": [5, 11, 17, 23],  # FPN taps, every 6th of depth 24
        "bands": ["B02", "B03", "B04", "B05", "B06", "B07"],
    },
}

# Default version used by existing code / deployed model.
DEFAULT_VERSION = "1.0"

# Convenience aliases (used by normalize_bands in data.py).
PRITHVI_MEAN = PRITHVI_VERSIONS[DEFAULT_VERSION]["mean"]
PRITHVI_STD  = PRITHVI_VERSIONS[DEFAULT_VERSION]["std"]
FEATURE_LAYER_INDICES = PRITHVI_VERSIONS[DEFAULT_VERSION]["feature_layers"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_mae_class(repo: str, mae_file: str):
    """Download prithvi_mae.py from HuggingFace and import PrithviMAE."""
    mae_path = hf_hub_download(repo, mae_file)
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrithviMAE


# ── Decoder ───────────────────────────────────────────────────────────────────

class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network decoder for ViT-based segmentation.

    All ViT layers produce the same 14×14 spatial resolution, but encode
    different levels of abstraction. FPN fuses them via top-down lateral
    connections (deepest → shallowest), then upsamples to pixel resolution.
    Works with any embed_dim (768 for 1.0, 1024 for 2.0).
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 2,
                 proj_dim: int = 256, n_layers: int = 4):
        super().__init__()

        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, proj_dim, 1) for _ in range(n_layers)
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, 3, padding=1),
                nn.BatchNorm2d(proj_dim),
                nn.GELU(),
            ) for _ in range(n_layers)
        ])
        # 14×14 → 224×224 in four 2× stages
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(proj_dim, 256, 2, stride=2), nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),      nn.BatchNorm2d(128), nn.GELU(),
            nn.ConvTranspose2d(128, 64,  2, stride=2),      nn.BatchNorm2d(64),  nn.GELU(),
            nn.ConvTranspose2d(64,  32,  2, stride=2),      nn.BatchNorm2d(32),  nn.GELU(),
        )
        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, layer_features: list[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, layer_features)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + laterals[i]
        fused = sum(conv(lat) for conv, lat in zip(self.fpn_convs, laterals))
        return self.head(self.upsample(fused))


# ── Main model ────────────────────────────────────────────────────────────────

class BurnScarModel(nn.Module):
    """
    Prithvi-EO ViT encoder + FPN decoder for burn scar segmentation.

    Supports Prithvi 1.0 (100M, ViT-Base) and 2.0 (300M, ViT-Large).
    Select via prithvi_version='1.0' or '2.0'.

    Input: (B, C, H, W) normalized bands — brightness-gained then z-scored.
    Output: (B, num_classes, H, W) logits.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 6,
        freeze_backbone: bool = False,
        prithvi_version: str = DEFAULT_VERSION,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes

        cfg = PRITHVI_VERSIONS[prithvi_version]
        embed_dim   = cfg["embed_dim"]
        num_frames  = cfg["num_frames"]
        self._feature_layers = cfg["feature_layers"]
        self._num_frames = num_frames

        logger.info(f"Loading Prithvi-EO-{prithvi_version} encoder "
                    f"(embed_dim={embed_dim}, depth={cfg['depth']}, "
                    f"num_frames={num_frames})...")

        PrithviMAE = _load_mae_class(cfg["repo"], cfg["mae_file"])
        mae = PrithviMAE(
            img_size=224,
            patch_size=(1, 16, 16),
            num_frames=num_frames,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            encoder_only=True,
        )

        weights_path = hf_hub_download(cfg["repo"], cfg["weights"])
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        missing, unexpected = mae.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing[:5]}...")
        logger.info(f"Prithvi-EO-{prithvi_version} weights loaded")

        self.encoder = mae.encoder

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Prithvi encoder frozen")

        self.decoder = FPNDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            n_layers=len(self._feature_layers),
        )

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Prithvi encoder unfrozen")

    def _reshape_encoder_output(self, tokens: torch.Tensor,
                                B: int, h: int, w: int) -> torch.Tensor:
        """Reshape encoder tokens → spatial feature map (B, embed_dim, h, w).
        Drops CLS token, temporal-mean-pools across frames."""
        enc = tokens[:, 1:].contiguous()                        # drop CLS
        enc = enc.reshape(B, self._num_frames, h, w, -1).mean(dim=1)  # temporal mean
        return enc.permute(0, 3, 1, 2).contiguous()            # (B, D, h, w)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) normalized HLS bands
        Returns:
            logits: (B, num_classes, H, W)
        """
        B, C, H, W = pixel_values.shape
        h = w = H // 16

        # Replicate the single post-fire scene across all temporal frames to
        # satisfy the encoder's (B, C, T, H, W) input shape.
        x = pixel_values.unsqueeze(2).expand(-1, -1, self._num_frames, -1, -1).contiguous()

        all_features = self.encoder.forward_features(x)

        layer_features = [
            self._reshape_encoder_output(all_features[i], B, h, w)
            for i in self._feature_layers
        ]

        return self.decoder(layer_features)
