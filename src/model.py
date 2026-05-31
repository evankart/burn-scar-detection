"""
Burn scar segmentation model using Prithvi-EO-1.0-100M as the encoder backbone.

Prithvi is a geospatial Vision Transformer pretrained by IBM/NASA on HLS
(Harmonized Landsat Sentinel-2) imagery. We attach an FPN-style decoder that
fuses features from multiple encoder layers for pixel-level burn scar
segmentation.

Architecture rationale: a ViT encoder produces features at a single spatial
resolution (14x14 for 224px input), but different layers encode different
levels of abstraction — early layers capture spectral/texture detail while
deeper layers capture semantic patterns. Tapping layers 3, 5, 8, and 12
(evenly spaced across the 12-layer encoder) and fusing them via top-down
lateral connections (FPN) gives the decoder access to the full abstraction
hierarchy, improving boundary precision over a single-layer baseline.

Reference: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
"""

import importlib.util
import logging

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

PRITHVI_REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"

# Prithvi pretraining normalization stats (surface reflectance, 0-1 scale)
PRITHVI_MEAN = [0.077523, 0.108099, 0.122859, 0.249720, 0.220421, 0.161083]
PRITHVI_STD  = [0.128153, 0.127003, 0.139948, 0.136834, 0.129168, 0.115451]

# 0-indexed encoder layers to tap for multi-scale feature fusion.
# Evenly spaced across the 12-layer ViT: early layers retain fine spectral/
# texture detail, deeper layers capture higher-level semantic patterns.
FEATURE_LAYER_INDICES = [2, 4, 7, 11]


def _load_prithvi_mae_class():
    """Download prithvi_mae.py from HuggingFace and import PrithviMAE."""
    mae_path = hf_hub_download(PRITHVI_REPO, "prithvi_mae.py")
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrithviMAE


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network decoder for ViT-based segmentation.

    All ViT layers produce the same 14×14 spatial resolution, but encode
    different levels of abstraction. FPN fuses them via top-down lateral
    connections (deepest → shallowest), then upsamples to pixel resolution.
    """

    def __init__(self, embed_dim: int = 768, num_classes: int = 2, proj_dim: int = 256):
        super().__init__()
        n_layers = len(FEATURE_LAYER_INDICES)

        # 1×1 projections: embed_dim → proj_dim for each tapped layer
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, proj_dim, 1) for _ in range(n_layers)
        ])

        # 3×3 smoothing after each lateral + top-down addition
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(proj_dim, proj_dim, 3, padding=1),
                nn.BatchNorm2d(proj_dim),
                nn.GELU(),
            ) for _ in range(n_layers)
        ])

        # 14×14 → 224×224 in four 2× stages
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(proj_dim, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, layer_features: list[torch.Tensor]) -> torch.Tensor:
        # Project each layer to common dimension
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, layer_features)]

        # Top-down fusion: add deeper features into shallower ones
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + laterals[i]

        # Smooth and sum all FPN levels
        fpn_outs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        fused = sum(fpn_outs)

        return self.head(self.upsample(fused))


class BurnScarModel(nn.Module):
    """
    Prithvi-EO ViT encoder + FPN decoder for burn scar segmentation.

    The encoder produces 14×14 patch embeddings (768-dim) for a 224×224 input.
    Features from layers 3, 5, 8, and 12 are fused via FPN-style top-down
    lateral connections, then upsampled 14→224 in four 2× stages.

    Normalization: expects bands already divided by 10000 (0-1 reflectance).
    Apply PRITHVI_MEAN / PRITHVI_STD before passing to this model.
    """

    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 6,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        embed_dim = 768

        # --- Encoder: Prithvi ViT ---
        logger.info("Loading Prithvi-EO-1.0-100M encoder...")
        PrithviMAE = _load_prithvi_mae_class()
        mae = PrithviMAE(
            img_size=224,
            patch_size=(1, 16, 16),
            num_frames=3,
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=12,
            num_heads=12,
            encoder_only=True,
        )

        weights_path = hf_hub_download(PRITHVI_REPO, "Prithvi_EO_V1_100M.pt")
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        missing, unexpected = mae.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading Prithvi weights: {missing[:5]}...")
        logger.info("Prithvi-EO-1.0-100M weights loaded")

        self.encoder = mae.encoder

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Prithvi encoder frozen")

        # --- Decoder: multi-layer FPN ---
        self.decoder = FPNDecoder(embed_dim=embed_dim, num_classes=num_classes)

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Prithvi encoder unfrozen")

    def _reshape_encoder_output(self, tokens: torch.Tensor, B: int, h: int, w: int) -> torch.Tensor:
        """Reshape encoder tokens to spatial feature map: (B, 589, 768) → (B, 768, 14, 14)."""
        enc = tokens[:, 1:].contiguous()  # drop CLS → (B, 588, 768)
        enc = enc.reshape(B, 3, h, w, -1).mean(dim=1)  # temporal mean pool → (B, h, w, 768)
        return enc.permute(0, 3, 1, 2).contiguous()  # (B, 768, h, w)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) normalized HLS bands
        Returns:
            logits: (B, num_classes, H, W)
        """
        B, C, H, W = pixel_values.shape
        h = w = H // 16

        # Prithvi expects (B, C, T, H, W) with T=3. Replicate single scene 3×
        # to match pretrained temporal positional embeddings.
        x = pixel_values.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()

        # Encoder → list of 12 hidden states, each (B, 589, 768)
        all_features = self.encoder.forward_features(x)

        # Extract and reshape features from tapped layers
        layer_features = [
            self._reshape_encoder_output(all_features[i], B, h, w)
            for i in FEATURE_LAYER_INDICES
        ]

        return self.decoder(layer_features)
