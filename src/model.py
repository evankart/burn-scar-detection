"""
Burn scar segmentation model using Prithvi-EO-1.0-100M as the encoder backbone.

Prithvi is a geospatial Vision Transformer pretrained by IBM/NASA on HLS
(Harmonized Landsat Sentinel-2) imagery. We attach a lightweight CNN decoder
for pixel-level burn scar segmentation.

Reference: https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M
"""

import importlib.util
import logging

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

PRITHVI_REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"

# Prithvi pretraining normalization stats (raw DN ÷ 10000 → 0-1 scale)
# Source: config.json mean/std divided by 10000
PRITHVI_MEAN = [0.077523, 0.108099, 0.122859, 0.249720, 0.220421, 0.161083]
PRITHVI_STD  = [0.128153, 0.127003, 0.139948, 0.136834, 0.129168, 0.115451]


def _load_prithvi_mae_class():
    """Download prithvi_mae.py from HuggingFace and import PrithviMAE."""
    mae_path = hf_hub_download(PRITHVI_REPO, "prithvi_mae.py")
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PrithviMAE


class BurnScarModel(nn.Module):
    """
    Prithvi-EO ViT encoder + CNN upsampling decoder for burn scar segmentation.

    The encoder produces 14×14 patch embeddings (768-dim) for a 224×224 input.
    The decoder upsamples 14→224 in four 2× stages.

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
        # num_frames=3 matches the pretrained positional embeddings (3×14×14+1=589 tokens)
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

        self.encoder = mae.encoder  # PrithviViT

        if freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Prithvi encoder frozen")

        # --- Decoder: 14×14 → 224×224 via 4× bilinear 2× upsampling ---
        self.decoder = nn.Sequential(
            # 14 → 28
            nn.ConvTranspose2d(embed_dim, 512, kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.GELU(),
            # 28 → 56
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 56 → 112
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 112 → 224
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # logits
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def unfreeze_backbone(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Prithvi encoder unfrozen")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W) normalized Sentinel-2 bands

        Returns:
            logits: (B, num_classes, H, W)
        """
        B, C, H, W = pixel_values.shape

        # Prithvi expects (B, C, T, H, W) with T=3.
        # We only have a single post-fire scene, so replicate it 3× to match
        # the pretrained temporal positional embeddings.
        # .contiguous() is required — expand() is non-contiguous and breaks the backward pass.
        x = pixel_values.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()  # (B, C, 3, H, W)

        # Encoder → list of hidden states, each (B, 3*14*14+1, embed_dim) = (B, 589, 768)
        features = self.encoder.forward_features(x)

        # Remove CLS token → (B, 588, 768), reshape to (B, 3, 14, 14, 768),
        # mean-pool over temporal dim → (B, 14, 14, 768) → (B, 768, 14, 14)
        enc = features[-1][:, 1:].contiguous()  # drop CLS; .contiguous() for reshape
        h = w = H // 16  # 224//16 = 14
        enc = enc.reshape(B, 3, h, w, -1).mean(dim=1)  # temporal mean pool → (B, 14, 14, 768)
        feature_map = enc.permute(0, 3, 1, 2).contiguous()  # (B, 768, 14, 14); contiguous for ConvTranspose2d backward

        return self.decoder(feature_map)
