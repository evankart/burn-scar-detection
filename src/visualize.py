"""
Visualization utilities for burn scar predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


BURN_CMAP = mcolors.ListedColormap(["#2d6a4f", "#d62828"])  # green=bg, red=burn

# USGS dNBR burn-severity classes (Key & Benson). Each entry is
# (label, dnbr_low_inclusive, dnbr_high_exclusive, hex_color). Pixels below the
# lowest break (dNBR < 0.10) are unburned/regrowth and render transparent.
SEVERITY_CLASSES = [
    ("Low", 0.10, 0.27, "#ffffb2"),
    ("Moderate-low", 0.27, 0.44, "#fecc5c"),
    ("Moderate-high", 0.44, 0.66, "#fd8d3c"),
    ("High", 0.66, float("inf"), "#bd0026"),
]


def plot_predictions(
    image: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Side-by-side plot: RGB composite | dNBR | prediction | overlay.

    Args:
        image: (C, H, W) normalized HLS bands (uses B4, B3, B2 for RGB)
        true_mask: (H, W) ground truth mask
        pred_mask: (H, W) predicted mask
        title: figure title
        save_path: optional path to save the figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    rgb = np.stack([image[2], image[1], image[0]], axis=-1).astype(np.float32)
    rgb = _percentile_stretch(rgb)
    rgb = np.nan_to_num(rgb, nan=0.0)
    axes[0].imshow(rgb)
    axes[0].set_title("HLS RGB")

    axes[1].imshow(true_mask, cmap=BURN_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("dNBR Burn Severity")

    axes[2].imshow(pred_mask, cmap=BURN_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Model Prediction")

    axes[3].imshow(rgb)
    burn_overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    burn_overlay[pred_mask == 1] = [1, 0.15, 0.15, 0.5]
    axes[3].imshow(burn_overlay)
    axes[3].set_title("Prediction Overlay")

    legend_elements = [
        Patch(facecolor="#2d6a4f", label="Unburned"),
        Patch(facecolor="#d62828", label="Burn Scar"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=8)

    for ax in axes:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(history: dict, save_path: str | None = None) -> plt.Figure:
    """Plot training and validation loss/IoU curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history["train"]) + 1)
    train_loss = [m["loss"] for m in history["train"]]
    val_loss = [m["loss"] for m in history["val"]]
    train_iou = [m["mean_iou"] for m in history["train"]]
    val_iou = [m["mean_iou"] for m in history["val"]]

    ax1.plot(epochs, train_loss, "b-", label="Train", linewidth=2)
    ax1.plot(epochs, val_loss, "r-", label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_iou, "b-", label="Train", linewidth=2)
    ax2.plot(epochs, val_iou, "r-", label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean IoU")
    ax2.set_title("Training & Validation IoU")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _percentile_stretch(rgb: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch each channel independently to [0, 1] using percentile clipping."""
    out = np.empty_like(rgb, dtype=np.float32)
    for c in range(rgb.shape[-1]):
        ch = rgb[..., c]
        finite = ch[np.isfinite(ch)]
        if finite.size == 0:
            out[..., c] = 0.0
            continue
        p_lo, p_hi = np.percentile(finite, [lo, hi])
        if p_hi > p_lo:
            out[..., c] = np.clip((ch - p_lo) / (p_hi - p_lo), 0, 1)
        else:
            out[..., c] = 0.0
    return out


def create_sentinel_overlay(
    image: np.ndarray,
    bounds: list[list[float]],
    max_size: int = 1024,
    rgb_indices: tuple[int, int, int] = (2, 1, 0),
):
    """
    Create a Folium ImageOverlay showing HLS RGB imagery.

    Args:
        image: (C, H, W) HLS bands (may be Prithvi z-score normalized)
        bounds: [[south, west], [north, east]] lat/lon bounds
        max_size: downsample so the longer dimension is at most this many pixels
        rgb_indices: which band indices to use for R, G, B (default: B04, B03, B02)
    """
    import folium

    r, g, b = rgb_indices
    rgb = np.stack([image[r], image[g], image[b]], axis=-1).astype(np.float32)
    rgb = _percentile_stretch(rgb)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb_uint8 = (rgb * 255).astype(np.uint8)

    # Nodata: NaN anywhere across bands (z-score of raw 0 is ~-0.6, not 0)
    valid = ~np.isnan(image).any(axis=0)
    alpha = (valid * 255).astype(np.uint8)
    rgba = np.dstack([rgb_uint8, alpha])

    h, w = rgba.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        try:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(rgba, mode="RGBA")
            pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
            rgba = np.array(pil_img)
        except ImportError:
            step_y = max(1, h // new_h)
            step_x = max(1, w // new_w)
            rgba = rgba[::step_y, ::step_x]

    return folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=1.0,
        name="HLS RGB",
    )


def create_folium_overlay(
    pred_mask: np.ndarray,
    bounds: list[list[float]],
    opacity: float = 0.5,
):
    """
    Create a Folium-compatible image overlay for burn scar predictions.

    Args:
        pred_mask: (H, W) binary prediction mask
        bounds: [[south, west], [north, east]] lat/lon bounds
        opacity: overlay transparency
    """
    import folium

    rgba = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
    rgba[pred_mask == 1] = [214, 40, 40, int(255 * opacity)]
    rgba[pred_mask == 0] = [0, 0, 0, 0]

    return folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=1.0,
        name="Burn Scar Predictions",
    )


def create_severity_overlay(
    dnbr: np.ndarray,
    bounds: list[list[float]],
    opacity: float = 0.7,
    max_size: int = 1024,
    show: bool = True,
):
    """
    Create a Folium ImageOverlay coloring dNBR by USGS burn-severity class.

    Args:
        dnbr: (H, W) continuous dNBR array (NaN = nodata)
        bounds: [[south, west], [north, east]] lat/lon bounds
        opacity: alpha applied to colored (burned) pixels; unburned stays clear
        max_size: downsample so the longer dimension is at most this many pixels
    """
    import folium

    alpha = int(255 * opacity)
    rgba = np.zeros((*dnbr.shape, 4), dtype=np.uint8)
    for _, low, high, hexcol in SEVERITY_CLASSES:
        r, g, b = (np.array(mcolors.to_rgb(hexcol)) * 255).astype(np.uint8)
        # NaN compares False, so nodata never matches and stays transparent.
        sel = (dnbr >= low) & (dnbr < high)
        rgba[sel] = [r, g, b, alpha]

    h, w = rgba.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        try:
            from PIL import Image as PILImage
            # NEAREST: severity classes are categorical, do not interpolate.
            pil_img = PILImage.fromarray(rgba, mode="RGBA").resize(
                (new_w, new_h), PILImage.NEAREST
            )
            rgba = np.array(pil_img)
        except ImportError:
            step_y = max(1, h // new_h)
            step_x = max(1, w // new_w)
            rgba = rgba[::step_y, ::step_x]

    return folium.raster_layers.ImageOverlay(
        image=rgba,
        bounds=bounds,
        opacity=1.0,
        name="dNBR Burn Severity",
        show=show,
    )
