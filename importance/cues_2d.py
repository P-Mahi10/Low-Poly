from dataclasses import dataclass
from typing import List, Tuple, Optional
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from importance.config import Cues2DConfig


@dataclass
class Cues2DOutput:
    """Outputs from the 2D image cue extraction module."""
    I2D: np.ndarray             # (H, W) float64 in [0, 1]
    edge_map: np.ndarray        # (H, W) float64
    saliency_map: np.ndarray    # (H, W) float64
    entropy_map: np.ndarray     # (H, W) float64
    polylines: List[np.ndarray] # List of (N, 2) arrays of pixel coordinates


def edge_preserving_denoise(
    gray_image: np.ndarray,
    d: int = 7,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """Apply bilateral filtering to smooth textures while strictly preserving sharp boundaries."""
    u8 = (np.clip(gray_image, 0.0, 1.0) * 255.0).astype(np.uint8)
    denoised = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    return denoised.astype(np.float64) / 255.0


def compute_multiscale_edges(
    image_gray: np.ndarray,
    sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0),
) -> np.ndarray:
    """Compute multi-scale gradient magnitude edge strength."""
    edge_accum = np.zeros_like(image_gray, dtype=np.float64)

    for sigma in sigmas:
        blurred = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        edge_accum += mag

    if len(sigmas) > 0:
        edge_accum /= len(sigmas)

    return edge_accum


def compute_contrast_saliency(image_gray: np.ndarray) -> np.ndarray:
    """Compute contrast-based visual saliency using spectral residual / center-surround DoG."""
    # Difference-of-Gaussians center-surround saliency
    fine = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
    coarse = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=8.0, sigmaY=8.0)
    saliency = np.abs(fine - coarse)

    max_val = np.max(saliency)
    if max_val > 1e-6:
        saliency /= max_val
    return saliency


def compute_local_entropy(image_gray: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Compute local Shannon entropy map measuring information density/texture complexity."""
    # Discretize to 16 quantization bins for fast histogram entropy
    quantized = np.clip(np.floor(image_gray * 16.0).astype(np.int32), 0, 15)

    pad = kernel_size // 2
    padded = np.pad(quantized, pad, mode="reflect")
    h, w = image_gray.shape

    # Fast block-sliding local entropy approximation via local variance & gradient entropy
    gx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.hypot(gx, gy)

    # Local variance in window
    local_mean = cv2.blur(image_gray, (kernel_size, kernel_size))
    local_sq_mean = cv2.blur(image_gray ** 2, (kernel_size, kernel_size))
    local_var = np.maximum(0.0, local_sq_mean - local_mean ** 2)

    # Combine local variance and gradient activity as an entropy surrogate
    entropy_map = np.sqrt(local_var) * (1.0 + grad_mag)
    max_e = np.max(entropy_map)
    if max_e > 1e-6:
        entropy_map /= max_e
    return entropy_map


def normalize_local_density(
    edge_map: np.ndarray,
    kernel_size: int = 15,
    suppression_factor: float = 3.0,
) -> np.ndarray:
    """Suppress high-density texture clutter (fur, grass, carpet) relative to clean structural edges.

    Dividing edge strength by local edge density downweights areas filled with repeated high-frequency edges.
    """
    local_density = cv2.blur(edge_map, (kernel_size, kernel_size))
    # Normalized edge: clean isolated edges have low local density; dense texture has high density
    normalized = edge_map / (1.0 + suppression_factor * local_density)
    return normalized


def extract_2d_polylines(
    edge_map: np.ndarray,
    threshold: float = 0.35,
    epsilon: float = 2.0,
) -> List[np.ndarray]:
    """Trace edge pixels above threshold and simplify using Douglas-Peucker into 2D polylines."""
    binary = (edge_map >= threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    polylines: List[np.ndarray] = []
    for cnt in contours:
        if len(cnt) < 4:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=False)
        pts = approx.reshape(-1, 2)
        if len(pts) >= 2:
            polylines.append(pts)

    return polylines


def compute_2d_cues(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    config: Optional[Cues2DConfig] = None,
) -> Cues2DOutput:
    """Extract complete 2D image cues from an image and optional silhouette mask.

    Pipeline:
      1. Edge-preserving denoise (Bilateral filter).
      2. Multi-scale edge strength.
      3. Contrast saliency & local entropy.
      4. Local density normalization (suppressing fur, grass, clutter).
      5. Background masking.
      6. Combined I2D map & Douglas-Peucker 2D polylines.
    """
    if config is None:
        config = Cues2DConfig()

    # Convert to grayscale float64 in [0, 1]
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    else:
        gray = image.astype(np.float64) / 255.0 if image.max() > 1.0 else image.astype(np.float64)

    # 1. Bilateral denoise
    denoised = edge_preserving_denoise(
        gray,
        d=config.bilateral_d,
        sigma_color=config.bilateral_sigma_color,
        sigma_space=config.bilateral_sigma_space,
    )

    # 2. Multi-scale edges
    raw_edges = compute_multiscale_edges(denoised, sigmas=config.edge_blur_sigmas)

    # 3. Density normalization (suppressing dense textures like fur or grass)
    norm_edges = normalize_local_density(
        raw_edges,
        kernel_size=config.density_norm_kernel_size,
    )

    # 4. Saliency and Entropy maps
    saliency = compute_contrast_saliency(denoised)
    entropy = compute_local_entropy(denoised)

    # 5. Combined 2D Importance
    I2D_raw = norm_edges * (1.0 + config.weight_saliency * saliency) * (1.0 + config.weight_entropy * entropy)

    # 6. Silhouette Masking
    if mask is not None:
        mask_binary = (mask > 0).astype(np.float64)
        I2D_raw *= mask_binary
        norm_edges *= mask_binary

    # Robust normalization to [0, 1]
    p_low = np.percentile(I2D_raw, 1.0)
    p_high = np.percentile(I2D_raw, 99.0)
    if not np.isclose(p_high, p_low):
        I2D = np.clip((I2D_raw - p_low) / (p_high - p_low), 0.0, 1.0)
    else:
        I2D = np.clip(I2D_raw, 0.0, 1.0)

    # 7. 2D Polylines
    polylines = extract_2d_polylines(
        I2D,
        threshold=config.polyline_threshold,
        epsilon=config.douglas_peucker_epsilon,
    )

    return Cues2DOutput(
        I2D=I2D,
        edge_map=norm_edges,
        saliency_map=saliency,
        entropy_map=entropy,
        polylines=polylines,
    )
