from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Cues3DConfig:
    """Configuration for 3D geometric cue extraction."""
    # Multi-scale curvature rings
    curvature_rings: Tuple[int, ...] = (1, 2, 3)

    # Sharp edge threshold in degrees (dihedral angle)
    sharp_edge_threshold_deg: float = 35.0

    # Weights for I3D composition
    weight_curvature: float = 0.5
    weight_sharp_edges: float = 0.5
    weight_thickness: float = 0.0

    # Noise ratio sensitivity for confidence c3D
    noise_ratio_decay: float = 0.5

    # Thickness calculation (optional stub, default False)
    thickness_enabled: bool = False

    # Percentile bounds for robust normalization [p_min, p_max]
    percentile_clip: Tuple[float, float] = (1.0, 99.0)

    # Determinism
    random_seed: int = 42


@dataclass
class Cues2DConfig:
    """Configuration for 2D image cue extraction."""
    # Bilateral filter parameters
    bilateral_d: int = 7
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0

    # Multi-scale edge detection scales (Gaussian blurs)
    edge_blur_sigmas: Tuple[float, ...] = (1.0, 2.0, 4.0)

    # Density normalization window to suppress dense textures (fur, grass)
    density_norm_kernel_size: int = 15

    # Saliency and entropy boost weights in I2D
    weight_saliency: float = 0.5
    weight_entropy: float = 0.5

    # Douglas-Peucker simplification epsilon for 2D polylines
    polyline_threshold: float = 0.35
    douglas_peucker_epsilon: float = 2.0


@dataclass
class LiftingConfig:
    """Configuration for lifting 2D image cues to 3D mesh vertices."""
    depth_tolerance_ratio: float = 0.03  # Max relative depth discrepancy for visibility
    min_grazing_cos: float = 0.05        # Discard views with grazing angle cos(n, d_view) < min_grazing_cos
    variance_penalty: float = 2.0        # Exponential decay on c2D when multi-view variance is high
    z_buffer_resolution: Tuple[int, int] = (480, 640)


@dataclass
class ImportanceConfig:
    """Top-level configuration for the entire importance module."""
    cues_3d: Cues3DConfig = field(default_factory=Cues3DConfig)
    cues_2d: Cues2DConfig = field(default_factory=Cues2DConfig)
    lifting: LiftingConfig = field(default_factory=LiftingConfig)
    beta: float = 1.0  # Weight for 2D cues in gated fusion: I = c3D * I3D * (1 + beta * c2D * I2D)
    smoothing_iterations: int = 2  # Graph Laplacian smoothing steps on final importance
    random_seed: int = 42
