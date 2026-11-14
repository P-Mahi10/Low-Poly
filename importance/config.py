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
class ImportanceConfig:
    """Top-level configuration for the entire importance module."""
    cues_3d: Cues3DConfig = field(default_factory=Cues3DConfig)
    beta: float = 1.0  # Weight for 2D cues in gated fusion: I = c3D * I3D * (1 + beta * c2D * I2D)
    smoothing_iterations: int = 2  # Graph Laplacian smoothing steps on final importance
    random_seed: int = 42
