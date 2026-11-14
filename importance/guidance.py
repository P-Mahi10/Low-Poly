from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class GuidanceOutput:
    """Consolidated multimodal guidance output for low-poly mesh decimation."""
    importance: np.ndarray             # (V,) float64 in [0, 1], fused and smoothed
    feature_curves: List[List[int]]    # List of ordered vertex ID polylines
    pinned_flags: np.ndarray           # (V,) bool, True for vertices on feature curves
    confidence: np.ndarray             # (V,) float64 in [0, 1]
    I3D: np.ndarray                    # (V,) float64 in [0, 1]
    I2D: np.ndarray                    # (V,) float64 in [0, 1]
    c3D: np.ndarray                    # (V,) float64 in [0, 1]
    c2D: np.ndarray                    # (V,) float64 in [0, 1]
    sizing_field: Optional[np.ndarray] = None  # (V,) float64 target edge lengths
