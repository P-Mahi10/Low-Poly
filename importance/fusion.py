from typing import List, Set, Optional, Tuple
import numpy as np

from importance.config import ImportanceConfig
from importance.cues_3d import Cues3DOutput, normalize_scalar
from importance.guidance import GuidanceOutput
from importance.mesh_prep import build_vertex_neighbors


def smooth_scalar_on_mesh(
    values: np.ndarray,
    neighbors: List[Set[int]],
    iterations: int = 2,
    blend_weight: float = 0.5,
) -> np.ndarray:
    """Apply graph Laplacian neighborhood averaging to smooth a scalar field on the mesh."""
    if iterations <= 0:
        return values.copy()

    smoothed = values.copy()
    num_verts = len(values)

    for _ in range(iterations):
        next_vals = np.zeros(num_verts, dtype=np.float64)
        for v in range(num_verts):
            nbrs = neighbors[v]
            if len(nbrs) > 0:
                avg_nbr = float(np.mean([smoothed[n] for n in nbrs]))
                next_vals[v] = (1.0 - blend_weight) * smoothed[v] + blend_weight * avg_nbr
            else:
                next_vals[v] = smoothed[v]
        smoothed = next_vals

    return smoothed


def compute_target_sizing_field(
    importance: np.ndarray,
    min_edge_len: float = 0.02,
    max_edge_len: float = 0.2,
) -> np.ndarray:
    """Compute target edge length sizing field inversely proportional to importance: h(v) ~ 1/(1+I)."""
    # High importance -> small target edge length (dense triangles)
    # Low importance -> large target edge length (coarse triangles)
    inv_imp = 1.0 - np.clip(importance, 0.0, 1.0)
    sizing = min_edge_len + inv_imp * (max_edge_len - min_edge_len)
    return sizing


def fuse_and_package_guidance(
    vertices: np.ndarray,
    faces: np.ndarray,
    cues_3d: Cues3DOutput,
    lifted_I2D: np.ndarray,
    c2D: np.ndarray,
    config: Optional[ImportanceConfig] = None,
) -> GuidanceOutput:
    """Perform gated fusion of 3D and lifted 2D cues and package into GuidanceOutput.

    Gated Fusion:
      I_fused = c3D * I3D * (1 + beta * c2D * I2D)
    """
    if config is None:
        config = ImportanceConfig()

    num_verts = len(vertices)
    I3D = cues_3d.I3D
    c3D = cues_3d.c3D

    # 1. Gated Multi-Modal Fusion
    I_fused = c3D * I3D * (1.0 + config.beta * c2D * lifted_I2D)

    # 2. Robust Percentile Normalization
    I_norm = normalize_scalar(I_fused, config.cues_3d.percentile_clip)

    # 3. Graph Laplacian Smoothing
    neighbors = build_vertex_neighbors(num_verts, faces)
    I_smoothed = smooth_scalar_on_mesh(
        I_norm,
        neighbors,
        iterations=config.smoothing_iterations,
    )
    I_final = np.clip(I_smoothed, 0.0, 1.0)

    # 4. Overall confidence
    confidence = np.clip(0.5 * (c3D + c2D), 0.0, 1.0)

    # 5. Sizing field
    sizing = compute_target_sizing_field(I_final)

    return GuidanceOutput(
        importance=I_final,
        feature_curves=cues_3d.feature_curves,
        pinned_flags=cues_3d.pinned_flags,
        confidence=confidence,
        I3D=I3D,
        I2D=lifted_I2D,
        c3D=c3D,
        c2D=c2D,
        sizing_field=sizing,
    )
