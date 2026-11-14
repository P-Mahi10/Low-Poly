from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import numpy as np

from importance.config import Cues3DConfig
from importance.mesh_prep import (
    compute_normals,
    build_vertex_neighbors,
    build_edge_to_faces,
)


@dataclass
class Cues3DOutput:
    """Outputs from the 3D cue extraction module."""
    I3D: np.ndarray                        # (V,) float64 in [0, 1]
    c3D: np.ndarray                        # (V,) float64 in [0, 1]
    feature_curves: List[List[int]]       # List of ordered vertex ID chains
    pinned_flags: np.ndarray               # (V,) bool
    dihedral_angles: Dict[Tuple[int, int], float]  # Edge -> dihedral angle in radians
    sharp_edges: List[Tuple[int, int]]     # Edges exceeding dihedral threshold
    vertex_sharpness: np.ndarray           # (V,) float64 in [0, 1]
    curvature_fine: np.ndarray             # (V,) float64
    curvature_coarse: np.ndarray           # (V,) float64
    noise_ratio: np.ndarray                # (V,) float64


def compute_k_ring_neighbors(
    num_vertices: int,
    one_ring: List[Set[int]],
    k: int,
) -> List[Set[int]]:
    """Expand 1-ring neighbors to k-ring neighborhood for each vertex."""
    if k <= 1:
        return [set(nbrs) for nbrs in one_ring]

    k_rings: List[Set[int]] = [set(nbrs) for nbrs in one_ring]
    for _ in range(2, k + 1):
        next_rings = [set(s) for s in k_rings]
        for v in range(num_vertices):
            for nbr in k_rings[v]:
                next_rings[v].update(one_ring[nbr])
            next_rings[v].discard(v)
        k_rings = next_rings

    return k_rings


def compute_multiscale_curvature(
    vertices: np.ndarray,
    vertex_normals: np.ndarray,
    one_ring: List[Set[int]],
    rings: Tuple[int, ...] = (1, 2, 3),
) -> Dict[int, np.ndarray]:
    """Compute mean curvature magnitude at multiple neighborhood rings.

    Curvature at vertex i is estimated via normal-projected Laplacian displacement:
        kappa_i = | n_i . (1 / |N(i)| sum_{j in N(i)} (v_j - v_i)) |
    """
    num_verts = len(vertices)
    curvatures: Dict[int, np.ndarray] = {}

    current_ring_neighbors = one_ring
    for r in range(1, max(rings) + 1):
        if r > 1:
            # Expand neighborhood by one ring
            next_ring = [set(s) for s in current_ring_neighbors]
            for v in range(num_verts):
                for nbr in current_ring_neighbors[v]:
                    next_ring[v].update(one_ring[nbr])
                next_ring[v].discard(v)
            current_ring_neighbors = next_ring

        if r in rings:
            curv = np.zeros(num_verts, dtype=np.float64)
            for v in range(num_verts):
                nbrs = current_ring_neighbors[v]
                if len(nbrs) > 0:
                    nbr_pts = vertices[list(nbrs)]
                    displacement = np.mean(nbr_pts - vertices[v], axis=0)
                    curv[v] = np.abs(np.dot(vertex_normals[v], displacement))
            curvatures[r] = curv

    return curvatures


def compute_dihedral_angles(
    faces: np.ndarray,
    face_normals: np.ndarray,
    edge_to_faces: Dict[Tuple[int, int], List[int]],
) -> Dict[Tuple[int, int], float]:
    """Compute dihedral angle in radians for each manifold edge with 2 incident faces."""
    dihedral: Dict[Tuple[int, int], float] = {}

    for edge, f_list in edge_to_faces.items():
        if len(f_list) == 2:
            n1 = face_normals[f_list[0]]
            n2 = face_normals[f_list[1]]
            dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle = float(np.arccos(dot))
            dihedral[edge] = angle
        else:
            # Boundary or non-manifold edge: default to 0.0 or pi
            dihedral[edge] = 0.0

    return dihedral


def extract_feature_curves(
    num_vertices: int,
    sharp_edges: List[Tuple[int, int]],
) -> Tuple[List[List[int]], np.ndarray]:
    """Chain sharp edges into feature curves (polylines) and return pinned flags."""
    pinned_flags = np.zeros(num_vertices, dtype=bool)
    if not sharp_edges:
        return [], pinned_flags

    # Build adjacency on the sharp-edge graph
    adj: Dict[int, Set[int]] = defaultdict(set)
    for u, v in sharp_edges:
        adj[u].add(v)
        adj[v].add(u)
        pinned_flags[u] = True
        pinned_flags[v] = True

    visited_edges: Set[Tuple[int, int]] = set()
    curves: List[List[int]] = []

    def make_edge(a: int, b: int) -> Tuple[int, int]:
        return (min(a, b), max(a, b))

    # First trace from degree-1 endpoints (open polylines)
    endpoints = [v for v, nbrs in adj.items() if len(nbrs) == 1]
    for start_node in endpoints:
        for nbr in adj[start_node]:
            e = make_edge(start_node, nbr)
            if e in visited_edges:
                continue

            chain = [start_node, nbr]
            visited_edges.add(e)
            curr = nbr

            while True:
                next_candidates = [n for n in adj[curr] if make_edge(curr, n) not in visited_edges]
                if len(next_candidates) == 1:
                    nxt = next_candidates[0]
                    visited_edges.add(make_edge(curr, nxt))
                    chain.append(nxt)
                    curr = nxt
                else:
                    break
            curves.append(chain)

    # Next trace remaining cycles/loops
    for start_node, nbrs in adj.items():
        for nbr in nbrs:
            e = make_edge(start_node, nbr)
            if e in visited_edges:
                continue

            chain = [start_node, nbr]
            visited_edges.add(e)
            curr = nbr

            while True:
                next_candidates = [n for n in adj[curr] if make_edge(curr, n) not in visited_edges]
                if len(next_candidates) >= 1:
                    nxt = next_candidates[0]
                    visited_edges.add(make_edge(curr, nxt))
                    chain.append(nxt)
                    curr = nxt
                    if curr == start_node:
                        break
                else:
                    break
            curves.append(chain)

    return curves, pinned_flags


def normalize_scalar(
    values: np.ndarray,
    percentile_clip: Tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Robust normalization of scalar array to [0, 1] using percentile clipping."""
    if len(values) == 0:
        return np.array([], dtype=np.float64)

    vals = np.asarray(values, dtype=np.float64)
    p_low, p_high = np.percentile(vals, percentile_clip)
    if np.isclose(p_high, p_low):
        return np.zeros_like(vals)

    clipped = np.clip(vals, p_low, p_high)
    norm = (clipped - p_low) / (p_high - p_low)
    return np.clip(norm, 0.0, 1.0)


def compute_thickness_stub(
    vertices: np.ndarray,
    faces: np.ndarray,
    enabled: bool = False,
) -> np.ndarray:
    """Thickness estimation stub. Disabled by default to avoid unverified ray-tracing wheels."""
    if not enabled:
        return np.zeros(len(vertices), dtype=np.float64)

    # Lightweight ray-mesh or sphere-inscribed thickness fallback if needed in future
    return np.zeros(len(vertices), dtype=np.float64)


def compute_3d_cues(
    vertices: np.ndarray,
    faces: np.ndarray,
    config: Optional[Cues3DConfig] = None,
) -> Cues3DOutput:
    """Extract complete 3D geometric cues from mesh V and F.

    Pipeline:
      1. Compute normals and areas.
      2. Multi-scale curvature (fine vs coarse).
      3. Dihedral angles and sharp edge detection.
      4. Feature curves and pinned vertex flags.
      5. Noise ratio and confidence c3D.
      6. Combined I3D weighted score.
    """
    if config is None:
        config = Cues3DConfig()

    num_verts = len(vertices)
    face_normals, vertex_normals, face_areas = compute_normals(vertices, faces)
    one_ring = build_vertex_neighbors(num_verts, faces)
    edge_to_f = build_edge_to_faces(faces)

    # 1. Multi-scale curvature
    curv_dict = compute_multiscale_curvature(
        vertices=vertices,
        vertex_normals=vertex_normals,
        one_ring=one_ring,
        rings=config.curvature_rings,
    )
    fine_ring = min(config.curvature_rings)
    coarse_ring = max(config.curvature_rings)

    curv_fine = curv_dict[fine_ring]
    curv_coarse = curv_dict[coarse_ring]

    # 2. Noise ratio and confidence c3D
    # Fine-scale over coarse-scale: high ratio indicates high-frequency surface noise
    eps = 1e-6
    noise_ratio = curv_fine / (curv_coarse + eps)
    # When noise ratio is excessive (> 1.0), confidence is penalized exponentially
    c3D = np.exp(-config.noise_ratio_decay * np.maximum(0.0, noise_ratio - 1.0))
    c3D = np.clip(c3D, 0.0, 1.0)

    # 3. Dihedral angles and sharp edges
    dihedral_angles = compute_dihedral_angles(faces, face_normals, edge_to_f)
    threshold_rad = np.radians(config.sharp_edge_threshold_deg)

    sharp_edges = [edge for edge, angle in dihedral_angles.items() if angle >= threshold_rad]

    # Map sharp edges to per-vertex sharpness score
    vertex_sharpness = np.zeros(num_verts, dtype=np.float64)
    for (u, v) in sharp_edges:
        angle = dihedral_angles.get((u, v), 0.0)
        vertex_sharpness[u] = max(vertex_sharpness[u], angle)
        vertex_sharpness[v] = max(vertex_sharpness[v], angle)

    # 4. Feature curves and pinned flags
    curves, pinned_flags = extract_feature_curves(num_verts, sharp_edges)

    # 5. Thickness (optional stub)
    thickness = compute_thickness_stub(vertices, faces, enabled=config.thickness_enabled)

    # 6. Normalize components and fuse into I3D
    norm_curv = normalize_scalar(curv_fine, config.percentile_clip)
    norm_sharp = normalize_scalar(vertex_sharpness, config.percentile_clip)
    norm_thick = normalize_scalar(thickness, config.percentile_clip) if config.thickness_enabled else thickness

    I3D_raw = (
        config.weight_curvature * norm_curv
        + config.weight_sharp_edges * norm_sharp
        + config.weight_thickness * norm_thick
    )
    total_w = config.weight_curvature + config.weight_sharp_edges + config.weight_thickness
    if total_w > 0:
        I3D_raw /= total_w

    I3D = normalize_scalar(I3D_raw, config.percentile_clip)

    return Cues3DOutput(
        I3D=I3D,
        c3D=c3D,
        feature_curves=curves,
        pinned_flags=pinned_flags,
        dihedral_angles=dihedral_angles,
        sharp_edges=sharp_edges,
        vertex_sharpness=vertex_sharpness,
        curvature_fine=curv_fine,
        curvature_coarse=curv_coarse,
        noise_ratio=noise_ratio,
    )
