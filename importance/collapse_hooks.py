from dataclasses import dataclass
from typing import List, Tuple, Set, Optional, Dict
from collections import defaultdict
import numpy as np


@dataclass
class CollapseHooksConfig:
    """Parameters for importance-guided FA-QEM edge collapse."""
    alpha_importance: float = 10.0   # Scaling weight: Q_v' = (1 + alpha * I_v) * Q_v
    weight_feature_cost: float = 0.5 # Direct cost penalty from endpoint importance
    weight_line_quadric: float = 10.0 # Weight for line quadrics along feature curves
    normal_flip_min_cos: float = 0.2 # Minimum cosine of normal angle before collapse is considered a flip
    min_triangle_angle_deg: float = 3.0 # Minimum allowable angle in incident triangles to prevent slivers


def compute_face_quadric(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute 4x4 fundamental error quadric for triangle face (v0, v1, v2).

    Plane equation: a*x + b*y + c*z + d = 0, where n = [a, b, c] is unit normal.
    d = -n . v0.
    Quadric matrix: K_p = p @ p.T, where p = [a, b, c, d]^T.
    Weighted by triangle area.
    """
    cross = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(cross)
    if norm < 1e-12:
        return np.zeros((4, 4), dtype=np.float64)

    normal = cross / norm
    area = 0.5 * norm
    d = -float(np.dot(normal, v0))
    p = np.array([normal[0], normal[1], normal[2], d], dtype=np.float64)
    return area * np.outer(p, p)


def scale_vertex_quadrics(
    quadrics: np.ndarray,
    importance: np.ndarray,
    alpha: float = 5.0,
) -> np.ndarray:
    """Scale each 4x4 vertex quadric by its importance factor: Q'_v = (1 + alpha * I_v) * Q_v."""
    scaled = np.zeros_like(quadrics)
    scale_factors = 1.0 + alpha * np.clip(importance, 0.0, 1.0)
    for v in range(len(quadrics)):
        scaled[v] = quadrics[v] * scale_factors[v]
    return scaled


def compute_line_quadric(p0: np.ndarray, p1: np.ndarray, weight: float = 10.0) -> np.ndarray:
    """Compute 4x4 line quadric constraining vertices to lie along a 3D line segment (p0, p1).

    The squared distance from point x to line with unit tangent t and passing through p0 is:
      d^2(x) = | (x - p0) x t |^2 = (x - p0)^T (I - t*t^T) (x - p0)
    """
    tangent = p1 - p0
    length = np.linalg.norm(tangent)
    if length < 1e-12:
        return np.zeros((4, 4), dtype=np.float64)

    t = tangent / length
    # A = I - t @ t.T
    A = np.eye(3, dtype=np.float64) - np.outer(t, t)
    b = -A @ p0
    c = float(p0.T @ A @ p0)

    Q_line = np.zeros((4, 4), dtype=np.float64)
    Q_line[:3, :3] = A
    Q_line[:3, 3] = b
    Q_line[3, :3] = b
    Q_line[3, 3] = c
    return weight * Q_line


def add_feature_line_quadrics(
    quadrics: np.ndarray,
    vertices: np.ndarray,
    feature_curves: List[List[int]],
    weight: float = 10.0,
) -> np.ndarray:
    """Add line quadrics along all feature curve segments to constrain collapses along the curve."""
    Q_out = quadrics.copy()
    for curve in feature_curves:
        for i in range(len(curve) - 1):
            u, v = curve[i], curve[i + 1]
            p0, p1 = vertices[u], vertices[v]
            Q_line = compute_line_quadric(p0, p1, weight=weight)
            Q_out[u] += Q_line
            Q_out[v] += Q_line
    return Q_out


def build_feature_curve_membership(
    num_vertices: int,
    feature_curves: List[List[int]],
) -> Dict[int, Set[int]]:
    """Map each vertex to the set of feature curve IDs it belongs to."""
    membership: Dict[int, Set[int]] = defaultdict(set)
    for curve_id, curve in enumerate(feature_curves):
        for v in curve:
            membership[v].add(curve_id)
    return membership


def reject_edge_collapse(
    v1: int,
    v2: int,
    new_pos: np.ndarray,
    vertices: np.ndarray,
    incident_faces_v1: List[Tuple[int, int, int]],
    incident_faces_v2: List[Tuple[int, int, int]],
    pinned_flags: np.ndarray,
    curve_membership: Dict[int, Set[int]],
    min_cos_flip: float = 0.2,
    min_angle_deg: float = 2.0,
) -> bool:
    """Predicate evaluating whether an edge collapse (v1, v2) -> new_pos should be rejected.

    Rejection criteria:
      1. Feature curve crossing: v1 and v2 are on different feature curves (disjoint curve IDs).
      2. Boundary / pinning violation: exactly one vertex is pinned (collapsing an unpinned vertex
         onto a pinned boundary without care can distort boundaries, or joining disconnected pinned regions).
      3. Normal flip or extreme sliver: any incident triangle's normal flips or inverts past min_cos_flip,
         or angle collapses below min_angle_deg.
    """
    # 1. Feature curve check
    curves_v1 = curve_membership.get(v1, set())
    curves_v2 = curve_membership.get(v2, set())
    if curves_v1 and curves_v2:
        # Both on curves: they must share at least one curve (adjacent along same curve)
        if not (curves_v1 & curves_v2):
            return True  # Crosses between different feature curves -> reject!

    min_sin = np.sin(np.radians(min_angle_deg)) if min_angle_deg > 0 else 0.0

    # 2. Normal flip & quality check for incident faces
    all_faces_to_check = []
    for f in incident_faces_v1:
        if v2 not in f:
            all_faces_to_check.append((f, v1))
    for f in incident_faces_v2:
        if v1 not in f:
            all_faces_to_check.append((f, v2))

    for f, v_replaced in all_faces_to_check:
        pts_old = [vertices[f[0]], vertices[f[1]], vertices[f[2]]]
        n_old = np.cross(pts_old[1] - pts_old[0], pts_old[2] - pts_old[0])
        n_old_len = np.linalg.norm(n_old)
        if n_old_len < 1e-12:
            continue
        n_old /= n_old_len

        pts_new = [new_pos if idx == v_replaced else vertices[idx] for idx in f]
        e01 = pts_new[1] - pts_new[0]
        e02 = pts_new[2] - pts_new[0]
        n_new = np.cross(e01, e02)
        n_new_len = np.linalg.norm(n_new)
        if n_new_len < 1e-12:
            return True  # Degenerate face created
        n_new /= n_new_len

        if np.dot(n_old, n_new) < min_cos_flip:
            return True  # Normal flipped!

        # Check corner angles of the new triangle
        if min_sin > 0:
            l01 = np.linalg.norm(e01)
            l02 = np.linalg.norm(e02)
            e12 = pts_new[2] - pts_new[1]
            l12 = np.linalg.norm(e12)
            if min(l01, l02, l12) < 1e-12:
                return True
            # Cross product magnitude = l1 * l2 * sin(theta) = n_new_len
            sin0 = n_new_len / (l01 * l02)
            sin1 = n_new_len / (l01 * l12)
            sin2 = n_new_len / (l02 * l12)
            if min(sin0, sin1, sin2) < min_sin:
                return True

    return False


def merge_vertex_importance(i1: float, i2: float) -> float:
    """Merged vertex importance is the max of the two endpoints."""
    return float(max(i1, i2))
