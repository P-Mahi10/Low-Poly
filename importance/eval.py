from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import scipy.spatial as spatial
import cv2

from importance.io_utils import Camera


def compute_chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute bidirectional Chamfer distance between two point sets.

    Returns:
        mean_chamfer: (dist_a_to_b + dist_b_to_a) / 2
        dist_a_to_b: Mean distance from A to nearest point in B
        dist_b_to_a: Mean distance from B to nearest point in A
    """
    tree_b = spatial.cKDTree(points_b)
    dist_a, _ = tree_b.query(points_a)

    tree_a = spatial.cKDTree(points_a)
    dist_b, _ = tree_a.query(points_b)

    d_ab = float(np.mean(dist_a))
    d_ba = float(np.mean(dist_b))
    mean_d = 0.5 * (d_ab + d_ba)
    return mean_d, d_ab, d_ba


def render_mesh_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    """Rasterize a binary silhouette mask (H, W uint8) for a mesh under a given camera."""
    mask = np.zeros((camera.height, camera.width), dtype=np.uint8)
    uv, depth = camera.project(vertices)
    vis = camera.is_visible(uv, depth)

    # Project each triangle whose vertices are visible or near bounds
    for f in faces:
        pts = uv[f]
        if np.any(vis[f]):
            pts_int = np.rint(pts).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts_int], 255)

    return mask


def compute_silhouette_iou(
    verts_orig: np.ndarray,
    faces_orig: np.ndarray,
    verts_dec: np.ndarray,
    faces_dec: np.ndarray,
    cameras: List[Camera],
) -> float:
    """Compute mean Silhouette Intersection-over-Union (IoU) across a set of camera viewpoints."""
    if not cameras:
        return 1.0

    ious = []
    for cam in cameras:
        mask_orig = render_mesh_silhouette(verts_orig, faces_orig, cam)
        mask_dec = render_mesh_silhouette(verts_dec, faces_dec, cam)

        inter = np.count_nonzero((mask_orig > 0) & (mask_dec > 0))
        union = np.count_nonzero((mask_orig > 0) | (mask_dec > 0))

        iou = float(inter) / float(union) if union > 0 else 1.0
        ious.append(iou)

    return float(np.mean(ious))


def compute_topology_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Dict[str, Any]:
    """Compute topological quality metrics: valence distribution and triangle minimum angles."""
    num_verts = len(vertices)
    if len(faces) == 0 or num_verts == 0:
        return {"mean_valence": 0.0, "min_angle_deg": 0.0, "valences": []}

    # Valence (degree) of each vertex
    valences = np.zeros(num_verts, dtype=np.int32)
    for f in faces:
        valences[f[0]] += 1
        valences[f[1]] += 1
        valences[f[2]] += 1

    # Triangle corner angles
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e01 = v1 - v0
    e12 = v2 - v1
    e20 = v0 - v2

    def norm_vecs(v):
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    u01 = norm_vecs(e01)
    u02 = norm_vecs(v2 - v0)
    u10 = norm_vecs(v0 - v1)
    u12 = norm_vecs(e12)
    u20 = norm_vecs(e20)
    u21 = norm_vecs(v1 - v2)

    a0 = np.arccos(np.clip(np.sum(u01 * u02, axis=1), -1.0, 1.0))
    a1 = np.arccos(np.clip(np.sum(u10 * u12, axis=1), -1.0, 1.0))
    a2 = np.arccos(np.clip(np.sum(u20 * u21, axis=1), -1.0, 1.0))

    all_angles_deg = np.degrees(np.hstack([a0, a1, a2]))

    return {
        "mean_valence": float(np.mean(valences)),
        "std_valence": float(np.std(valences)),
        "min_angle_deg": float(np.min(all_angles_deg)) if len(all_angles_deg) > 0 else 0.0,
        "mean_angle_deg": float(np.mean(all_angles_deg)) if len(all_angles_deg) > 0 else 0.0,
    }


def compute_feature_curve_preservation_error(
    feature_curves: List[List[int]],
    orig_vertices: np.ndarray,
    decimated_vertices: np.ndarray,
) -> float:
    """Measure mean distance from original feature curve vertices to nearest decimated surface vertices."""
    if not feature_curves:
        return 0.0

    curve_vert_indices = list({v for curve in feature_curves for v in curve})
    if not curve_vert_indices:
        return 0.0

    curve_pts = orig_vertices[curve_vert_indices]
    tree = spatial.cKDTree(decimated_vertices)
    dists, _ = tree.query(curve_pts)
    return float(np.mean(dists))
