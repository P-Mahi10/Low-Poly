import numpy as np
import pytest
import trimesh

from importance.io_utils import (
    create_synthetic_sphere,
    create_synthetic_cube,
    create_synthetic_torus,
    create_camera_opencv,
)
from importance.cues_3d import compute_3d_cues
from importance.collapse_hooks import (
    CollapseHooksConfig,
    compute_face_quadric,
    scale_vertex_quadrics,
    compute_line_quadric,
    reject_edge_collapse,
)
from reference.collapse_loop import reference_qem_decimate
from importance.eval import (
    compute_chamfer_distance,
    compute_silhouette_iou,
    compute_topology_metrics,
    compute_feature_curve_preservation_error,
)


def test_quadric_computation_and_scaling():
    """Verify face quadric formulation and importance scaling."""
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    q = compute_face_quadric(v0, v1, v2)

    assert q.shape == (4, 4)
    assert np.allclose(q, q.T)  # Must be symmetric
    # Point on the triangle plane (z=0) should evaluate to 0 error
    p_on_plane = np.array([0.2, 0.2, 0.0, 1.0])
    assert np.isclose(float(p_on_plane.T @ q @ p_on_plane), 0.0, atol=1e-8)

    # Point off the plane (z=1.0) should have positive error
    p_off_plane = np.array([0.2, 0.2, 1.0, 1.0])
    err = float(p_off_plane.T @ q @ p_off_plane)
    assert err > 0.0

    # Verify quadric scaling hook: Q' = (1 + alpha * I) * Q
    quadrics = np.array([q, q])
    importance = np.array([0.0, 1.0])
    scaled = scale_vertex_quadrics(quadrics, importance, alpha=4.0)

    assert np.allclose(scaled[0], q)           # 1 + 4*0 = 1x
    assert np.allclose(scaled[1], 5.0 * q)     # 1 + 4*1 = 5x


def test_reference_decimation_on_5k_sphere():
    """Test decimation on a 5,120-face mesh (subdivisions=4 icosphere) down to target count."""
    mesh_5k = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    verts = np.asarray(mesh_5k.vertices, dtype=np.float64)
    faces = np.asarray(mesh_5k.faces, dtype=np.int64)

    initial_face_count = len(faces)
    assert 5000 <= initial_face_count <= 20000

    target_faces = 1000
    simp_v, simp_f, simp_imp = reference_qem_decimate(
        verts, faces, target_faces=target_faces
    )

    assert len(simp_f) <= target_faces + 10
    assert len(simp_v) > 0
    assert np.all(np.isfinite(simp_v))

    # Evaluate geometry preservation via Chamfer distance
    mean_chamfer, _, _ = compute_chamfer_distance(verts, simp_v)
    assert mean_chamfer < 0.05  # Tight geometric bound on unit sphere


def test_importance_guided_density_bias():
    """Verify that high importance retains more vertices compared to zero importance."""
    mesh_5k = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    verts = np.asarray(mesh_5k.vertices, dtype=np.float64)
    faces = np.asarray(mesh_5k.faces, dtype=np.int64)

    # Set importance: high (1.0) on Northern hemisphere (Z > 0.2), low (0.0) on Southern (Z < -0.2)
    importance = np.zeros(len(verts), dtype=np.float64)
    importance[verts[:, 2] > 0.2] = 1.0

    target_faces = 1200
    simp_v, simp_f, _ = reference_qem_decimate(
        verts, faces, target_faces=target_faces, importance=importance,
        config=CollapseHooksConfig(alpha_importance=20.0, weight_feature_cost=1.0)
    )

    # Count remaining vertices in North vs South hemispheres
    north_remaining = np.count_nonzero(simp_v[:, 2] > 0.2)
    south_remaining = np.count_nonzero(simp_v[:, 2] < -0.2)

    # Northern (high importance) hemisphere should preserve significantly more vertices!
    assert north_remaining > south_remaining * 1.5


def test_evaluation_metrics_chamfer_and_silhouette():
    """Verify Chamfer distance, multi-view silhouette IoU, and topology metrics."""
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    simp_v, simp_f, _ = reference_qem_decimate(verts, faces, target_faces=1500)

    # 1. Topology metrics
    topo = compute_topology_metrics(simp_v, simp_f)
    assert np.isclose(topo["mean_valence"], 6.0, atol=0.8)
    assert topo["min_angle_deg"] > 1.0

    # 2. Multi-view cameras for silhouette IoU
    K = np.array([[400.0, 0, 200], [0, 400.0, 200], [0, 0, 1]], dtype=np.float64)
    cams = [
        create_camera_opencv(K, np.eye(3), np.array([0.0, 0.0, -3.0]), 400, 400),
        create_camera_opencv(K, np.eye(3), np.array([0.0, 0.0, -4.0]), 400, 400),
    ]

    iou = compute_silhouette_iou(verts, faces, simp_v, simp_f, cams)
    assert iou > 0.90  # Silhouette must match very closely on icosphere decimation


def test_feature_curve_preservation():
    """Verify feature curve distance metric."""
    verts, faces = create_synthetic_cube(side=2.0)
    cues = compute_3d_cues(verts, faces)

    simp_v, simp_f, _ = reference_qem_decimate(
        verts, faces, target_faces=len(faces), feature_curves=cues.feature_curves
    )

    err = compute_feature_curve_preservation_error(cues.feature_curves, verts, simp_v)
    assert np.isclose(err, 0.0, atol=1e-5)
