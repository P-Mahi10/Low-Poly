import numpy as np
import pytest

from importance.io_utils import (
    create_synthetic_sphere,
    create_synthetic_cube,
    create_synthetic_torus,
)
from importance.mesh_prep import (
    compute_normals,
    validate_and_repair_mesh,
    taubin_smoothing,
    build_vertex_neighbors,
    build_edge_to_faces,
)


def test_compute_normals_sphere():
    """Verify normals on a unit sphere point radially outward and surface area is close to 4*pi*r^2."""
    verts, faces = create_synthetic_sphere(radius=1.0, subdivisions=3)
    fn, vn, areas = compute_normals(verts, faces, weighting="angle")

    # Vertex normals for a sphere centered at origin must equal normalized vertex positions
    radial_unit = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    dot_products = np.sum(vn * radial_unit, axis=1)
    assert np.all(dot_products > 0.99)

    # Face areas sum should be close to 4 * pi * r^2 = 12.566
    total_area = np.sum(areas)
    expected_area = 4.0 * np.pi * (1.0 ** 2)
    assert np.isclose(total_area, expected_area, rtol=0.05)


def test_compute_normals_cube():
    """Verify face normals on a cube align with coordinate axes."""
    side = 2.0
    verts, faces = create_synthetic_cube(side=side)
    fn, vn, areas = compute_normals(verts, faces, weighting="area")

    # Total area of cube with side 2 is 6 * 4 = 24
    assert np.isclose(np.sum(areas), 24.0, atol=1e-5)

    # Face normals should be unit vectors along +-X, +-Y, or +-Z
    for n in fn:
        assert np.isclose(np.linalg.norm(n), 1.0)
        assert np.isclose(np.max(np.abs(n)), 1.0, atol=1e-5)


def test_validate_and_repair_mesh():
    """Verify removal of degenerate faces, zero-area faces, and unreferenced vertices."""
    verts, faces = create_synthetic_cube(side=1.0)
    num_clean_v = len(verts)
    num_clean_f = len(faces)

    # Inject corruptions:
    # 1. Unreferenced vertex
    corrupt_verts = np.vstack([verts, [[10.0, 10.0, 10.0]]])
    # 2. Degenerate face with duplicate index (v, v, w)
    corrupt_faces = np.vstack([faces, [[0, 0, 1]]])
    # 3. Collinear / zero-area face
    corrupt_faces = np.vstack([corrupt_faces, [[0, 1, 0]]])

    clean_v, clean_f, report = validate_and_repair_mesh(corrupt_verts, corrupt_faces)

    assert report["removed_vertices"] == 1
    assert report["removed_faces"] == 2
    assert len(clean_v) == num_clean_v
    assert len(clean_f) == num_clean_f
    assert report["is_edge_manifold"] is True
    assert report["is_closed"] is True


def test_taubin_smoothing_volume_preservation():
    """Verify Taubin smoothing denoises without causing the catastrophic shrinkage of pure Laplacian smoothing."""
    verts, faces = create_synthetic_sphere(radius=1.0, subdivisions=3)
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, size=verts.shape)
    noisy_verts = verts + noise

    initial_mean_radius = np.mean(np.linalg.norm(noisy_verts, axis=1))

    # Apply Taubin smoothing
    smoothed = taubin_smoothing(noisy_verts, faces, iterations=10, lambda_param=0.5, mu_param=-0.53)
    final_mean_radius = np.mean(np.linalg.norm(smoothed, axis=1))

    # Taubin should maintain radius within 2% of initial, while reducing variance
    assert np.isclose(final_mean_radius, initial_mean_radius, rtol=0.02)

    # Noise variance from unit sphere should decrease
    initial_std = np.std(np.linalg.norm(noisy_verts, axis=1))
    final_std = np.std(np.linalg.norm(smoothed, axis=1))
    assert final_std < initial_std * 0.7


def test_topology_helpers():
    """Verify neighbor and edge adjacency builders."""
    verts, faces = create_synthetic_cube(side=1.0)
    nbrs = build_vertex_neighbors(len(verts), faces)
    edge_to_f = build_edge_to_faces(faces)

    assert len(nbrs) == len(verts)
    for n in nbrs:
        assert len(n) >= 3

    # On a manifold closed cube, every edge has exactly 2 incident faces
    for (u, v), f_list in edge_to_f.items():
        assert len(f_list) == 2
