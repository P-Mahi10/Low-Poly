import numpy as np
import pytest

from importance.config import Cues3DConfig
from importance.io_utils import (
    create_synthetic_sphere,
    create_synthetic_cube,
    create_synthetic_torus,
)
from importance.cues_3d import (
    compute_3d_cues,
    compute_dihedral_angles,
    compute_multiscale_curvature,
    extract_feature_curves,
)
from importance.mesh_prep import compute_normals, build_edge_to_faces, build_vertex_neighbors


def test_cube_dihedral_and_sharp_edges():
    """Verify cube dihedral angles of 90 degrees are detected and chained into curves."""
    verts, faces = create_synthetic_cube(side=2.0)
    config = Cues3DConfig(sharp_edge_threshold_deg=45.0)

    output = compute_3d_cues(verts, faces, config=config)

    # Cube has 12 boundary edges at 90 deg (pi/2 ~= 1.57 rad)
    assert len(output.sharp_edges) > 0
    # Every 8 corner vertices of the cube must be pinned
    assert np.all(output.pinned_flags)
    # Feature curves must exist
    assert len(output.feature_curves) > 0

    # Dihedral angles for sharp edges must be close to pi/2 (1.57 rad)
    for edge in output.sharp_edges:
        angle = output.dihedral_angles[edge]
        assert np.isclose(angle, np.pi / 2.0, atol=1e-4)


def test_sphere_curvature_and_confidence():
    """Verify smooth sphere has no sharp edges and high confidence c3D."""
    verts, faces = create_synthetic_sphere(radius=1.0, subdivisions=3)
    config = Cues3DConfig(sharp_edge_threshold_deg=35.0)

    output = compute_3d_cues(verts, faces, config=config)

    # Smooth sphere should have no sharp edges above 35 degrees
    assert len(output.sharp_edges) == 0
    assert not np.any(output.pinned_flags)
    assert len(output.feature_curves) == 0

    # Confidence c3D should be high (> 0.8) everywhere because surface is smooth
    assert np.all(output.c3D > 0.8)
    assert np.all(output.I3D >= 0.0)
    assert np.all(output.I3D <= 1.0)


def test_noisy_sphere_confidence_degradation():
    """Verify high-frequency noise elevates noise_ratio and reduces confidence c3D."""
    verts, faces = create_synthetic_sphere(radius=1.0, subdivisions=3)

    # Clean sphere
    clean_output = compute_3d_cues(verts, faces)

    # Noisy sphere
    np.random.seed(42)
    noisy_verts = verts + np.random.normal(0, 0.05, size=verts.shape)
    noisy_output = compute_3d_cues(noisy_verts, faces)

    # Mean confidence should drop noticeably on the noisy surface
    assert np.mean(noisy_output.c3D) < np.mean(clean_output.c3D)
    # Fine curvature should be much higher than coarse
    assert np.mean(noisy_output.noise_ratio) > np.mean(clean_output.noise_ratio)


def test_torus_multiscale_curvature():
    """Verify multiscale curvature handles saddle and elliptic regions on a torus."""
    verts, faces = create_synthetic_torus(major_radius=1.5, minor_radius=0.4)
    output = compute_3d_cues(verts, faces)

    assert len(output.I3D) == len(verts)
    assert np.all(np.isfinite(output.I3D))
    assert np.max(output.I3D) > np.min(output.I3D)  # Variation across outer and inner rings


def test_thickness_stub_toggle():
    """Verify thickness flag stub runs cleanly whether False or True."""
    verts, faces = create_synthetic_cube(side=1.0)

    # Default disabled
    out_disabled = compute_3d_cues(verts, faces, config=Cues3DConfig(thickness_enabled=False))
    assert np.all(np.isfinite(out_disabled.I3D))

    # Enabled stub
    out_enabled = compute_3d_cues(verts, faces, config=Cues3DConfig(thickness_enabled=True, weight_thickness=0.2))
    assert np.all(np.isfinite(out_enabled.I3D))
