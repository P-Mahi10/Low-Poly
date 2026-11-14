import numpy as np
import pytest
import cv2

from importance.config import Cues2DConfig, LiftingConfig, ImportanceConfig
from importance.io_utils import (
    create_synthetic_sphere,
    create_synthetic_cube,
    create_camera_opencv,
)
from importance.mesh_prep import compute_normals
from importance.cues_3d import compute_3d_cues
from importance.cues_2d import (
    compute_2d_cues,
    edge_preserving_denoise,
    compute_multiscale_edges,
    normalize_local_density,
    extract_2d_polylines,
)
from importance.lifting import (
    lift_2d_cues_to_vertices,
    render_mesh_z_buffer,
)
from importance.fusion import (
    fuse_and_package_guidance,
    smooth_scalar_on_mesh,
    compute_target_sizing_field,
)


def test_2d_cues_on_synthetic_image():
    """Verify 2D cue extraction, edge detection, density normalization, and masking."""
    h, w = 200, 200
    # Synthetic image: dark background with bright centered circle
    image = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(image, (100, 100), 40, (255, 255, 255), -1)

    # Synthetic mask covering only the center
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[50:150, 50:150] = 255

    cues_out = compute_2d_cues(image, mask=mask)

    assert cues_out.I2D.shape == (h, w)
    assert np.all(cues_out.I2D >= 0.0)
    assert np.all(cues_out.I2D <= 1.0)

    # Edge map should peak along the circle perimeter (radius ~40)
    # Background outside the mask must be strictly 0.0
    assert np.all(cues_out.I2D[:30, :] == 0.0)
    assert np.all(cues_out.I2D[170:, :] == 0.0)

    # Polylines should trace the circle boundary
    assert len(cues_out.polylines) > 0
    for poly in cues_out.polylines:
        assert poly.ndim == 2
        assert poly.shape[1] == 2


def test_texture_density_normalization():
    """Verify local density normalization suppresses dense high-frequency grid clutter."""
    h, w = 100, 100
    # Isolated clean edge
    isolated = np.zeros((h, w), dtype=np.float64)
    isolated[:, 50] = 1.0

    # Dense texture pattern (grid lines every 2 pixels)
    dense = np.zeros((h, w), dtype=np.float64)
    dense[::2, :] = 1.0
    dense[:, ::2] = 1.0

    norm_iso = normalize_local_density(isolated, kernel_size=9, suppression_factor=5.0)
    norm_dense = normalize_local_density(dense, kernel_size=9, suppression_factor=5.0)

    # The peak of the isolated edge should remain much stronger than the dense grid edges
    assert np.max(norm_iso) > np.max(norm_dense) * 2.0


def test_lifting_2d_to_3d_visibility_and_grazing():
    """Verify lifting projects cues only onto visible, front-facing vertices."""
    verts, faces = create_synthetic_sphere(radius=1.0, subdivisions=3)
    _, vert_normals, _ = compute_normals(verts, faces)

    # Place camera at (0, 0, 3) pointing towards origin (looking down -Z in world)
    # Using OpenCV convention: camera looks along its +Z axis
    # We position camera at (0, 0, 3) and invert Z axis to face the origin
    K = np.array([[300.0, 0, 150], [0, 300.0, 150], [0, 0, 1]], dtype=np.float64)
    # Rotation facing -Z in world: R flips X and Z
    R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    t = np.array([0.0, 0.0, 3.0], dtype=np.float64)
    cam = create_camera_opencv(K, R, t, width=300, height=300)

    # 2D importance map: uniformly 1.0
    i2d_map = np.ones((300, 300), dtype=np.float64)

    lifted_I2D, c2D = lift_2d_cues_to_vertices(
        verts, faces, vert_normals, cameras=[cam], image_cues_list=[i2d_map]
    )

    # Vertices facing camera (+Z in world, near (0, 0, 1)) should have high lifted I2D and c2D
    front_verts = verts[:, 2] > 0.5
    back_verts = verts[:, 2] < -0.3

    assert np.mean(lifted_I2D[front_verts]) > 0.5
    assert np.mean(c2D[front_verts]) > 0.3
    assert np.max(c2D[front_verts]) > 0.7

    # Back-facing vertices must have zero lifted cues (culled by grazing angle / occlusion)
    assert np.all(lifted_I2D[back_verts] == 0.0)
    assert np.all(c2D[back_verts] == 0.0)


def test_gated_fusion_and_guidance_packaging():
    """Verify gated fusion formula and sizing field calculation."""
    verts, faces = create_synthetic_cube(side=1.0)
    cues_3d = compute_3d_cues(verts, faces)

    num_verts = len(verts)
    lifted_I2D = np.full(num_verts, 0.8, dtype=np.float64)
    c2D = np.full(num_verts, 0.9, dtype=np.float64)

    guidance = fuse_and_package_guidance(
        verts, faces, cues_3d, lifted_I2D, c2D,
        config=ImportanceConfig(beta=1.5, smoothing_iterations=2)
    )

    assert len(guidance.importance) == num_verts
    assert np.all(guidance.importance >= 0.0)
    assert np.all(guidance.importance <= 1.0)
    assert guidance.sizing_field is not None
    assert len(guidance.sizing_field) == num_verts

    # High importance vertices must have smaller target edge lengths in sizing field
    high_imp_idx = np.argmax(guidance.importance)
    low_imp_idx = np.argmin(guidance.importance)
    assert guidance.sizing_field[high_imp_idx] < guidance.sizing_field[low_imp_idx]
