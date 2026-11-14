from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from importance.io_utils import (
    Camera,
    create_camera_opencv,
    create_camera_from_co3d_record,
    load_mesh,
    save_mesh,
    create_synthetic_sphere,
    create_synthetic_cube,
    create_synthetic_torus,
    evaluate_points_inside_mask,
)
from importance.viz import (
    scalar_to_colormap,
    export_colored_mesh,
    create_projection_overlay,
    draw_polylines_on_image,
)


def test_synthetic_meshes_creation_and_io():
    """Verify sphere, cube, and torus mesh creation and save/load roundtripping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        meshes = [
            ("sphere", create_synthetic_sphere(radius=1.0, subdivisions=2)),
            ("cube", create_synthetic_cube(side=1.5)),
            ("torus", create_synthetic_torus(major_radius=1.0, minor_radius=0.3)),
        ]

        for name, (verts, faces) in meshes:
            assert len(verts) > 0
            assert len(faces) > 0
            assert verts.shape[1] == 3
            assert faces.shape[1] == 3
            assert np.all(np.isfinite(verts))

            for ext in [".ply", ".obj"]:
                file_path = tmp_path / f"{name}{ext}"
                save_mesh(file_path, verts, faces)
                assert file_path.exists()

                loaded_v, loaded_f = load_mesh(file_path)
                assert len(loaded_v) == len(verts)
                assert len(loaded_f) == len(faces)
                assert np.allclose(loaded_v, verts, atol=1e-5)


def test_camera_known_point_projection():
    """Project known 3D point and verify pixel coordinates against manual calculation and cv2.projectPoints."""
    width, height = 640, 480
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Place camera at (0, 0, 5) looking down -Z axis towards the origin
    # In OpenCV camera frame: +Z points forward.
    # So if camera center C = (0, 0, 5) and looks towards (0, 0, 0),
    # then looking direction is (0, 0, -1) in world coords.
    # To map world forward (0, 0, -1) to camera +Z (0, 0, 1):
    # R = diag([1, -1, -1]) or similar.
    # Let's verify standard world-to-cam translation:
    # If camera is at (0, 0, 5) with identity rotation looking towards +Z:
    # t = [0, 0, -5]
    # Then point at (0, 0, 10) in world coords has X_cam = (0, 0, 5), which is in front.
    R = np.eye(3, dtype=np.float64)
    t = np.array([0.0, 0.0, -5.0], dtype=np.float64)
    cam = create_camera_opencv(K, R, t, width=width, height=height)

    # Point at (0, 0, 10) in world -> camera coords (0, 0, 5)
    test_pt = np.array([[0.0, 0.0, 10.0]])
    uv, depth = cam.project(test_pt)

    assert np.isclose(depth[0], 5.0)
    assert np.isclose(uv[0, 0], cx)
    assert np.isclose(uv[0, 1], cy)
    assert cam.is_visible(uv, depth)[0]

    # Verify equivalence with OpenCV native projectPoints
    rvec, _ = cv2.Rodrigues(R)
    cv2_pts, _ = cv2.projectPoints(test_pt, rvec, t, K, None)
    assert np.allclose(uv, cv2_pts.reshape(-1, 2))

    # Point with non-zero X and Y:
    # X_world = 1.0, Y_world = -0.5, Z_world = 10.0
    # X_cam = 1.0, Y_cam = -0.5, Z_cam = 5.0
    # u = 500 * (1.0 / 5.0) + 320 = 100 + 320 = 420.0
    # v = 500 * (-0.5 / 5.0) + 240 = -50 + 240 = 190.0
    test_pt2 = np.array([[1.0, -0.5, 10.0]])
    uv2, depth2 = cam.project(test_pt2)
    assert np.isclose(uv2[0, 0], 420.0)
    assert np.isclose(uv2[0, 1], 190.0)

    # Point behind camera (Z_world = 0 -> Z_cam = -5) should not be visible
    pt_behind = np.array([[0.0, 0.0, 0.0]])
    uv_b, depth_b = cam.project(pt_behind)
    assert depth_b[0] < 0
    assert not cam.is_visible(uv_b, depth_b)[0]


def test_debug_overlay_and_wireframe():
    """Verify projection overlay creation and pixel drawing."""
    width, height = 320, 240
    K = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    cam = create_camera_opencv(K, R, t, width=width, height=height)

    image = np.zeros((height, width, 3), dtype=np.uint8)
    verts, faces = create_synthetic_cube(side=1.0)
    # Shift cube forward to world Z = 4.0
    verts_shifted = verts + np.array([0.0, 0.0, 4.0])

    overlay = create_projection_overlay(
        image=image,
        points=verts_shifted,
        camera=cam,
        scalars=verts_shifted[:, 2],
        point_radius=4,
        draw_wireframe=True,
        faces=faces,
    )

    assert overlay.shape == (height, width, 3)
    # Ensure drawing changed pixels from initial black
    assert np.count_nonzero(overlay) > 100


def test_co3d_camera_conversion():
    """Verify CO3D record loader faithfully converts PyTorch3D conventions to OpenCV."""
    # Synthetic CO3D record
    record = {
        "image": {"path": "images/test_001.jpg", "size": [480, 640]},
        "viewpoint": {
            "focal_length": [2.0, 2.0],
            "principal_point": [0.0, 0.0],
            "R": np.eye(3).tolist(),
            "T": [0.0, 0.0, 3.0],
        },
    }

    cam = create_camera_from_co3d_record(record)
    assert cam.width == 640
    assert cam.height == 480
    assert cam.image_name == "test_001.jpg"

    # In CO3D, scale = min(640, 480)/2 = 240.
    # focal = 2.0 * 240 = 480.0
    # principal_point = (320, 240)
    assert np.isclose(cam.K[0, 0], 480.0)
    assert np.isclose(cam.K[1, 1], 480.0)
    assert np.isclose(cam.K[0, 2], 320.0)
    assert np.isclose(cam.K[1, 2], 240.0)


def test_vertex_coloring_export_and_colormaps():
    """Verify colormapping and colored mesh export on synthetic meshes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        verts, faces = create_synthetic_sphere(radius=1.0, subdivisions=2)

        # Generate non-trivial scalar field (radial distance from (0, 0, 1))
        scalars = np.linalg.norm(verts - np.array([0.0, 0.0, 1.0]), axis=1)

        colors = scalar_to_colormap(scalars, cmap_name="turbo")
        assert colors.shape == (len(verts), 3)
        assert colors.dtype == np.uint8

        out_ply = tmp_path / "colored_sphere.ply"
        export_colored_mesh(out_ply, verts, faces, scalars, cmap_name="viridis")
        assert out_ply.exists()


def test_inside_mask_evaluation():
    """Verify inside-mask fraction calculation for point cloud alignment validation."""
    width, height = 400, 400
    K = np.array([[300.0, 0, 200], [0, 300.0, 200], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    t = np.array([0.0, 0.0, -3.0], dtype=np.float64)
    cam = create_camera_opencv(K, R, t, width=width, height=height)

    # Mask with a white square in the center (100 to 300 in X and Y)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[100:300, 100:300] = 255

    # Points at world Z=3 (camera Z=3 - (-3) -> depth=6, so projection is (200, 200))
    # Point at (0, 0, 6) in world -> camera Z = 3 -> maps to (200, 200), inside mask
    inside_pts = np.array([
        [0.0, 0.0, 6.0],
        [0.1, 0.1, 6.0],
        [-0.1, -0.1, 6.0],
    ])
    score_inside = evaluate_points_inside_mask(inside_pts, cam, mask)
    assert np.isclose(score_inside, 1.0)

    # Points far off to the side -> maps outside mask
    outside_pts = np.array([
        [3.0, 3.0, 6.0],
        [-3.0, -3.0, 6.0],
    ])
    score_outside = evaluate_points_inside_mask(outside_pts, cam, mask)
    assert np.isclose(score_outside, 0.0)
