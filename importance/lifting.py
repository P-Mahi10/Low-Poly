from typing import List, Tuple, Optional
import numpy as np
import cv2

from importance.config import LiftingConfig
from importance.io_utils import Camera


def render_mesh_z_buffer(
    vertices: np.ndarray,
    faces: np.ndarray,
    camera: Camera,
) -> np.ndarray:
    """Rasterize an exact depth map (z-buffer) of the mesh from the camera viewpoint.

    Uses analytical ray-plane depth interpolation per triangle.

    Returns:
        z_buffer: (H, W) float64 array with infinity for background pixels.
    """
    h, w = camera.height, camera.width
    z_buffer = np.full((h, w), np.inf, dtype=np.float64)

    # Transform all vertices into camera frame
    pts_cam = camera.world_to_camera(vertices)
    depths = pts_cam[:, 2]

    uv, _ = camera.project(vertices)
    vis = camera.is_visible(uv, depths)

    fx = camera.K[0, 0]
    fy = camera.K[1, 1]
    cx = camera.K[0, 2]
    cy = camera.K[1, 2]

    for f in faces:
        if not np.any(vis[f]):
            continue

        p0 = pts_cam[f[0]]
        p1 = pts_cam[f[1]]
        p2 = pts_cam[f[2]]

        # Normal in camera coordinates
        n_cam = np.cross(p1 - p0, p2 - p0)
        # Cull back-facing triangles from camera point of view
        # Triangle faces camera when dot(p0, n_cam) < 0
        if np.dot(p0, n_cam) >= 0:
            continue

        d_plane = float(np.dot(n_cam, p0))
        f_pts = uv[f]

        # 2D bounding box
        min_u = int(np.clip(np.floor(np.min(f_pts[:, 0])), 0, w - 1))
        max_u = int(np.clip(np.ceil(np.max(f_pts[:, 0])), 0, w - 1))
        min_v = int(np.clip(np.floor(np.min(f_pts[:, 1])), 0, h - 1))
        max_v = int(np.clip(np.ceil(np.max(f_pts[:, 1])), 0, h - 1))

        if min_u >= max_u or min_v >= max_v:
            continue

        pts_int = np.rint(f_pts).astype(np.int32).reshape((-1, 1, 2))
        tri_mask = np.zeros((max_v - min_v + 1, max_u - min_u + 1), dtype=np.uint8)
        shifted_pts = pts_int - np.array([min_u, min_v])
        cv2.fillPoly(tri_mask, [shifted_pts], 255)

        inside = tri_mask > 0
        if not np.any(inside):
            continue

        # Grid coordinates
        grid_v, grid_u = np.mgrid[min_v : max_v + 1, min_u : max_u + 1]
        rx = (grid_u - cx) / fx
        ry = (grid_v - cy) / fy
        denom = n_cam[0] * rx + n_cam[1] * ry + n_cam[2]

        valid_denom = np.abs(denom) > 1e-8
        z_tri = np.full_like(denom, np.inf, dtype=np.float64)
        z_tri[valid_denom] = d_plane / denom[valid_denom]

        valid_patch = inside & (z_tri > 0)
        patch = z_buffer[min_v : max_v + 1, min_u : max_u + 1]
        patch[valid_patch] = np.minimum(patch[valid_patch], z_tri[valid_patch])

    return z_buffer


def sample_bilinear(
    image_2d: np.ndarray,
    u: float,
    v: float,
) -> float:
    """Bilinear interpolation of scalar image at subpixel coordinates (u, v)."""
    h, w = image_2d.shape[:2]
    if u < 0 or u >= (w - 1) or v < 0 or v >= (h - 1):
        # Nearest boundary fallback
        iu = int(np.clip(round(u), 0, w - 1))
        iv = int(np.clip(round(v), 0, h - 1))
        return float(image_2d[iv, iu])

    u0, v0 = int(np.floor(u)), int(np.floor(v))
    u1, v1 = u0 + 1, v0 + 1
    du, dv = u - u0, v - v0

    val = (
        (1.0 - du) * (1.0 - dv) * image_2d[v0, u0]
        + du * (1.0 - dv) * image_2d[v0, u1]
        + (1.0 - du) * dv * image_2d[v1, u0]
        + du * dv * image_2d[v1, u1]
    )
    return float(val)


def lift_2d_cues_to_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray,
    cameras: List[Camera],
    image_cues_list: List[np.ndarray],
    config: Optional[LiftingConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift multi-view 2D importance maps to 3D mesh vertices.

    Performs:
      - Projection into each calibrated camera view.
      - Z-buffer depth occlusion testing.
      - Grazing angle cosine weighting: w_j = max(0, n . d_view)^2.
      - Multi-view weighted average of sampled I2D.
      - Cross-view variance coherence check to derive per-vertex confidence c2D.

    Returns:
        lifted_I2D: (V,) float64 in [0, 1]
        c2D: (V,) float64 in [0, 1]
    """
    if config is None:
        config = LiftingConfig()

    num_verts = len(vertices)
    if not cameras or not image_cues_list:
        return np.zeros(num_verts, dtype=np.float64), np.zeros(num_verts, dtype=np.float64)

    weighted_sum = np.zeros(num_verts, dtype=np.float64)
    total_weights = np.zeros(num_verts, dtype=np.float64)

    # Store individual valid samples for cross-view variance calculation
    vertex_samples: List[List[float]] = [[] for _ in range(num_verts)]
    vertex_weights: List[List[float]] = [[] for _ in range(num_verts)]

    for cam, i2d_map in zip(cameras, image_cues_list):
        uv, depths = cam.project(vertices)
        vis_bounds = cam.is_visible(uv, depths, margin=1.0)

        # Build depth buffer for occlusion testing
        z_buf = render_mesh_z_buffer(vertices, faces, cam)
        cam_center = cam.camera_center

        for v_idx in np.flatnonzero(vis_bounds):
            z_pt = depths[v_idx]
            u, v = uv[v_idx, 0], uv[v_idx, 1]
            iu, iv = int(np.clip(round(u), 0, cam.width - 1)), int(np.clip(round(v), 0, cam.height - 1))

            # Occlusion check: compare depth with z-buffer
            min_depth_in_buf = z_buf[iv, iu]
            if z_pt > min_depth_in_buf * (1.0 + config.depth_tolerance_ratio):
                continue  # Occluded by foreground surface

            # Viewing direction from vertex towards camera optical center
            view_dir = cam_center - vertices[v_idx]
            view_dist = np.linalg.norm(view_dir)
            if view_dist < 1e-12:
                continue
            view_dir /= view_dist

            # Grazing angle cos(theta) = dot(normal, view_direction)
            cos_theta = np.dot(vertex_normals[v_idx], view_dir)
            if cos_theta < config.min_grazing_cos:
                continue  # Back-facing or extreme grazing angle

            weight = float(cos_theta ** 2)
            sample = sample_bilinear(i2d_map, u, v)

            weighted_sum[v_idx] += weight * sample
            total_weights[v_idx] += weight
            vertex_samples[v_idx].append(sample)
            vertex_weights[v_idx].append(weight)

    # Multi-view aggregation
    lifted_I2D = np.zeros(num_verts, dtype=np.float64)
    c2D = np.zeros(num_verts, dtype=np.float64)

    for v_idx in range(num_verts):
        w_tot = total_weights[v_idx]
        if w_tot > 1e-6:
            mean_sample = weighted_sum[v_idx] / w_tot
            lifted_I2D[v_idx] = mean_sample

            # Cross-view variance
            samples = vertex_samples[v_idx]
            weights = vertex_weights[v_idx]
            if len(samples) > 1:
                variance = float(np.average((np.array(samples) - mean_sample) ** 2, weights=weights))
            else:
                variance = 0.0

            # Confidence c2D increases with view weight and decreases with disagreement/variance
            weight_factor = float(1.0 - np.exp(-1.5 * w_tot))
            agreement_factor = float(np.exp(-config.variance_penalty * variance))
            c2D[v_idx] = np.clip(weight_factor * agreement_factor, 0.0, 1.0)
        else:
            lifted_I2D[v_idx] = 0.0
            c2D[v_idx] = 0.0

    return lifted_I2D, c2D
