from pathlib import Path
from typing import Optional, Union, List, Tuple
import cv2
import numpy as np

from importance.io_utils import Camera, save_mesh


def scalar_to_colormap(
    values: np.ndarray,
    cmap_name: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Map a 1D scalar array of length V to (V, 3) RGB uint8 colors [0, 255].

    Supported colormaps: 'turbo', 'viridis', 'plasma', 'magma', 'inferno', 'jet'.
    """
    vals = np.asarray(values, dtype=np.float64).ravel()
    if vmin is None:
        vmin = float(np.nanmin(vals)) if len(vals) > 0 else 0.0
    if vmax is None:
        vmax = float(np.nanmax(vals)) if len(vals) > 0 else 1.0

    if np.isclose(vmin, vmax):
        normalized = np.zeros_like(vals)
    else:
        normalized = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)

    # OpenCV colormaps (input is uint8 0-255, output is BGR)
    cmap_lookup = {
        "turbo": cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET,
        "plasma": cv2.COLORMAP_PLASMA if hasattr(cv2, "COLORMAP_PLASMA") else cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA if hasattr(cv2, "COLORMAP_MAGMA") else cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO if hasattr(cv2, "COLORMAP_INFERNO") else cv2.COLORMAP_HOT,
        "jet": cv2.COLORMAP_JET,
        "bone": cv2.COLORMAP_BONE,
    }

    cv_cmap = cmap_lookup.get(cmap_name.lower(), cv2.COLORMAP_JET)
    lut_input = (normalized * 255.0).astype(np.uint8).reshape(-1, 1)
    bgr_colors = cv2.applyColorMap(lut_input, cv_cmap).reshape(-1, 3)
    rgb_colors = cv2.cvtColor(bgr_colors.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    return rgb_colors


def export_colored_mesh(
    output_path: Union[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    scalar_values: np.ndarray,
    cmap_name: str = "turbo",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Color mesh by a per-vertex scalar array and save as a colored PLY/OBJ."""
    rgb = scalar_to_colormap(scalar_values, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
    save_mesh(output_path, vertices, faces, vertex_colors=rgb)


def create_projection_overlay(
    image: np.ndarray,
    points: np.ndarray,
    camera: Camera,
    scalars: Optional[np.ndarray] = None,
    point_radius: int = 3,
    alpha: float = 0.7,
    cmap_name: str = "turbo",
    draw_wireframe: bool = False,
    faces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Create a 2D debug overlay by projecting 3D points or wireframe onto a camera image.

    Args:
        image: RGB or grayscale image (H, W, 3) or (H, W).
        points: (N, 3) world coordinates.
        camera: Camera model.
        scalars: Optional (N,) scalar values to color projected points.
        point_radius: Pixel radius for drawing projected points.
        alpha: Blend weight between base image and overlay points.
        cmap_name: Colormap name if scalars are supplied.
        draw_wireframe: If True and faces supplied, draws edges.
        faces: (F, 3) mesh faces.

    Returns:
        RGB image (H, W, 3) with overlay.
    """
    if image.ndim == 2:
        base = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        base = image.copy()

    overlay = base.copy()
    uv, depth = camera.project(points)
    visible = camera.is_visible(uv, depth)

    if scalars is not None:
        colors = scalar_to_colormap(scalars, cmap_name=cmap_name)
    else:
        # Default green points
        colors = np.tile([0, 255, 0], (len(points), 1)).astype(np.uint8)

    # Optional wireframe projection
    if draw_wireframe and faces is not None:
        edges = set()
        for f in faces:
            edges.add(tuple(sorted((f[0], f[1]))))
            edges.add(tuple(sorted((f[1], f[2]))))
            edges.add(tuple(sorted((f[2], f[0]))))

        for v0, v1 in edges:
            if visible[v0] and visible[v1]:
                pt0 = (int(round(uv[v0, 0])), int(round(uv[v0, 1])))
                pt1 = (int(round(uv[v1, 0])), int(round(uv[v1, 1])))
                cv2.line(overlay, pt0, pt1, (180, 180, 180), 1, cv2.LINE_AA)

    # Draw points
    for idx in np.flatnonzero(visible):
        pt = (int(round(uv[idx, 0])), int(round(uv[idx, 1])))
        c = tuple(int(x) for x in colors[idx])
        cv2.circle(overlay, pt, point_radius, c, -1, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0.0)
    return blended


def draw_polylines_on_image(
    image: np.ndarray,
    polylines: List[np.ndarray],
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a list of 2D polylines (lists of (N, 2) pixel coords) on an image."""
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

    for poly in polylines:
        pts = np.asarray(poly, dtype=np.int32).reshape((-1, 1, 2))
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    return canvas
