from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import json

import cv2
import numpy as np
import trimesh


@dataclass
class Camera:
    """Camera model using OpenCV coordinate convention.

    Coordinates:
        X_cam points right, Y_cam points down, Z_cam points forward along optical axis.
        World to camera transformation: X_cam = R @ X_world + t
        Projection:
            u = fx * (X_cam / Z_cam) + cx
            v = fy * (Y_cam / Z_cam) + cy
    """
    K: np.ndarray          # 3x3 intrinsic matrix (float64)
    R: np.ndarray          # 3x3 rotation matrix, world-to-cam (float64)
    t: np.ndarray          # 3-vector translation, world-to-cam (float64)
    width: int
    height: int
    image_name: Optional[str] = None
    dist_coeffs: Optional[np.ndarray] = None

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape((3, 3))
        self.R = np.asarray(self.R, dtype=np.float64).reshape((3, 3))
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        self.width = int(self.width)
        self.height = int(self.height)
        if self.dist_coeffs is not None:
            self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64).ravel()

    @property
    def camera_center(self) -> np.ndarray:
        """Position of camera optical center in world coordinates: C = -R^T @ t."""
        return -self.R.T @ self.t

    @property
    def optical_axis(self) -> np.ndarray:
        """Unit vector pointing forward (along +Z_cam) in world coordinates."""
        # Row 2 of R is the world-space direction corresponding to camera Z axis
        axis = self.R[2, :]
        return axis / (np.linalg.norm(axis) + 1e-12)

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """Transform N x 3 points from world coordinates to camera coordinates."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        return (pts @ self.R.T) + self.t

    def camera_to_world(self, points: np.ndarray) -> np.ndarray:
        """Transform N x 3 points from camera coordinates to world coordinates."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        return (pts - self.t) @ self.R

    def project(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project world-space points onto image plane.

        Args:
            points: (N, 3) array of world coordinates.

        Returns:
            uv: (N, 2) pixel coordinates [u, v].
            depth: (N,) Z_cam depth values.
        """
        pts_cam = self.world_to_camera(points)
        z = pts_cam[:, 2]

        # Guard against zero or negative depth for division
        safe_z = np.where(np.abs(z) < 1e-8, 1e-8, z)

        if self.dist_coeffs is None or np.all(self.dist_coeffs == 0):
            u = self.K[0, 0] * (pts_cam[:, 0] / safe_z) + self.K[0, 2]
            v = self.K[1, 1] * (pts_cam[:, 1] / safe_z) + self.K[1, 2]
            uv = np.column_stack((u, v))
        else:
            # OpenCV projectPoints handles distortion coefficients
            rvec, _ = cv2.Rodrigues(self.R)
            projected, _ = cv2.projectPoints(
                points.reshape(-1, 1, 3).astype(np.float64),
                rvec,
                self.t.reshape(3, 1).astype(np.float64),
                self.K,
                self.dist_coeffs,
            )
            uv = projected.reshape(-1, 2)

        return uv, z

    def is_visible(self, uv: np.ndarray, depth: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """Boolean mask indicating which projected points fall within image bounds and in front of camera."""
        return (
            (depth > 0)
            & (uv[:, 0] >= margin)
            & (uv[:, 0] < (self.width - margin))
            & (uv[:, 1] >= margin)
            & (uv[:, 1] < (self.height - margin))
        )


# ---------- Explicit Camera Loaders ----------

def create_camera_opencv(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    width: int,
    height: int,
    image_name: Optional[str] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Camera:
    """Create Camera from native OpenCV parameters (world-to-cam [R|t])."""
    return Camera(
        K=K,
        R=R,
        t=t,
        width=width,
        height=height,
        image_name=image_name,
        dist_coeffs=dist_coeffs,
    )


def create_camera_from_c2w(
    K: np.ndarray,
    c2w_rotation: np.ndarray,
    camera_center: np.ndarray,
    width: int,
    height: int,
    image_name: Optional[str] = None,
) -> Camera:
    """Create Camera from Camera-to-World rotation R_c2w and camera center C.

    R_w2c = R_c2w.T
    t_w2c = -R_c2w.T @ camera_center
    """
    R_c2w = np.asarray(c2w_rotation, dtype=np.float64)
    center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ center
    return Camera(
        K=K,
        R=R_w2c,
        t=t_w2c,
        width=width,
        height=height,
        image_name=image_name,
    )


def create_camera_from_co3d_record(
    record: Dict[str, Any],
    fallback_image_size: Optional[Tuple[int, int]] = None,
) -> Camera:
    """Explicit loader for CO3D frame record into OpenCV camera convention.

    CO3D / PyTorch3D camera conventions:
      - Coordinate system: +X left, +Y up, +Z forward.
      - Extrinsics: X_pt3d = X_world @ R_pt3d + T_pt3d (row-vector convention).
      - Intrinsics: NDC focal length and principal point scaled by min(W, H) / 2.

    OpenCV camera convention:
      - Coordinate system: +X right, +Y down, +Z forward.
      - Extrinsics: X_cv = R_cv @ X_world + t_cv.
      - Transformation: Flip X and Y axes via S = diag([-1, -1, 1]).
        R_cv = S @ R_pt3d.T
        t_cv = S @ T_pt3d
    """
    if "image" in record and "size" in record["image"]:
        height, width = record["image"]["size"]
    elif fallback_image_size is not None:
        height, width = fallback_image_size
    else:
        raise ValueError("Image size must be provided in record or fallback_image_size.")

    focal_x, focal_y = record["viewpoint"]["focal_length"]
    point_x, point_y = record["viewpoint"]["principal_point"]
    half_width, half_height = width / 2.0, height / 2.0
    scale = min(half_width, half_height)

    # In CO3D:
    # u = half_width - point_x * scale - focal_x * scale * (X_pt3d / Z)
    # Since X_cv = -X_pt3d, u = half_width - point_x * scale + focal_x * scale * (X_cv / Z)
    # v = half_height - point_y * scale - focal_y * scale * (Y_pt3d / Z)
    # Since Y_cv = -Y_pt3d, v = half_height - point_y * scale + focal_y * scale * (Y_cv / Z)
    K = np.array([
        [focal_x * scale, 0.0, half_width - point_x * scale],
        [0.0, focal_y * scale, half_height - point_y * scale],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    R_pt3d = np.asarray(record["viewpoint"]["R"], dtype=np.float64)
    T_pt3d = np.asarray(record["viewpoint"]["T"], dtype=np.float64).reshape(3)

    # Convert PyTorch3D row-vector (+X left, +Y up) to OpenCV column-vector (+X right, +Y down)
    flip_xy = np.diag([-1.0, -1.0, 1.0])
    R_cv = flip_xy @ R_pt3d.T
    t_cv = flip_xy @ T_pt3d

    image_name = Path(record["image"]["path"]).name if "image" in record else None
    return Camera(
        K=K,
        R=R_cv,
        t=t_cv,
        width=width,
        height=height,
        image_name=image_name,
    )


def load_cameras_from_json(path: Union[str, Path]) -> List[Camera]:
    """Load camera list from a JSON file.

    Supports:
      1. A list of dicts with keys: 'K', 'R', 't', 'width', 'height', (optional 'image_name').
      2. Output from point_cloud.py: 'rotation_camera_to_world', 'camera_center', 'image'.
         Requires K, width, height or default values.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cameras = []
    for item in data:
        if "K" in item and "R" in item and "t" in item:
            cameras.append(Camera(
                K=np.array(item["K"], dtype=np.float64),
                R=np.array(item["R"], dtype=np.float64),
                t=np.array(item["t"], dtype=np.float64),
                width=int(item["width"]),
                height=int(item["height"]),
                image_name=item.get("image_name") or item.get("image"),
            ))
        elif "rotation_camera_to_world" in item and "camera_center" in item:
            # Poses from point_cloud.py export
            K = np.array(item.get("K", np.eye(3)), dtype=np.float64)
            width = int(item.get("width", 640))
            height = int(item.get("height", 480))
            cameras.append(create_camera_from_c2w(
                K=K,
                c2w_rotation=item["rotation_camera_to_world"],
                camera_center=item["camera_center"],
                width=width,
                height=height,
                image_name=item.get("image"),
            ))
        else:
            raise ValueError(f"Unrecognized camera dictionary format: {item.keys()}")

    return cameras


def evaluate_points_inside_mask(
    points: np.ndarray,
    camera: Camera,
    mask: np.ndarray,
) -> float:
    """Calculate the fraction of visible projected points that land on positive mask pixels.

    Useful for verifying camera convention against subject masks and failing loudly if wrong.
    """
    uv, depth = camera.project(points)
    visible = camera.is_visible(uv, depth)
    if not np.any(visible):
        return 0.0

    valid_uv = np.rint(uv[visible]).astype(np.int32)
    h, w = mask.shape[:2]
    in_bounds = (valid_uv[:, 0] >= 0) & (valid_uv[:, 0] < w) & (valid_uv[:, 1] >= 0) & (valid_uv[:, 1] < h)
    if not np.any(in_bounds):
        return 0.0

    valid_uv = valid_uv[in_bounds]
    inside_count = np.count_nonzero(mask[valid_uv[:, 1], valid_uv[:, 0]] > 0)
    return float(inside_count) / float(len(valid_uv))


# ---------- Mesh I/O ----------

def load_mesh(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load mesh file (OBJ, PLY, OFF) and return (vertices, faces).

    Returns:
        vertices: (V, 3) float64 ndarray
        faces: (F, 3) int64 ndarray
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file does not exist: {path}")

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)])
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Loaded object from {path} is not a Trimesh instance.")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    return vertices, faces


def save_mesh(
    path: Union[str, Path],
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_colors: Optional[np.ndarray] = None,
) -> None:
    """Save mesh with optional per-vertex colors (RGB/RGBA, 0-255 uint8 or 0-1 float)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    colors = None
    if vertex_colors is not None:
        vc = np.asarray(vertex_colors)
        if vc.dtype in (np.float32, np.float64) and vc.max() <= 1.0:
            colors = np.clip(vc * 255.0, 0, 255).astype(np.uint8)
        else:
            colors = np.clip(vc, 0, 255).astype(np.uint8)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=colors,
        process=False,
    )
    mesh.export(str(path))


# ---------- Synthetic Mesh Helpers for Testing ----------

def create_synthetic_sphere(radius: float = 1.0, subdivisions: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Create an icosphere mesh."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def create_synthetic_cube(side: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create a cube mesh with sharp edges and dihedral angles."""
    mesh = trimesh.creation.box(extents=(side, side, side))
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def create_synthetic_torus(major_radius: float = 1.0, minor_radius: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    """Create a torus mesh with mixed positive/negative Gaussian curvature."""
    mesh = trimesh.creation.torus(major_radius=major_radius, minor_radius=minor_radius)
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


# ---------- Image and Mask I/O ----------

def load_image(path: Union[str, Path], color_mode: str = "rgb") -> np.ndarray:
    """Load image from disk as RGB, BGR, or Grayscale ndarray."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image at {path}")

    if color_mode.lower() == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode.lower() == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color_mode.lower() == "bgr":
        return img
    else:
        raise ValueError(f"Unsupported color mode: {color_mode}")


def load_mask(path: Union[str, Path], threshold: int = 127) -> np.ndarray:
    """Load binary silhouette mask as uint8 (0 or 255)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask not found: {path}")

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask at {path}")

    binary = np.where(mask >= threshold, 255, 0).astype(np.uint8)
    return binary
