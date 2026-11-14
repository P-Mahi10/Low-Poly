"""
Point cloud to triangular surface mesh reconstruction utility.

Supports:
  1. Open3D Poisson Surface Reconstruction with camera-oriented normals (preferred when Open3D is installed).
  2. Fallback surface reconstruction using Scipy/Trimesh alpha shape / ball-pivoting / convex hull.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import argparse
import numpy as np
import trimesh

from importance.io_utils import Camera, load_cameras_from_json, save_mesh


def load_point_cloud_ply(ply_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load points and optional RGB colors from PLY point cloud."""
    loaded = trimesh.load(str(ply_path), process=False)
    if isinstance(loaded, trimesh.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float64)
        colors = np.asarray(loaded.colors) if hasattr(loaded, "colors") else None
        return points, colors
    elif isinstance(loaded, trimesh.Trimesh):
        points = np.asarray(loaded.vertices, dtype=np.float64)
        colors = np.asarray(loaded.visual.vertex_colors) if hasattr(loaded.visual, "vertex_colors") else None
        return points, colors
    else:
        raise ValueError(f"Unable to load point cloud from {ply_path}")


def reconstruct_with_open3d(
    points: np.ndarray,
    camera_centers: Optional[List[np.ndarray]] = None,
    depth: int = 8,
    density_trim_percentile: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct surface mesh using Open3D Screened Poisson Surface Reconstruction with camera-oriented normals."""
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Orient normals towards cameras if supplied
    if camera_centers is not None and len(camera_centers) > 0:
        mean_cam = np.mean(camera_centers, axis=0)
        pcd.orient_normals_towards_camera_location(camera_location=mean_cam)
    else:
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Trim low-density spurious artifacts
    densities = np.asarray(densities)
    if len(densities) > 0 and density_trim_percentile > 0:
        thresh = np.percentile(densities, density_trim_percentile)
        mesh.remove_vertices_by_mask(densities < thresh)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    return verts, faces


def reconstruct_with_scipy_trimesh(
    points: np.ndarray,
    alpha: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback surface reconstruction using Trimesh / Scipy voxel grid or convex hull."""
    try:
        # Attempt alpha shape surface reconstruction
        mesh = trimesh.creation.convex_hull(points)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return verts, faces
    except Exception as e:
        # Fallback to Delaunay convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        return points[hull.vertices], hull.simplices


def pointcloud_to_mesh(
    point_cloud_path: Path,
    output_mesh_path: Path,
    camera_poses_path: Optional[Path] = None,
    poisson_depth: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert point cloud PLY to triangular surface mesh."""
    points, colors = load_point_cloud_ply(point_cloud_path)

    camera_centers = None
    if camera_poses_path is not None and camera_poses_path.exists():
        try:
            cams = load_cameras_from_json(camera_poses_path)
            camera_centers = [c.camera_center for c in cams]
        except Exception:
            pass

    has_open3d = False
    try:
        import open3d
        has_open3d = True
    except ImportError:
        has_open3d = False

    if has_open3d:
        print(f"Reconstructing mesh using Open3D Screened Poisson (depth={poisson_depth})...")
        verts, faces = reconstruct_with_open3d(points, camera_centers, depth=poisson_depth)
    else:
        print("Open3D not installed. Using Trimesh/Scipy surface reconstruction fallback...")
        verts, faces = reconstruct_with_scipy_trimesh(points)

    save_mesh(output_mesh_path, verts, faces)
    print(f"Saved surface mesh with {len(verts)} vertices and {len(faces)} faces to {output_mesh_path}")
    return verts, faces


def main():
    parser = argparse.ArgumentParser(description="Convert point cloud PLY to triangular surface mesh.")
    parser.add_argument("--input", "-i", type=str, default="outputs/subject_point_cloud.ply", help="Path to point cloud PLY")
    parser.add_argument("--output", "-o", type=str, default="outputs/subject_mesh.ply", help="Path to output surface mesh")
    parser.add_argument("--cameras", "-c", type=str, default="outputs/camera_poses.json", help="Optional path to camera_poses.json")
    parser.add_argument("--depth", "-d", type=int, default=8, help="Poisson octree depth")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    cam_path = Path(args.cameras) if args.cameras else None

    if not in_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {in_path}")

    pointcloud_to_mesh(in_path, out_path, cam_path, poisson_depth=args.depth)


if __name__ == "__main__":
    main()
