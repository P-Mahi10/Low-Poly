import json
from pathlib import Path

import cv2
import numpy as np

from config import (CO3D_CONSISTENCY_VIEWS, CO3D_DEPTH_SAMPLE_STRIDE,
                    CO3D_DEPTH_TOLERANCE, CO3D_FOREGROUND_THRESHOLD,
                    CO3D_VOXEL_SIZE, OUTPUT_DIR, PER_IMAGE_OUTPUT_DIR)
from point_cloud import co3d_camera_matrix, polish_point_cloud, save_ply, voxel_downsample


def load_depth_frame(dataset_root, record):
    image_path = dataset_root / "images" / Path(record["image"]["path"]).name
    mask_path = dataset_root / "masks" / Path(record["mask"]["path"]).name
    depth_path = dataset_root / "depths" / Path(record["depth"]["path"]).name
    valid_path = dataset_root / "depth_masks" / Path(record["depth"]["mask_path"]).name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    valid_depth = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
    if any(item is None for item in (image, mask, raw_depth, valid_depth)):
        return None
    depth = raw_depth.astype(np.uint16).view(np.float16).astype(np.float32)
    depth *= float(record["depth"]["scale_adjustment"])
    foreground = ((mask >= CO3D_FOREGROUND_THRESHOLD) & (valid_depth > 0)
                  & np.isfinite(depth) & (depth > 0))
    if np.any(foreground):
        low, high = np.percentile(depth[foreground], (0.5, 99.5))
        foreground &= (depth >= low) & (depth <= high)
        fill = float(np.median(depth[foreground]))
        filtered = depth.copy()
        filtered[~foreground] = fill
        depth[foreground] = cv2.bilateralFilter(filtered, 7, 0.15, 3)[foreground]
    valid_depth = np.where(foreground, 255, 0).astype(np.uint8)
    rotation = np.asarray(record["viewpoint"]["R"], dtype=np.float64)
    translation = np.asarray(record["viewpoint"]["T"], dtype=np.float64)
    return {"image": image, "mask": mask, "depth": depth, "valid": valid_depth,
            "camera": co3d_camera_matrix(record), "rotation": rotation,
            "translation": translation, "center": -rotation @ translation,
            "name": image_path.name}


def depth_to_camera_points(frame):
    image, mask = frame["image"], frame["mask"]
    height, width = image.shape[:2]
    y, x = np.mgrid[0:height:CO3D_DEPTH_SAMPLE_STRIDE,
                     0:width:CO3D_DEPTH_SAMPLE_STRIDE]
    valid = ((mask[y, x] >= CO3D_FOREGROUND_THRESHOLD)
             & (frame["valid"][y, x] > 0) & (frame["depth"][y, x] > 0))
    u, v = x[valid].astype(np.float64), y[valid].astype(np.float64)
    z = frame["depth"][y[valid], x[valid]].astype(np.float64)
    camera = frame["camera"]
    points = np.column_stack((-(u - camera[0, 2]) * z / camera[0, 0],
                              -(v - camera[1, 2]) * z / camera[1, 1], z))
    return points, image[y[valid], x[valid]]


def export_per_image_clouds(dataset_root, sequence_name, records):
    PER_IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for record in records:
        frame = load_depth_frame(dataset_root, record)
        if frame is None:
            continue
        points, colors = depth_to_camera_points(frame)
        if len(points):
            save_ply(PER_IMAGE_OUTPUT_DIR / f"{Path(frame['name']).stem}.ply", points, colors)
            saved += 1
    print(f"Using CO3D sequence {sequence_name}; saved {saved} per-image point clouds.")


def project_to_frame(points, frame):
    camera_points = points @ frame["rotation"] + frame["translation"]
    z = camera_points[:, 2]
    camera = frame["camera"]
    u = camera[0, 2] - camera[0, 0] * camera_points[:, 0] / np.maximum(z, 1e-8)
    v = camera[1, 2] - camera[1, 1] * camera_points[:, 1] / np.maximum(z, 1e-8)
    return u, v, z


def nearest_view_sets(frames):
    centers = np.vstack([frame["center"] for frame in frames])
    nearby = []
    for index, center in enumerate(centers):
        order = np.argsort(np.linalg.norm(centers - center, axis=1))
        nearby.append([item for item in order if item != index][:CO3D_CONSISTENCY_VIEWS])
    return nearby


def filter_consistent_points(points, source_index, frames, nearby_indices):
    checks = np.zeros(len(points), dtype=np.int16)
    matches = np.zeros(len(points), dtype=np.int16)
    for index in nearby_indices[source_index]:
        target = frames[index]
        u, v, projected_depth = project_to_frame(points, target)
        x = np.rint(u).astype(np.int32)
        y = np.rint(v).astype(np.int32)
        inside = ((projected_depth > 0) & (x >= 0) & (y >= 0)
                  & (x < target["image"].shape[1]) & (y < target["image"].shape[0]))
        if not np.any(inside):
            continue
        chosen = np.flatnonzero(inside)
        depth = target["depth"][y[chosen], x[chosen]]
        visible = ((target["mask"][y[chosen], x[chosen]] >= CO3D_FOREGROUND_THRESHOLD)
                   & (target["valid"][y[chosen], x[chosen]] > 0) & (depth > 0))
        checks[chosen] += visible
        matches[chosen] += visible & (np.abs(projected_depth[chosen] - depth)
                                      <= CO3D_DEPTH_TOLERANCE * depth)
    required = np.where(checks >= 2, 2, checks)
    return (checks > 0) & (matches >= required)


def reconstruct_dense_cloud(dataset_root, sequence_name, records):
    loaded = [load_depth_frame(dataset_root, record) for record in records]
    available = [(record, frame) for record, frame in zip(records, loaded) if frame is not None]
    if not available:
        raise RuntimeError("No valid depth maps were available.")
    records, frames = zip(*available)
    nearby = nearest_view_sets(frames)
    point_sets, color_sets = [], []
    for index, frame in enumerate(frames):
        camera_points, colors = depth_to_camera_points(frame)
        if not len(camera_points):
            continue
        world_points = (camera_points - frame["translation"]) @ frame["rotation"].T
        keep = filter_consistent_points(world_points, index, frames, nearby)
        if np.any(keep):
            point_sets.append(world_points[keep])
            color_sets.append(colors[keep])
    if not point_sets:
        raise RuntimeError("No depth points were consistent across nearby views.")
    points, colors = voxel_downsample(np.vstack(point_sets), np.vstack(color_sets), CO3D_VOXEL_SIZE)
    points, colors = polish_point_cloud(points, colors)
    save_ply(OUTPUT_DIR / "subject_point_cloud.ply", points, colors)
    poses = [{"image": Path(record["image"]["path"]).name,
              "rotation_camera_to_world": record["viewpoint"]["R"],
              "camera_center": (-np.asarray(record["viewpoint"]["R"])
                                @ np.asarray(record["viewpoint"]["T"])).tolist()}
             for record in records]
    (OUTPUT_DIR / "camera_poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")
    print(f"Using CO3D sequence {sequence_name}; saved {len(points)} dense points.")
    return points, colors
