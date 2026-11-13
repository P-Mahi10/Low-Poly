import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from config import (CO3D_DEPTH_SAMPLE_STRIDE, CO3D_FOREGROUND_THRESHOLD,
                    CO3D_VOXEL_SIZE, CO3D_CONSISTENCY_VIEWS,
                    CO3D_DEPTH_TOLERANCE, OUTPUT_DIR, PER_IMAGE_OUTPUT_DIR)


def save_ply(path, points, colors):
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors):
            file.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                       f"{int(color[2])} {int(color[1])} {int(color[0])}\n")


def find_co3d_sequence(dataset_root):
    annotation_path = dataset_root / "frame_annotations.jgz"
    mask_folder = dataset_root / "masks"
    if not annotation_path.exists() or not mask_folder.is_dir():
        return None
    with gzip.open(annotation_path, "rt", encoding="utf-8") as file:
        annotations = json.load(file)
    local_mass = {}
    for mask_path in mask_folder.glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            local_mass[mask_path.name] = float(mask.sum()) / 255.0
    groups = {}
    for record in annotations:
        name = Path(record["mask"]["path"]).name
        if name in local_mass:
            groups.setdefault(record["sequence_name"], []).append(record)
    candidates = []
    for name, records in groups.items():
        errors = [abs(local_mass[Path(record["mask"]["path"]).name]
                      - record["mask"]["mass"]) for record in records]
        if len(records) >= 3:
            candidates.append((float(np.median(errors)), name, records))
    if not candidates:
        return None
    error, name, records = min(candidates, key=lambda item: item[0])
    return (name, sorted(records, key=lambda record: record["frame_number"])) if error <= 2.0 else None


def co3d_camera_matrix(record):
    height, width = record["image"]["size"]
    focal_x, focal_y = record["viewpoint"]["focal_length"]
    point_x, point_y = record["viewpoint"]["principal_point"]
    half_width, half_height = width / 2.0, height / 2.0
    scale = min(half_width, half_height)
    return np.array([[focal_x * scale, 0.0, half_width - point_x * scale],
                     [0.0, focal_y * scale, half_height - point_y * scale],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def voxel_downsample(points, colors, voxel_size):
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    count = np.bincount(inverse)
    merged_points = np.zeros((len(count), 3), dtype=np.float64)
    merged_colors = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(merged_points, inverse, points)
    np.add.at(merged_colors, inverse, colors)
    return merged_points / count[:, None], np.rint(merged_colors / count[:, None]).astype(np.uint8)


def load_co3d_frame(dataset_root, record):
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
    rotation = np.asarray(record["viewpoint"]["R"], dtype=np.float64)
    translation = np.asarray(record["viewpoint"]["T"], dtype=np.float64)
    return {"image": image, "mask": mask, "depth": depth, "valid": valid_depth,
            "camera": co3d_camera_matrix(record), "rotation": rotation,
            "translation": translation, "center": -rotation @ translation,
            "name": image_path.name}


def project_to_frame(points, frame):
    camera_points = points @ frame["rotation"] + frame["translation"]
    z = camera_points[:, 2]
    camera = frame["camera"]
    u = camera[0, 2] - camera[0, 0] * camera_points[:, 0] / np.maximum(z, 1e-8)
    v = camera[1, 2] - camera[1, 1] * camera_points[:, 1] / np.maximum(z, 1e-8)
    return u, v, z


def frame_point_cloud(frame):
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


def reconstruct_co3d_per_image(dataset_root, sequence_name, records):
    PER_IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0
    for record in records:
        frame = load_co3d_frame(dataset_root, record)
        if frame is None:
            continue
        points, colors = frame_point_cloud(frame)
        if not len(points):
            continue
        save_ply(PER_IMAGE_OUTPUT_DIR / f"{Path(frame['name']).stem}.ply", points, colors)
        saved += 1
    print(f"Using CO3D sequence {sequence_name}; saved {saved} per-image point clouds.")


def filter_consistent_points(points, source_index, frames, nearby_indices):
    checks = np.zeros(len(points), dtype=np.int16)
    matches = np.zeros(len(points), dtype=np.int16)
    for index in nearby_indices[source_index]:
        target = frames[index]
        u, v, projected_depth = project_to_frame(points, target)
        x = np.rint(u).astype(np.int32)
        y = np.rint(v).astype(np.int32)
        inside = ((projected_depth > 0) & (x >= 0) & (x < target["image"].shape[1])
                  & (y >= 0) & (y < target["image"].shape[0]))
        if not np.any(inside):
            continue
        chosen = np.flatnonzero(inside)
        depth = target["depth"][y[chosen], x[chosen]]
        visible = ((target["mask"][y[chosen], x[chosen]] >= CO3D_FOREGROUND_THRESHOLD)
                   & (target["valid"][y[chosen], x[chosen]] > 0) & (depth > 0))
        checks[chosen] += visible
        matched = visible & (np.abs(projected_depth[chosen] - depth)
                             <= CO3D_DEPTH_TOLERANCE * depth)
        matches[chosen] += matched
    return matches >= np.maximum(1, checks)


def reconstruct_co3d(dataset_root, sequence_name, records):
    all_points, all_colors = [], []
    frames = [load_co3d_frame(dataset_root, record) for record in records]
    available = [(record, frame) for record, frame in zip(records, frames) if frame is not None]
    records, frames = zip(*available)
    centers = np.vstack([frame["center"] for frame in frames])
    nearby_indices = []
    for index, center in enumerate(centers):
        distance = np.linalg.norm(centers - center, axis=1)
        order = np.argsort(distance)
        nearby_indices.append([item for item in order if item != index][:CO3D_CONSISTENCY_VIEWS])
    for index, (record, frame) in enumerate(zip(records, frames)):
        image, mask = frame["image"], frame["mask"]
        height, width = image.shape[:2]
        y, x = np.mgrid[0:height:CO3D_DEPTH_SAMPLE_STRIDE,
                         0:width:CO3D_DEPTH_SAMPLE_STRIDE]
        valid = ((mask[y, x] >= CO3D_FOREGROUND_THRESHOLD)
                 & (frame["valid"][y, x] > 0) & (frame["depth"][y, x] > 0))
        if not np.any(valid):
            continue
        u, v = x[valid].astype(np.float64), y[valid].astype(np.float64)
        z = frame["depth"][y[valid], x[valid]].astype(np.float64)
        camera = frame["camera"]
        camera_points = np.column_stack((-(u - camera[0, 2]) * z / camera[0, 0],
                                         -(v - camera[1, 2]) * z / camera[1, 1], z))
        world_points = (camera_points - frame["translation"]) @ frame["rotation"].T
        consistent = filter_consistent_points(world_points, index, frames, nearby_indices)
        all_points.append(world_points[consistent])
        all_colors.append(image[y[valid], x[valid]][consistent])
    if not all_points:
        raise RuntimeError("No valid CO3D depth points were available.")
    points, colors = voxel_downsample(np.vstack(all_points), np.vstack(all_colors),
                                      CO3D_VOXEL_SIZE)
    save_ply(OUTPUT_DIR / "subject_point_cloud.ply", points, colors)
    poses = [{"image": Path(record["image"]["path"]).name,
              "rotation_camera_to_world": record["viewpoint"]["R"],
              "camera_center": (-np.asarray(record["viewpoint"]["R"])
                                @ np.asarray(record["viewpoint"]["T"])).tolist()}
             for record in records]
    (OUTPUT_DIR / "camera_poses.json").write_text(json.dumps(poses, indent=2), encoding="utf-8")
    print(f"Using CO3D sequence {sequence_name}; saved {len(points)} dense points.")
