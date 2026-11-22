import cv2
import numpy as np
from scipy.optimize import least_squares

from config import (CAMERA_MATRIX_FILE, KNOWN_POSE_PAIR_OFFSETS,
                    KNOWN_POSE_REPROJECTION_ERROR_PX, MAX_DEPTH, MAX_FEATURES,
                    MIN_MATCHES, RANSAC_THRESHOLD_PX, RATIO_THRESHOLD)
from point_cloud import co3d_camera_matrix, voxel_downsample


def create_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=MAX_FEATURES), cv2.NORM_L2
    return cv2.ORB_create(nfeatures=MAX_FEATURES), cv2.NORM_HAMMING


def extract_features(images, detector):
    for image in images:
        image["keypoints"], image["descriptors"] = detector.detectAndCompute(
            image["gray"], image["mask"])


def match_features(descriptors_a, descriptors_b, norm):
    if (descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2
            or len(descriptors_b) < 2):
        return []
    matcher = cv2.BFMatcher(norm)
    forward = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    reverse = matcher.knnMatch(descriptors_b, descriptors_a, k=2)
    accepted = {pair[0].queryIdx: pair[0] for pair in forward
                if len(pair) == 2 and pair[0].distance < RATIO_THRESHOLD * pair[1].distance}
    backward = {pair[0].queryIdx: pair[0].trainIdx for pair in reverse
                if len(pair) == 2 and pair[0].distance < RATIO_THRESHOLD * pair[1].distance}
    return [match for query, match in accepted.items() if backward.get(match.trainIdx) == query]


def load_camera_matrix(shape):
    height, width = shape[:2]
    if CAMERA_MATRIX_FILE.exists():
        matrix = np.load(CAMERA_MATRIX_FILE).astype(np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix.npy must contain a 3x3 matrix")
        return matrix
    focal = float(max(width, height))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def refine_pair_pose(points_a, points_b, rotation, translation):
    if len(points_a) < 8:
        return rotation, translation
    initial_rotation, _ = cv2.Rodrigues(rotation)
    scale = max(float(np.linalg.norm(translation)), 1e-8)
    initial = np.concatenate((initial_rotation.ravel(), translation.ravel()))

    def residual(parameters):
        refined_rotation, _ = cv2.Rodrigues(parameters[:3])
        direction = parameters[3:]
        refined_translation = direction * scale / max(np.linalg.norm(direction), 1e-8)
        first_projection = np.hstack((np.eye(3), np.zeros((3, 1))))
        second_projection = np.hstack((refined_rotation, refined_translation[:, None]))
        points = cv2.triangulatePoints(first_projection, second_projection,
                                      points_a.T, points_b.T)
        points = (points[:3] / points[3]).T
        first = points[:, :2] / points[:, 2:3]
        transformed = points @ refined_rotation.T + refined_translation
        second = transformed[:, :2] / transformed[:, 2:3]
        return np.concatenate(((first - points_a).ravel(), (second - points_b).ravel()))

    try:
        result = least_squares(residual, initial, loss="soft_l1", max_nfev=100)
    except (ValueError, np.linalg.LinAlgError):
        return rotation, translation
    refined_rotation, _ = cv2.Rodrigues(result.x[:3])
    direction = result.x[3:]
    refined_translation = direction * scale / max(np.linalg.norm(direction), 1e-8)
    return refined_rotation, refined_translation


def estimate_pair(first, second, matches, camera_a, camera_b):
    if len(matches) < MIN_MATCHES:
        return None
    points_a = np.float64([first["keypoints"][match.queryIdx].pt for match in matches])
    points_b = np.float64([second["keypoints"][match.trainIdx].pt for match in matches])
    normal_a = cv2.undistortPoints(points_a.reshape(-1, 1, 2), camera_a, None).reshape(-1, 2)
    normal_b = cv2.undistortPoints(points_b.reshape(-1, 1, 2), camera_b, None).reshape(-1, 2)
    focal = np.mean([camera_a[0, 0], camera_a[1, 1], camera_b[0, 0], camera_b[1, 1]])
    essential, ransac = cv2.findEssentialMat(normal_a, normal_b, np.eye(3), cv2.RANSAC,
                                              0.999, RANSAC_THRESHOLD_PX / focal)
    if essential is None or ransac is None or essential.shape != (3, 3):
        return None
    count, rotation, translation, keep = cv2.recoverPose(essential, normal_a, normal_b,
                                                          np.eye(3), mask=ransac.copy())
    if count < MIN_MATCHES:
        return None
    selected = keep.ravel() != 0
    rotation, translation = refine_pair_pose(normal_a[selected], normal_b[selected],
                                             rotation, translation.reshape(3))
    return rotation, translation, points_a[selected], points_b[selected]


def reconstruct_sparse(images):
    intrinsics = [load_camera_matrix(record["image"].shape) for record in images]
    detector, norm = create_detector()
    extract_features(images, detector)
    rotation, center = np.eye(3), np.zeros(3)
    clouds, colors = [], []
    for index in range(len(images) - 1):
        matches = match_features(images[index]["descriptors"], images[index + 1]["descriptors"], norm)
        result = estimate_pair(images[index], images[index + 1], matches,
                               intrinsics[index], intrinsics[index + 1])
        if result is None:
            continue
        relative_rotation, translation, points_a, points_b = result
        next_rotation = rotation @ relative_rotation.T
        next_center = center - next_rotation @ translation
        world_to_a, world_to_b = rotation.T, next_rotation.T
        project_a = intrinsics[index] @ np.hstack((world_to_a, -world_to_a @ center[:, None]))
        project_b = intrinsics[index + 1] @ np.hstack((world_to_b, -world_to_b @ next_center[:, None]))
        homogeneous = cv2.triangulatePoints(project_a, project_b, points_a.T, points_b.T)
        valid = np.abs(homogeneous[3]) > 1e-10
        points = (homogeneous[:3, valid] / homogeneous[3, valid]).T
        points = points[np.isfinite(points).all(axis=1) & (np.abs(points[:, 2]) < MAX_DEPTH)]
        if len(points):
            clouds.append(points)
            colors.append(np.tile(np.mean(images[index]["image"], axis=(0, 1)).astype(np.uint8),
                                  (len(points), 1)))
        rotation, center = next_rotation, next_center
    if not clouds:
        raise RuntimeError("No stable sparse points were reconstructed.")
    return np.vstack(clouds), np.vstack(colors)


def co3d_projection(record):
    camera = co3d_camera_matrix(record).copy()
    camera[0, 0] *= -1.0
    camera[1, 1] *= -1.0
    rotation = np.asarray(record["viewpoint"]["R"], dtype=np.float64)
    translation = np.asarray(record["viewpoint"]["T"], dtype=np.float64)
    return camera @ np.hstack((rotation.T, translation[:, None])), rotation, translation


def reconstruct_with_known_poses(images, records):
    records_by_name = {record["image"]["path"].rsplit("/", 1)[-1]: record for record in records}
    entries = [(image, records_by_name[image["path"].name]) for image in images
               if image["path"].name in records_by_name]
    if len(entries) < 2:
        raise RuntimeError("CO3D camera annotations do not match the image files.")
    detector, norm = create_detector()
    extract_features([image for image, _ in entries], detector)
    projections = [co3d_projection(record) for _, record in entries]
    point_sets, color_sets = [], []
    successful_pairs = 0
    for first_index in range(len(entries)):
        for offset in KNOWN_POSE_PAIR_OFFSETS:
            second_index = first_index + offset
            if second_index >= len(entries):
                continue
            first, _ = entries[first_index]
            second, _ = entries[second_index]
            matches = match_features(first["descriptors"], second["descriptors"], norm)
            if len(matches) < MIN_MATCHES:
                continue
            points_a = np.float64([first["keypoints"][match.queryIdx].pt for match in matches])
            points_b = np.float64([second["keypoints"][match.trainIdx].pt for match in matches])
            homogeneous = cv2.triangulatePoints(projections[first_index][0],
                                                projections[second_index][0], points_a.T, points_b.T)
            valid = np.abs(homogeneous[3]) > 1e-10
            world = np.full((len(matches), 3), np.nan, dtype=np.float64)
            world[valid] = (homogeneous[:3, valid] / homogeneous[3, valid]).T
            errors = []
            for projection, observed in ((projections[first_index][0], points_a),
                                         (projections[second_index][0], points_b)):
                projected = (projection @ np.column_stack((world, np.ones(len(world)))).T).T
                errors.append(np.linalg.norm(projected[:, :2] / projected[:, 2:3] - observed, axis=1))
            rotation_a, translation_a = projections[first_index][1:]
            rotation_b, translation_b = projections[second_index][1:]
            depth_a = (world @ rotation_a + translation_a)[:, 2]
            depth_b = (world @ rotation_b + translation_b)[:, 2]
            keep = (np.isfinite(world).all(axis=1) & (depth_a > 0) & (depth_b > 0)
                    & (errors[0] <= KNOWN_POSE_REPROJECTION_ERROR_PX)
                    & (errors[1] <= KNOWN_POSE_REPROJECTION_ERROR_PX))
            if np.any(keep):
                point_sets.append(world[keep])
                pixels = np.rint(points_a[keep]).astype(int)
                pixels[:, 0] = np.clip(pixels[:, 0], 0, first["image"].shape[1] - 1)
                pixels[:, 1] = np.clip(pixels[:, 1], 0, first["image"].shape[0] - 1)
                color_sets.append(first["image"][pixels[:, 1], pixels[:, 0]])
                successful_pairs += 1
    if not point_sets:
        raise RuntimeError("No feature matches survived known-pose triangulation.")
    points, colors = voxel_downsample(np.vstack(point_sets), np.vstack(color_sets), 0.04)
    print(f"Used {len(entries)} images and {successful_pairs} matched image pairs.")
    return points, colors
