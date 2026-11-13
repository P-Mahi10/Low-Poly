import cv2
import numpy as np

from config import (CAMERA_MATRIX_FILE, MAX_DEPTH, MAX_FEATURES, MIN_MATCHES,
                    RANSAC_THRESHOLD_PX, RATIO_THRESHOLD, SUPPORTED_EXTENSIONS)


def load_images(folder):
    paths = sorted((p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS),
                   key=lambda p: p.name.lower())
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append({"path": path, "image": image,
                           "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)})
    if len(images) < 2:
        raise ValueError(f"At least two readable photos are required in {folder}")
    return images


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


def create_detector():
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=MAX_FEATURES), cv2.NORM_L2
    return cv2.ORB_create(nfeatures=MAX_FEATURES), cv2.NORM_HAMMING


def match_features(desc_a, desc_b, norm):
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return []
    matcher = cv2.BFMatcher(norm)
    forward = matcher.knnMatch(desc_a, desc_b, k=2)
    reverse = matcher.knnMatch(desc_b, desc_a, k=2)
    accepted = {pair[0].queryIdx: pair[0] for pair in forward
                if len(pair) == 2 and pair[0].distance < RATIO_THRESHOLD * pair[1].distance}
    backward = {pair[0].queryIdx: pair[0].trainIdx for pair in reverse
                if len(pair) == 2 and pair[0].distance < RATIO_THRESHOLD * pair[1].distance}
    return [match for query, match in accepted.items() if backward.get(match.trainIdx) == query]


def estimate_pair(first, second, matches, camera_a, camera_b):
    if len(matches) < MIN_MATCHES:
        return None
    points_a = np.float64([first["keypoints"][m.queryIdx].pt for m in matches])
    points_b = np.float64([second["keypoints"][m.trainIdx].pt for m in matches])
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
    return rotation, translation.reshape(3), points_a[selected], points_b[selected]


def reconstruct_sparse(images):
    intrinsics = [load_camera_matrix(record["image"].shape) for record in images]
    detector, norm = create_detector()
    for record in images:
        record["keypoints"], record["descriptors"] = detector.detectAndCompute(record["gray"], record["mask"])
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
