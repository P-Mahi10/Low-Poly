from pathlib import Path
import csv

import cv2
import numpy as np


# ---------- Configuration ----------

IMAGE_DIR = Path("data/images")
OUTPUT_DIR = Path("outputs")

RATIO_THRESHOLD = 0.75
RANSAC_PROBABILITY = 0.999
RANSAC_THRESHOLD_PX = 1.0
MIN_GOOD_MATCHES = 12

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------- Image loading ----------

def load_images(folder):
    paths = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if len(paths) < 2:
        raise ValueError("Provide at least two photographs.")

    images = []

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        # Keep original resolution for the initial experiment.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        images.append((path, image, gray))

    if len(images) < 2:
        raise ValueError("Fewer than two images could be loaded.")

    return images


# ---------- Camera intrinsics ----------

def make_initial_camera_matrix(image_shape):
    height, width = image_shape[:2]

    # Rough initial guess only.
    # Replace with calibrated or otherwise justified intrinsics later.
    focal_guess = float(max(width, height))

    K = np.array([
        [focal_guess, 0.0, width / 2.0],
        [0.0, focal_guess, height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K


# ---------- Feature detection and matching ----------

def create_detector():
    # SIFT is preferred for this baseline.
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2

    # Fallback for installations without SIFT.
    return cv2.ORB_create(nfeatures=5000), cv2.NORM_HAMMING


def extract_features(gray, detector):
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, norm_type):
    if desc1 is None or desc2 is None:
        return []

    matcher = cv2.BFMatcher(norm_type)

    # k=2 is required for the ratio test.
    candidates = matcher.knnMatch(desc1, desc2, k=2)

    good = []

    for pair in candidates:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < RATIO_THRESHOLD * second.distance:
            good.append(first)

    return good


# ---------- Relative camera pose ----------

def estimate_relative_pose(kp1, kp2, matches, K):
    if len(matches) < MIN_GOOD_MATCHES:
        return None

    points1 = np.float64([
        kp1[m.queryIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    points2 = np.float64([
        kp2[m.trainIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    E, ransac_mask = cv2.findEssentialMat(
        points1,
        points2,
        K,
        method=cv2.RANSAC,
        prob=RANSAC_PROBABILITY,
        threshold=RANSAC_THRESHOLD_PX,
    )

    if E is None or ransac_mask is None:
        return None

    # Some configurations can return multiple candidate essential matrices.
    if E.shape != (3, 3):
        return None

    inlier_count, R, t, pose_mask = cv2.recoverPose(
        E,
        points1,
        points2,
        K,
        mask=ransac_mask.copy(),
    )

    if pose_mask is None:
        return None

    ransac_inliers = int(np.count_nonzero(ransac_mask))
    pose_inliers = int(np.count_nonzero(pose_mask))
    total_matches = len(matches)

    # OpenCV returns pose satisfying X2 = R @ X1 + t.
    # The camera-center displacement in camera-1 coordinates is -R.T @ t.
    camera_delta = -R.T @ t
    camera_delta /= max(np.linalg.norm(camera_delta), 1e-12)

    return {
        "E": E,
        "R": R,
        "t": t,
        "camera_delta_direction": camera_delta.reshape(3),
        "total_matches": total_matches,
        "ransac_inliers": ransac_inliers,
        "pose_inliers": pose_inliers,
        "ransac_inlier_ratio": ransac_inliers / total_matches,
        "pose_inlier_ratio": pose_inliers / total_matches,
        "points1": points1,
        "points2": points2,
        "pose_mask": pose_mask,
    }


# ---------- Diagnostics ----------

def save_match_visualization(
    image1, kp1, image2, kp2, matches, pose_mask, output_path
):
    selected = [
        match for match, keep in zip(matches, pose_mask.ravel())
        if keep
    ]

    preview = cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        selected,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(str(output_path), preview)


# ---------- Main pipeline ----------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = load_images(IMAGE_DIR)
    detector, norm_type = create_detector()

    print(f"Loaded {len(images)} images.")
    print(f"Feature detector: {type(detector).__name__}")

    feature_data = []

    for path, image, gray in images:
        keypoints, descriptors = extract_features(gray, detector)

        print(f"{path.name}: {len(keypoints)} keypoints")

        feature_data.append({
            "path": path,
            "image": image,
            "gray": gray,
            "keypoints": keypoints,
            "descriptors": descriptors,
        })

    # Initial version: estimate motion between consecutive photographs.
    results = []

    for i in range(len(feature_data) - 1):
        first = feature_data[i]
        second = feature_data[i + 1]

        matches = match_features(
            first["descriptors"],
            second["descriptors"],
            norm_type,
        )

        print(
            f"\n{first['path'].name} -> {second['path'].name}: "
            f"{len(matches)} ratio-test matches"
        )

        if len(matches) < MIN_GOOD_MATCHES:
            print("Insufficient matches; skipping this pair.")
            continue

        K = make_initial_camera_matrix(first["gray"].shape)

        result = estimate_relative_pose(
            first["keypoints"],
            second["keypoints"],
            matches,
            K,
        )

        if result is None:
            print("Pose estimation failed or was ambiguous.")
            continue

        print(f"RANSAC inliers: {result['ransac_inliers']}")
        print(f"Pose inliers: {result['pose_inliers']}")
        print(
            "RANSAC inlier ratio: "
            f"{result['ransac_inlier_ratio']:.3f}"
        )
        print(
            "Camera movement direction (unit vector): "
            f"{result['camera_delta_direction']}"
        )

        # Save the geometric inlier visualization.
        output_name = (
            f"{first['path'].stem}_to_{second['path'].stem}_matches.jpg"
        )

        save_match_visualization(
            first["image"],
            first["keypoints"],
            second["image"],
            second["keypoints"],
            matches,
            result["pose_mask"],
            OUTPUT_DIR / output_name,
        )

        results.append({
            "image_1": first["path"].name,
            "image_2": second["path"].name,
            "total_matches": result["total_matches"],
            "ransac_inliers": result["ransac_inliers"],
            "pose_inliers": result["pose_inliers"],
            "ransac_inlier_ratio": result["ransac_inlier_ratio"],
            "pose_inlier_ratio": result["pose_inlier_ratio"],
            "delta_x_direction": result["camera_delta_direction"][0],
            "delta_y_direction": result["camera_delta_direction"][1],
            "delta_z_direction": result["camera_delta_direction"][2],
            "R": result["R"],
            "t": result["t"],
        })

    # Save summary metrics; rotation and translation matrices are omitted
    # here to keep the CSV readable.
    csv_path = OUTPUT_DIR / "pairwise_motion.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        columns = [
            "image_1",
            "image_2",
            "total_matches",
            "ransac_inliers",
            "pose_inliers",
            "ransac_inlier_ratio",
            "pose_inlier_ratio",
            "delta_x_direction",
            "delta_y_direction",
            "delta_z_direction",
        ]

        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for result in results:
            writer.writerow({
                key: result[key] for key in columns
            })

    print(f"\nSuccessful pairs: {len(results)}")
    print(f"Summary saved to: {csv_path}")
    print(f"Visualizations saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()