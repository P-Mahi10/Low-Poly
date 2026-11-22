import cv2

from config import SUPPORTED_EXTENSIONS
from stage2 import (create_detector, estimate_pair, load_camera_matrix,
                    match_features, reconstruct_sparse, reconstruct_with_known_poses)


def load_images(folder):
    nested_images = folder / "images"
    if nested_images.is_dir():
        folder = nested_images
    paths = sorted((path for path in folder.iterdir()
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS),
                   key=lambda path: path.name.lower())
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            images.append({"path": path, "image": image,
                           "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)})
    if len(images) < 2:
        raise ValueError(f"At least two readable photos are required in {folder}")
    return images
