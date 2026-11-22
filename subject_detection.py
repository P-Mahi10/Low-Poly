import cv2
import numpy as np
from pathlib import Path

from config import (
    MASK_DIR,
    MIN_MASK_ASSOCIATION_IOU,
    MIN_SEGMENTATION_CONFIDENCE,
    SEGMENTATION_MODEL,
)


def create_segmenter():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Automatic subject segmentation needs Ultralytics. Install with: "
            ".venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from error
    return YOLO(SEGMENTATION_MODEL)


def mask_iou(mask_a, mask_b):
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(mask_b, (mask_a.shape[1], mask_a.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
    intersection = np.count_nonzero((mask_a > 0) & (mask_b > 0))
    union = np.count_nonzero((mask_a > 0) | (mask_b > 0))
    return intersection / union if union else 0.0


def detect_subject_masks(images, segmenter):
    masks = []
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    subject_class = None
    previous_mask = None
    for record in images:
        result = segmenter.predict(record["image"], verbose=False)[0]
        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            raise RuntimeError(f"No segmented object detected in {record['path'].name}")
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        raw_masks = result.masks.data.cpu().numpy()
        candidates = []
        for index, (class_id, confidence) in enumerate(zip(classes, confidences)):
            if confidence < MIN_SEGMENTATION_CONFIDENCE:
                continue
            mask = cv2.resize(raw_masks[index].astype(np.uint8),
                              (record["image"].shape[1], record["image"].shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            candidates.append((int(class_id), float(confidence),
                               int(np.count_nonzero(mask)), mask * 255))
        if not candidates:
            raise RuntimeError(f"No confident segmented object in {record['path'].name}")
        if subject_class is None:
            chosen = max(candidates, key=lambda item: item[2])
            subject_class = chosen[0]
            print("Automatically selected subject class: "
                  f"{segmenter.names.get(subject_class, subject_class)}")
        else:
            same_class = [item for item in candidates if item[0] == subject_class]
            if same_class:
                chosen = max(same_class, key=lambda item: item[2])
            else:
                overlap, chosen = max(
                    ((mask_iou(previous_mask, item[3]), item) for item in candidates),
                    key=lambda item: item[0],
                )
                if overlap < MIN_MASK_ASSOCIATION_IOU:
                    raise RuntimeError(f"Subject could not be associated in {record['path'].name}")
                print(f"{record['path'].name}: associated subject by mask overlap "
                      f"(IoU {overlap:.2f})")
        masks.append(chosen[3])
        previous_mask = chosen[3]
        cv2.imwrite(str(MASK_DIR / f"{record['path'].stem}_mask.png"), chosen[3])
        print(f"{record['path'].name}: selected {chosen[2]} subject-mask pixels "
              f"(confidence {chosen[1]:.2f})")
    return masks


def load_supplied_masks(images, mask_folder):
    masks = []
    for record in images:
        path = Path(mask_folder) / f"{record['path'].stem}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape != record["image"].shape[:2]:
            mask = cv2.resize(mask, (record["image"].shape[1], record["image"].shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        masks.append((mask >= 127).astype(np.uint8) * 255)
    return masks
