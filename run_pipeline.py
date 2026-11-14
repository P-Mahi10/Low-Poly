from config import IMAGE_DIR, OUTPUT_DIR
from point_cloud import (find_co3d_sequence, reconstruct_co3d_per_image,
                         save_ply)
from photogrammetry import load_images, reconstruct_sparse
from subject_detection import create_segmenter, detect_subject_masks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    co3d = find_co3d_sequence(IMAGE_DIR)
    if co3d is not None:
        reconstruct_co3d_per_image(IMAGE_DIR, *co3d)
        return
    images = load_images(IMAGE_DIR)
    masks = detect_subject_masks(images, create_segmenter())
    for image, mask in zip(images, masks):
        image["mask"] = mask
    points, colors = reconstruct_sparse(images)
    save_ply(OUTPUT_DIR / "subject_point_cloud.ply", points, colors)
    print(f"Saved {len(points)} sparse points to {OUTPUT_DIR / 'subject_point_cloud.ply'}")


if __name__ == "__main__":
    main()
