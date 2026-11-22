from config import IMAGE_DIR, OUTPUT_DIR
from point_cloud import find_co3d_pose_sequence, find_co3d_sequence, save_ply
from photogrammetry import load_images
from stage2 import reconstruct_sparse, reconstruct_with_known_poses
from stage3 import export_per_image_clouds, reconstruct_dense_cloud
from subject_detection import create_segmenter, detect_subject_masks, load_supplied_masks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    co3d = find_co3d_sequence(IMAGE_DIR)
    if co3d is not None:
        export_per_image_clouds(IMAGE_DIR, *co3d)
        reconstruct_dense_cloud(IMAGE_DIR, *co3d)
        return
    images = load_images(IMAGE_DIR)
    masks = load_supplied_masks(images, IMAGE_DIR / "masks")
    if masks is None:
        masks = detect_subject_masks(images, create_segmenter())
    else:
        print(f"Using {len(masks)} supplied foreground masks.")
    for image, mask in zip(images, masks):
        image["mask"] = mask
    co3d_poses = find_co3d_pose_sequence(IMAGE_DIR)
    if co3d_poses is not None:
        points, colors = reconstruct_with_known_poses(images, co3d_poses[1])
    else:
        points, colors = reconstruct_sparse(images)
    save_ply(OUTPUT_DIR / "subject_point_cloud.ply", points, colors)
    print(f"Saved {len(points)} sparse points to {OUTPUT_DIR / 'subject_point_cloud.ply'}")


if __name__ == "__main__":
    main()
