"""
Top-level orchestration pipeline for feature identification, importance guidance,
and low-poly mesh decimation.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
import argparse
import json
import numpy as np

from importance.config import ImportanceConfig
from importance.io_utils import (
    Camera,
    load_mesh,
    save_mesh,
    load_image,
    load_mask,
    load_cameras_from_json,
    create_synthetic_sphere,
)
from importance.mesh_prep import validate_and_repair_mesh, compute_normals
from importance.cues_3d import compute_3d_cues
from importance.cues_2d import compute_2d_cues
from importance.lifting import lift_2d_cues_to_vertices
from importance.fusion import fuse_and_package_guidance
from importance.guidance import GuidanceOutput
from importance.viz import export_colored_mesh, create_projection_overlay
from reference.collapse_loop import reference_qem_decimate
from importance.eval import (
    compute_chamfer_distance,
    compute_silhouette_iou,
    compute_topology_metrics,
)


def run_importance_pipeline(
    mesh_path: Path,
    images_dir: Optional[Path] = None,
    cameras_file: Optional[Path] = None,
    masks_dir: Optional[Path] = None,
    output_dir: Path = Path("outputs"),
    target_faces: int = 1000,
    config: Optional[ImportanceConfig] = None,
) -> Dict[str, Any]:
    """Execute end-to-end multimodal feature guidance and decimation."""
    if config is None:
        config = ImportanceConfig()

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "debug_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- 1. Loading Mesh: {mesh_path} ---")
    raw_v, raw_f = load_mesh(mesh_path)
    clean_v, clean_f, repair_rep = validate_and_repair_mesh(raw_v, raw_f)
    print(f"Mesh loaded: {len(clean_v)} vertices, {len(clean_f)} faces.")

    # Compute normals
    _, vert_normals, _ = compute_normals(clean_v, clean_f)

    print("--- 2. Extracting 3D Cues (Curvature, Dihedral, Feature Curves) ---")
    cues_3d = compute_3d_cues(clean_v, clean_f, config=config.cues_3d)
    print(f"Found {len(cues_3d.sharp_edges)} sharp edges, {len(cues_3d.feature_curves)} feature curves.")
    print(f"Pinned vertices: {np.count_nonzero(cues_3d.pinned_flags)} / {len(clean_v)}")

    # 3. Load Cameras, Images, and Masks for 2D Cues
    cameras: List[Camera] = []
    i2d_maps: List[np.ndarray] = []

    if cameras_file is not None and cameras_file.exists() and images_dir is not None and images_dir.exists():
        print(f"--- 3. Extracting 2D Cues from Multi-View Images in {images_dir} ---")
        loaded_cams = load_cameras_from_json(cameras_file)

        for cam in loaded_cams:
            if not cam.image_name:
                continue
            img_path = images_dir / cam.image_name
            if not img_path.exists():
                continue

            img = load_image(img_path)
            mask = None
            if masks_dir is not None:
                mask_candidate = masks_dir / f"{img_path.stem}_mask.png"
                if mask_candidate.exists():
                    mask = load_mask(mask_candidate)

            cues_2d = compute_2d_cues(img, mask=mask, config=config.cues_2d)
            cameras.append(cam)
            i2d_maps.append(cues_2d.I2D)

            # Save debug projection overlay
            overlay = create_projection_overlay(
                image=img,
                points=clean_v,
                camera=cam,
                scalars=cues_3d.I3D,
                point_radius=2,
            )
            save_path = overlay_dir / f"{img_path.stem}_overlay.jpg"
            import cv2
            cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        print(f"Processed {len(cameras)} valid camera views.")
    else:
        print("Images or camera parameters not provided; proceeding in 3D-only mode.")

    # 4. Lift 2D cues to 3D vertices
    if cameras and i2d_maps:
        print("--- 4. Lifting 2D Cues onto 3D Mesh Vertices ---")
        lifted_I2D, c2D = lift_2d_cues_to_vertices(
            clean_v, clean_f, vert_normals, cameras, i2d_maps, config=config.lifting
        )
    else:
        lifted_I2D = np.zeros(len(clean_v), dtype=np.float64)
        c2D = np.zeros(len(clean_v), dtype=np.float64)

    # 5. Gated Multi-Modal Fusion
    print("--- 5. Performing Gated Fusion and Sizing Field Generation ---")
    guidance = fuse_and_package_guidance(
        clean_v, clean_f, cues_3d, lifted_I2D, c2D, config=config
    )

    # Export colored importance mesh
    colored_mesh_path = output_dir / "guidance_colored_mesh.ply"
    export_colored_mesh(colored_mesh_path, clean_v, clean_f, guidance.importance, cmap_name="turbo")
    print(f"Saved vertex-colored importance mesh: {colored_mesh_path}")

    # 6. Feature-Guided Decimation
    print(f"--- 6. Decimating Mesh to Target Polygon Count: {target_faces} faces ---")
    dec_v, dec_f, dec_imp = reference_qem_decimate(
        clean_v,
        clean_f,
        target_faces=target_faces,
        importance=guidance.importance,
        feature_curves=guidance.feature_curves,
        pinned_flags=guidance.pinned_flags,
    )
    low_poly_path = output_dir / "low_poly_mesh.ply"
    save_mesh(low_poly_path, dec_v, dec_f)
    print(f"Saved low-poly mesh: {low_poly_path} ({len(dec_v)} vertices, {len(dec_f)} faces)")

    # 7. Evaluation Metrics
    print("--- 7. Computing Evaluation Metrics ---")
    chamfer_mean, _, _ = compute_chamfer_distance(clean_v, dec_v)
    topo_metrics = compute_topology_metrics(dec_v, dec_f)

    sil_iou = 1.0
    if cameras:
        sil_iou = compute_silhouette_iou(clean_v, clean_f, dec_v, dec_f, cameras)

    metrics = {
        "initial_faces": len(clean_f),
        "target_faces": target_faces,
        "final_faces": len(dec_f),
        "final_vertices": len(dec_v),
        "chamfer_distance": float(chamfer_mean),
        "mean_valence": float(topo_metrics["mean_valence"]),
        "min_angle_deg": float(topo_metrics["min_angle_deg"]),
        "silhouette_iou": float(sil_iou),
    }

    metrics_path = output_dir / "metrics_report.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved metrics report to: {metrics_path}")
    print(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run multimodal feature importance and decimation pipeline.")
    parser.add_argument("--mesh", "-m", type=str, required=True, help="Path to input 3D mesh (PLY/OBJ)")
    parser.add_argument("--images", "-i", type=str, default="data/images", help="Path to images directory")
    parser.add_argument("--cameras", "-c", type=str, default="outputs/camera_poses.json", help="Path to cameras JSON file")
    parser.add_argument("--masks", type=str, default="outputs/masks", help="Path to masks directory")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory")
    parser.add_argument("--target-faces", "-t", type=int, default=1000, help="Target face count")
    args = parser.parse_args()

    run_importance_pipeline(
        mesh_path=Path(args.mesh),
        images_dir=Path(args.images) if args.images else None,
        cameras_file=Path(args.cameras) if args.cameras else None,
        masks_dir=Path(args.masks) if args.masks else None,
        output_dir=Path(args.output),
        target_faces=args.target_faces,
    )


if __name__ == "__main__":
    main()
