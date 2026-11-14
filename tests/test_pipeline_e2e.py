import tempfile
from pathlib import Path
import numpy as np
import cv2
import json

from importance.io_utils import (
    create_synthetic_sphere,
    save_mesh,
    create_camera_opencv,
)
from importance.pipeline import run_importance_pipeline


def test_pipeline_end_to_end():
    """Verify complete end-to-end execution of the multimodal pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mesh_path = tmp_path / "input_sphere.ply"
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "masks"
        output_dir = tmp_path / "outputs"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        # 1. Create dense input mesh (1280 faces)
        verts, faces = create_synthetic_sphere(radius=1.0, subdivisions=3)
        save_mesh(mesh_path, verts, faces)

        # 2. Create synthetic camera, image, and mask
        K = np.array([[300.0, 0, 150], [0, 300.0, 150], [0, 0, 1]], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        t = np.array([0.0, 0.0, -3.0], dtype=np.float64)

        img_name = "test_view_01.jpg"
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(img, (150, 150), 60, (255, 255, 255), -1)
        cv2.imwrite(str(images_dir / img_name), img)

        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[80:220, 80:220] = 255
        cv2.imwrite(str(masks_dir / "test_view_01_mask.png"), mask)

        cameras_file = tmp_path / "cameras.json"
        cam_dict = [
            {
                "K": K.tolist(),
                "R": R.tolist(),
                "t": t.tolist(),
                "width": 300,
                "height": 300,
                "image_name": img_name,
            }
        ]
        cameras_file.write_text(json.dumps(cam_dict), encoding="utf-8")

        # 3. Run entire pipeline
        target_faces = 400
        metrics = run_importance_pipeline(
            mesh_path=mesh_path,
            images_dir=images_dir,
            cameras_file=cameras_file,
            masks_dir=masks_dir,
            output_dir=output_dir,
            target_faces=target_faces,
        )

        # 4. Verify outputs
        assert (output_dir / "guidance_colored_mesh.ply").exists()
        assert (output_dir / "low_poly_mesh.ply").exists()
        assert (output_dir / "metrics_report.json").exists()
        assert (output_dir / "debug_overlays" / "test_view_01_overlay.jpg").exists()

        assert metrics["final_faces"] <= target_faces + 10
        assert metrics["chamfer_distance"] < 0.10
        assert metrics["silhouette_iou"] > 0.80
