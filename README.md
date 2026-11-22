# Low-Poly Image Reconstruction

Run the entire pipeline with one command:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

`run_pipeline.py` is the single entry point. It chooses the best available
reconstruction path automatically.

## Project structure

- `run_pipeline.py` runs the complete workflow.
- `subject_detection.py` finds the subject automatically and saves masks for
  ordinary photo folders.
- `stage2.py` performs feature extraction, feature matching, camera-pose
  estimation, pair refinement, and sparse reconstruction.
- `stage3.py` decodes depth maps, exports per-image clouds, checks agreement
  between views, fuses the depth maps, and writes the cleaned dense cloud.
- `photogrammetry.py` and `point_cloud.py` preserve the original public entry
  points for other project files.
- `config.py` contains shared folders and adjustable settings.

## Current CO3D data

The files in `data/images/` contain a complete CO3D sequence. The runner
detects it from `frame_annotations.jgz` and uses `images/`, `masks/`,
`depths/`, and `depth_masks/`. This path does not use YOLO or estimated camera
poses. It back-projects valid foreground depth pixels with the supplied camera
geometry and produces a dense PLY point cloud.

CO3D mode writes one point cloud per image in `outputs/per_image/` and fuses
the geometrically consistent foreground points into one cleaned dense cloud.

The result is written to `outputs/subject_point_cloud.ply` and can be opened in
Blender, MeshLab, or CloudCompare.

## Ordinary photo folders

If the CO3D metadata folders are absent, place at least two overlapping photos
in `data/images/`. The runner then performs automatic instance segmentation,
classical feature matching, relative-pose estimation, and sparse triangulation.
This fallback needs a recognisable, textured, stationary subject and benefits
from camera calibration in `data/camera_matrix.npy`.

Install dependencies when setting up a new environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
