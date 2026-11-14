# Feature Importance Module (`importance/`)

This directory houses the multimodal (3D mesh + 2D image) guidance and feature-identification pipeline for low-poly mesh decimation.

---

## Architecture & Module Overview

```
importance/
├── __init__.py
├── config.py          # Configuration dataclasses (weights, thresholds, seeds)
├── io_utils.py        # Robust I/O for meshes, OpenCV cameras, CO3D conversions, images, masks
├── viz.py             # Visualizations: vertex coloring (PLY/OBJ), 2D projection overlays, polylines
├── mesh_prep.py       # Mesh validation, manifold check, repair, Taubin smoothing, normal calculation
├── cues_3d.py         # Multi-scale curvature, dihedral angles, sharp-edge feature curves, noise ratio, c3D, I3D
├── cues_2d.py         # Bilateral denoise, multi-scale edges, contrast saliency, entropy, density norm, polylines
├── lifting.py         # Multi-view 2D-to-3D projection, depth z-buffer occlusion, grazing weight, c2D variance
├── fusion.py          # Gated fusion (c3D * I3D * (1 + beta * c2D * I2D)), Laplacian smoothing, sizing field
├── guidance.py        # GuidanceOutput dataclass bundling all importance, curve, and confidence outputs
├── collapse_hooks.py  # Hooks interfacing with FA-QEM (quadric scaling, line quadrics, rejection rules)
└── eval.py            # Geometric & topological evaluation metrics (Chamfer distance, silhouette IoU, manifoldness)
```

---

## 1. `importance/config.py`

Houses configuration dataclasses defining parameters for feature extraction, thresholds, and determinism.

| Class / Field | Type | Description |
| :--- | :--- | :--- |
| **`Cues3DConfig`** | `dataclass` | Configuration for 3D geometric cues. |
| `curvature_rings` | `tuple[int, ...]` | Neighborhood rings for curvature estimation (default: `(1, 2, 3)`). |
| `sharp_edge_threshold_deg` | `float` | Dihedral angle threshold in degrees to qualify an edge as sharp (default: `35.0°`). |
| `weight_curvature` | `float` | Weight $w_{\text{curv}}$ in combined $I_{3D}$ (default: `0.5`). |
| `weight_sharp_edges` | `float` | Weight $w_{\text{sharp}}$ in combined $I_{3D}$ (default: `0.5`). |
| `weight_thickness` | `float` | Weight $w_{\text{thin}}$ in combined $I_{3D}$ (default: `0.0`). |
| `noise_ratio_decay` | `float` | Exponential sensitivity factor for confidence degradation $c_{3D}$ (default: `0.5`). |
| `thickness_enabled` | `bool` | Toggle for thickness calculation stub (default: `False`). |
| `percentile_clip` | `tuple[float, float]`| Percentile bounds `(p_min, p_max)` for robust $[0, 1]$ clipping (default: `(1.0, 99.0)`). |
| `random_seed` | `int` | Seed for deterministic runs (default: `42`). |
| **`ImportanceConfig`** | `dataclass` | Top-level configuration for the entire pipeline. |
| `beta` | `float` | Weight for 2D cues in gated fusion: $I = c_{3D} I_{3D} (1 + \beta c_{2D} I_{2D})$. |
| `smoothing_iterations` | `int` | Graph Laplacian smoothing steps on the final scalar importance field. |

---

## 2. `importance/io_utils.py`

Provides standard camera geometry, explicit dataset loaders, mesh I/O, and synthetic test shape generators.

### Classes

#### `Camera`
Internal camera representation using standard **OpenCV coordinate conventions**:
* $+X$ points right, $+Y$ points down, $+Z$ points forward along the optical axis.
* World-to-camera transformation: $X_{\text{cam}} = R \cdot X_{\text{world}} + t$.
* Pinhole projection:
  $$u = f_x \frac{X_{\text{cam}}}{Z_{\text{cam}}} + c_x, \quad v = f_y \frac{Y_{\text{cam}}}{Z_{\text{cam}}} + c_y$$

* **Methods:**
  * `world_to_camera(points)`: Transforms $(N, 3)$ world points to camera coordinates.
  * `camera_to_world(points)`: Transforms $(N, 3)$ camera points back to world coordinates.
  * `camera_center`: Property returning world-space optical center $C = -R^T t$.
  * `optical_axis`: Property returning the normalized world-space viewing direction.
  * `project(points)`: Projects world points to image coordinates $(u, v)$ and returns depth $Z_{\text{cam}}$.
  * `is_visible(uv, depth, margin=0.0)`: Returns boolean mask of points inside image bounds with $Z_{\text{cam}} > 0$.

### Functions

| Function | Signature | Description |
| :--- | :--- | :--- |
| `create_camera_opencv` | `(K, R, t, width, height, ...)` | Instantiates a `Camera` from native OpenCV intrinsics and world-to-cam $[R \mid t]$. |
| `create_camera_from_c2w` | `(K, c2w_rotation, camera_center, ...)` | Instantiates a `Camera` from camera-to-world rotation and center coordinates. |
| `create_camera_from_co3d_record` | `(record, fallback_image_size=None)` | **Explicit loader** for CO3D frame annotations. Converts PyTorch3D conventions (row-vector transpose, $+X$ left, $+Y$ up, NDC focal length) to standard OpenCV format. |
| `load_cameras_from_json` | `(path)` | Reads camera parameters from JSON files exported by reconstruction pipelines. |
| `evaluate_points_inside_mask` | `(points, camera, mask)` | Projects 3D points into a silhouette mask and calculates the inside-mask fraction. Used to test camera alignment and catch wrong conventions. |
| `load_mesh` | `(path)` | Loads OBJ, PLY, or OFF meshes and returns `(vertices, faces)` as `(float64, int64)`. |
| `save_mesh` | `(path, vertices, faces, vertex_colors=None)` | Writes meshes with optional per-vertex RGB colors. |
| `create_synthetic_sphere` | `(radius=1.0, subdivisions=3)` | Generates an icosphere test mesh. |
| `create_synthetic_cube` | `(side=2.0)` | Generates a cube test mesh with sharp $90^\circ$ dihedral angles. |
| `create_synthetic_torus` | `(major_radius=1.0, minor_radius=0.3)` | Generates a torus test mesh with mixed elliptic and hyperbolic curvature. |
| `load_image` | `(path, color_mode="rgb")` | Loads image as RGB, BGR, or Grayscale. |
| `load_mask` | `(path, threshold=127)` | Loads and binarizes silhouette masks (0 or 255). |

---

## 3. `importance/viz.py`

Export and debug visualization tools for scalar fields, projections, and curves.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `scalar_to_colormap` | `(values, cmap_name="turbo", vmin=None, vmax=None)` | Maps a 1D scalar array (e.g. $I, I_{3D}, I_{2D}$) to $(V, 3)$ RGB uint8 colors using `turbo`, `viridis`, `plasma`, `magma`, or `jet`. |
| `export_colored_mesh` | `(output_path, vertices, faces, scalar_values, ...)` | Colors a 3D mesh by scalar values and writes it to a colored PLY or OBJ. |
| `create_projection_overlay` | `(image, points, camera, scalars=None, draw_wireframe=False, faces=None, ...)` | Projects 3D points or wireframes onto a 2D camera image with scalar-colored points. |
| `draw_polylines_on_image` | `(image, polylines, color=(255, 0, 0), thickness=2)` | Renders 2D feature polylines onto an image canvas. |

---

## 4. `importance/mesh_prep.py`

Mesh repair, topological validation, and volume-preserving Taubin smoothing.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `compute_normals` | `(vertices, faces, weighting="angle")` | Computes unit face normals, area- or angle-weighted vertex normals, and individual triangle face areas. Angle-weighted is preferred for preserving sharp corners. |
| `validate_and_repair_mesh` | `(vertices, faces, min_area=1e-10)` | Cleans raw meshes: removes duplicate vertex references in triangles, culls zero-area degenerate faces, removes floating unreferenced vertices, re-indexes faces, and verifies manifoldness. |
| `build_vertex_neighbors` | `(num_vertices, faces)` | Constructs 1-ring vertex adjacency list. |
| `build_edge_to_faces` | `(faces)` | Constructs mapping from unique undirected edges $(u, v)$ to their incident face indices. |
| `build_laplacian_matrix` | `(num_vertices, faces)` | Builds a sparse CSR umbrella Laplacian matrix $L$ with zero row sums. |
| `taubin_smoothing` | `(vertices, faces, iterations=5, lambda_param=0.5, mu_param=-0.53, preserve_boundary=True)` | Non-shrinking Laplacian smoothing that cycles between positive diffusion ($+\lambda L V$) and negative anti-diffusion ($+\mu L V$) to suppress high-frequency noise while preserving shape volume. |

---

## 5. `importance/cues_3d.py`

3D geometric cue extraction, multi-scale curvature, sharp edge chaining, and noise estimation.

| Function / Class | Signature | Description |
| :--- | :--- | :--- |
| **`Cues3DOutput`** | `dataclass` | Packages 3D cue outputs: `I3D`, `c3D`, `feature_curves`, `pinned_flags`, `dihedral_angles`, `sharp_edges`, `vertex_sharpness`, `curvature_fine`, `curvature_coarse`, and `noise_ratio`. |
| `compute_k_ring_neighbors` | `(num_vertices, one_ring, k)` | Expands 1-ring neighborhoods to $k$-ring neighborhoods. |
| `compute_multiscale_curvature` | `(vertices, vertex_normals, one_ring, rings=(1, 2, 3))` | Computes mean curvature at multiple spatial scales via normal-projected Laplacian displacements: $\kappa_i^{(k)} = \| n_i \cdot \frac{1}{\|\mathcal{N}_k(i)\|} \sum_{j \in \mathcal{N}_k(i)} (v_j - v_i) \|$. |
| `compute_dihedral_angles` | `(faces, face_normals, edge_to_faces)` | Calculates dihedral angle $\theta = \arccos(n_1 \cdot n_2)$ for all manifold edges. |
| `extract_feature_curves` | `(num_vertices, sharp_edges)` | Chains sharp edges ($\theta \ge \theta_{\text{thresh}}$) into continuous ordered polylines (feature curves) and sets `pinned_flags`. |
| `normalize_scalar` | `(values, percentile_clip=(1.0, 99.0))` | Robust percentile-based normalization to range $[0, 1]$. |
| `compute_thickness_stub` | `(vertices, faces, enabled=False)` | Lightweight stub for ray-based thickness estimation. Controlled via config flag. |
| `compute_3d_cues` | `(vertices, faces, config=None)` | **Primary 3D Pipeline**: Computes normals, multi-scale curvature, sharp edges, feature curve polylines, noise ratio ($\kappa_{\text{fine}} / \kappa_{\text{coarse}}$), confidence $c_{3D}$, and combined importance $I_{3D} \in [0, 1]$. |

---

## 6. `importance/cues_2d.py`

2D image cues, multi-scale edge detection, contrast saliency, texture density normalization, and Douglas-Peucker polylines.

| Function / Class | Signature | Description |
| :--- | :--- | :--- |
| **`Cues2DOutput`** | `dataclass` | Packages 2D outputs: `I2D`, `edge_map`, `saliency_map`, `entropy_map`, and `polylines`. |
| `edge_preserving_denoise` | `(gray_image, d=7, sigma_color=50, sigma_space=50)` | Bilateral filtering to suppress flat-region noise while keeping structural edges intact. |
| `compute_multiscale_edges` | `(image_gray, sigmas=(1.0, 2.0, 4.0))` | Multi-scale gradient magnitude edge map using Gaussian smoothing and Sobel kernels. |
| `compute_contrast_saliency`| `(image_gray)` | Center-surround difference-of-Gaussians (DoG) visual saliency. |
| `compute_local_entropy` | `(image_gray, kernel_size=9)` | Sliding-window local entropy / texture activity measuring local information density. |
| `normalize_local_density` | `(edge_map, kernel_size=15, suppression_factor=3.0)` | Downweights edge strength in high-density regions to suppress repetitive textures (fur, grass, carpet) relative to clean structural contours. |
| `extract_2d_polylines` | `(edge_map, threshold=0.35, epsilon=2.0)` | Contours edge pixels above threshold and applies Douglas-Peucker simplification to extract 2D polylines. |
| `compute_2d_cues` | `(image, mask=None, config=None)` | **Primary 2D Pipeline**: Bilateral denoise, multi-scale edge detection, density normalization, saliency/entropy boost, background masking, and normalized $I_{2D}$. |

---

## 7. `importance/lifting.py`

Lifting multi-view 2D image cues onto 3D mesh vertices with occlusion culling and grazing angle weighting.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `render_mesh_z_buffer` | `(vertices, faces, camera)` | Rasterizes an exact analytical ray-plane depth map (z-buffer) of the mesh from the camera viewpoint. |
| `sample_bilinear` | `(image_2d, u, v)` | Subpixel bilinear interpolation of scalar values at continuous coordinates $(u, v)$. |
| `lift_2d_cues_to_vertices` | `(vertices, faces, vertex_normals, cameras, image_cues_list, config=None)` | **Primary Lifting Pipeline**: Projects vertices into calibrated camera views, performs z-buffer occlusion checks, weights views by grazing cosine $w_j = \max(0, n \cdot d_{\text{view}})^2$, computes multi-view weighted average $I_{2D}$, and calculates per-vertex confidence $c_{2D}$ penalized when views disagree (cross-view variance). |

---

## 8. `importance/fusion.py` & `importance/guidance.py`

Gated multi-modal fusion, scalar Laplacian smoothing, and sizing field generation.

| Function / Class | Signature | Description |
| :--- | :--- | :--- |
| **`GuidanceOutput`** | `dataclass` | Final guidance package: `importance`, `feature_curves`, `pinned_flags`, `confidence`, `I3D`, `I2D`, `c3D`, `c2D`, and `sizing_field`. |
| `smooth_scalar_on_mesh` | `(values, neighbors, iterations=2, blend_weight=0.5)` | 1-ring neighborhood Laplacian smoothing on scalar arrays over the mesh. |
| `compute_target_sizing_field` | `(importance, min_edge_len=0.02, max_edge_len=0.2)` | Calculates target mesh edge lengths inversely proportional to importance: $h(v) \propto \frac{1}{1 + I(v)}$. |
| `fuse_and_package_guidance` | `(vertices, faces, cues_3d, lifted_I2D, c2D, config=None)` | **Primary Fusion Pipeline**: Executes gated formula $I = c_{3D} I_{3D} (1 + \beta c_{2D} I_{2D})$, normalizes via percentile clipping, applies mesh Laplacian smoothing, and packages all guidance outputs. |

---

## 9. `importance/collapse_hooks.py`

Interfaces and predicates hooking feature importance directly into FA-QEM quadric decimation loops.

| Function / Class | Signature | Description |
| :--- | :--- | :--- |
| **`CollapseHooksConfig`** | `dataclass` | Controls quadric scaling weight `alpha_importance`, feature cost weight `weight_feature_cost`, line quadric weight `weight_line_quadric`, and minimum angle thresholds. |
| `compute_face_quadric` | `(v0, v1, v2)` | Computes the fundamental error quadric $K_p = p p^T$ for a triangular plane $p = [n^T, -n^T v_0]^T$, weighted by face area. |
| `scale_vertex_quadrics` | `(quadrics, importance, alpha=10.0)` | Scales each vertex quadric by importance: $Q'_v = (1 + \alpha I_v) Q_v$. High importance inflates collapse cost, protecting features. |
| `compute_line_quadric` | `(p0, p1, weight=10.0)` | Computes a 4x4 line quadric penalizing squared perpendicular distance from a 3D line segment $(p_0, p_1)$. |
| `add_feature_line_quadrics` | `(quadrics, vertices, feature_curves, weight=10.0)` | Adds line quadrics along all detected sharp feature curve segments. |
| `reject_edge_collapse` | `(v1, v2, new_pos, vertices, incident_faces_v1, incident_faces_v2, pinned_flags, curve_membership, ...)` | **Rejection Predicate**: Rejects collapses that cross distinct feature curves, invert face normals ($n_{\text{old}} \cdot n_{\text{new}} < \cos_{\text{min}}$), or collapse angles below minimum threshold. |
| `merge_vertex_importance` | `(i1, i2)` | Merges vertex scalar importance upon collapse via $\max(I_{v1}, I_{v2})$. |

---

## 10. `importance/eval.py`

Geometric, silhouette, and topological evaluation metrics.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `compute_chamfer_distance` | `(points_a, points_b)` | Computes bidirectional nearest-neighbor Chamfer distance between two point clouds / meshes. |
| `render_mesh_silhouette` | `(vertices, faces, camera)` | Rasterizes binary 2D silhouette masks of a mesh under a given camera projection. |
| `compute_silhouette_iou` | `(verts_orig, faces_orig, verts_dec, faces_dec, cameras)` | Computes mean multi-view silhouette Intersection-over-Union (IoU) across camera viewpoints. |
| `compute_topology_metrics` | `(vertices, faces)` | Evaluates mesh quality: mean vertex valence, valence standard deviation, and triangle minimum angle distribution. |
| `compute_feature_curve_preservation_error` | `(feature_curves, orig_vertices, decimated_vertices)` | Measures the mean Euclidean distance from original feature curve vertices to the nearest decimated vertices. |

---

## 11. `reference/collapse_loop.py`

Minimal reference decimation loop implementing the full importance-guided collapse algorithm.

| Function | Signature | Description |
| :--- | :--- | :--- |
| `compute_optimal_collapse_position_and_cost` | `(Q, v0, v1)` | Solves for the optimal merged vertex position $v^*$; falls back to endpoints and midpoint if singular. |
| `reference_qem_decimate` | `(vertices, faces, target_faces, importance=None, feature_curves=None, pinned_flags=None, config=None)` | Executes priority-queue edge collapse with version stamps, applying importance quadric scaling, line quadrics, and rejection rules down to `target_faces`. Tested on 5k–20k face meshes. |

---

## 12. `importance/pipeline.py` & `scripts/pointcloud_to_mesh.py`

Top-level orchestration entry points for end-to-end processing and point-cloud meshing.

### `importance/pipeline.py`
Orchestrates the complete flow: mesh loading $\to$ repair $\to$ 3D cues $\to$ 2D multi-view cues $\to$ lifting $\to$ gated fusion $\to$ vertex-colored mesh export $\to$ FA-QEM decimation $\to$ evaluation metrics.

**Usage Example:**
```powershell
.\.venv\Scripts\python.exe -m importance.pipeline --mesh path/to/dense_mesh.ply --images data/images --cameras outputs/camera_poses.json --masks outputs/masks --target-faces 1000 --output outputs
```

### `scripts/pointcloud_to_mesh.py`
Converts `outputs/subject_point_cloud.ply` into a triangular surface mesh ($V, F$) using Open3D Screened Poisson (with camera-oriented normals) or a robust Trimesh/Scipy fallback.

**Usage Example:**
```powershell
.\.venv\Scripts\python.exe scripts/pointcloud_to_mesh.py --input outputs/subject_point_cloud.ply --output outputs/subject_mesh.ply --cameras outputs/camera_poses.json
```

