# File Overview

A brief description of the source files and the sample data in this project.

| File | What it does |
|---|---|
| `main.cpp` | Program entry point. Parses command-line arguments, loads and cleans the input mesh, configures the simplifier, optionally loads COLMAP cameras and silhouette masks, runs simplification, and writes the output mesh and optional history. |
| `types.h` | Defines the shared CGAL mesh types, the small `Vec3` math helper, the 4×4 quadric representation, and the ML hint structures for vertex importance and protected vertices/edges. |
| `cost.h` | Implements the simplification cost functions. It contains the feature-aware quadric error metric (FA-QEM), geometric/boundary/sharp-feature penalties, normal-deviation penalty, and the optional multi-view silhouette cost term. |
| `simplifier.h` | Runs the edge-collapse simplification loop. It maintains a priority queue of candidate edges, evaluates costs, checks candidate collapses, updates nearby candidates, and records each accepted collapse. |
| `validator.h` | Checks whether an edge collapse is safe. It rejects collapses that violate mesh topology, flip or severely alter normals, create degenerate triangles, or produce poor-quality triangles. |
| `camera.h` | Reads COLMAP `cameras.txt` and `images.txt`, stores camera intrinsics and poses, computes camera centres, and projects 3D points into image coordinates. |
| `silhouette.h` | Loads per-view silhouette masks through OpenCV, builds signed distance fields for the masks, and samples those fields during silhouette-cost evaluation. This feature is compiled only when OpenCV is detected. |
| `history.h` | Stores information about each accepted edge collapse and writes the records to the text file specified by `--history`. |
| `CMakeLists.txt` | Defines the CMake project, requires C++17 and CGAL, creates the `lowpoly` executable, and optionally detects/links OpenCV. |
| `out_poisson.ply` | Included example input mesh. It is a Poisson-reconstructed PLY mesh that can be used to test the simplifier. |

## Overall flow

1. `main.cpp` reads the mesh and performs initial cleanup using CGAL.
2. `simplifier.h` builds and processes the edge-collapse priority queue.
3. `cost.h` scores candidate collapses using FA-QEM and, when enabled, the silhouette term.
4. `validator.h` prevents invalid or damaging collapses.
5. `camera.h` and `silhouette.h` provide the camera and mask data used by the optional silhouette term.
6. `history.h` records accepted collapses when history output is requested.
7. `main.cpp` writes the final simplified mesh.

**Note:** The ML fields in `types.h` are integration hooks only at this stage. This project does not currently load or execute an ML model.
