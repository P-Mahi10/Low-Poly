from typing import Tuple, Dict, Any, List, Set, Optional
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import trimesh


def build_vertex_neighbors(num_vertices: int, faces: np.ndarray) -> List[Set[int]]:
    """Build 1-ring vertex neighborhood adjacency list."""
    neighbors: List[Set[int]] = [set() for _ in range(num_vertices)]
    for f in faces:
        v0, v1, v2 = f[0], f[1], f[2]
        neighbors[v0].add(v1)
        neighbors[v0].add(v2)
        neighbors[v1].add(v0)
        neighbors[v1].add(v2)
        neighbors[v2].add(v0)
        neighbors[v2].add(v1)
    return neighbors


def build_edge_to_faces(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    """Map each undirected edge (min(u, v), max(u, v)) to a list of adjacent face indices."""
    edge_to_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for f_idx, (v0, v1, v2) in enumerate(faces):
        e0 = (min(v0, v1), max(v0, v1))
        e1 = (min(v1, v2), max(v1, v2))
        e2 = (min(v2, v0), max(v2, v0))
        edge_to_faces[e0].append(f_idx)
        edge_to_faces[e1].append(f_idx)
        edge_to_faces[e2].append(f_idx)
    return edge_to_faces


def compute_normals(
    vertices: np.ndarray,
    faces: np.ndarray,
    weighting: str = "angle",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute face normals, vertex normals, and face areas.

    Args:
        vertices: (V, 3) float64
        faces: (F, 3) int64
        weighting: 'angle' (default, angle-weighted) or 'area' (area-weighted)

    Returns:
        face_normals: (F, 3) float64 unit vectors
        vertex_normals: (V, 3) float64 unit vectors
        face_areas: (F,) float64
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e01 = v1 - v0
    e02 = v2 - v0
    cross = np.cross(e01, e02)
    cross_norm = np.linalg.norm(cross, axis=1)

    # Face areas: half the magnitude of cross product
    face_areas = 0.5 * cross_norm

    # Normalized face normals
    safe_norm = np.where(cross_norm < 1e-12, 1e-12, cross_norm)
    face_normals = cross / safe_norm[:, None]

    # Vertex normals
    num_verts = len(vertices)
    vert_normals = np.zeros((num_verts, 3), dtype=np.float64)

    if weighting == "angle":
        # Compute angles at each corner of each face
        # Corner 0 (at v0)
        e10 = v0 - v1
        e12 = v2 - v1
        e20 = v0 - v2
        e21 = v1 - v2

        def unit_vectors(v):
            n = np.linalg.norm(v, axis=1, keepdims=True)
            return v / np.maximum(n, 1e-12)

        u01, u02 = unit_vectors(e01), unit_vectors(e02)
        u10, u12 = unit_vectors(e10), unit_vectors(e12)
        u20, u21 = unit_vectors(e20), unit_vectors(e21)

        angle0 = np.arccos(np.clip(np.sum(u01 * u02, axis=1), -1.0, 1.0))
        angle1 = np.arccos(np.clip(np.sum(u10 * u12, axis=1), -1.0, 1.0))
        angle2 = np.arccos(np.clip(np.sum(u20 * u21, axis=1), -1.0, 1.0))

        np.add.at(vert_normals, faces[:, 0], face_normals * angle0[:, None])
        np.add.at(vert_normals, faces[:, 1], face_normals * angle1[:, None])
        np.add.at(vert_normals, faces[:, 2], face_normals * angle2[:, None])
    else:
        # Area weighted
        weighted = face_normals * face_areas[:, None]
        np.add.at(vert_normals, faces[:, 0], weighted)
        np.add.at(vert_normals, faces[:, 1], weighted)
        np.add.at(vert_normals, faces[:, 2], weighted)

    vn_norm = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    vert_normals = vert_normals / np.maximum(vn_norm, 1e-12)

    return face_normals, vert_normals, face_areas


def validate_and_repair_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    min_area: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Inspect and clean mesh: remove degenerate faces, unreferenced vertices, and check manifoldness.

    Returns:
        cleaned_vertices: (V_clean, 3)
        cleaned_faces: (F_clean, 3)
        report: Dict containing diagnostics (counts of removed elements, manifold status).
    """
    initial_v_count = len(vertices)
    initial_f_count = len(faces)

    # 1. Remove invalid indices or faces where vertices are identical (e.g. (v, v, w))
    valid_face_mask = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
        & (faces[:, 0] >= 0)
        & (faces[:, 1] >= 0)
        & (faces[:, 2] >= 0)
        & (faces[:, 0] < initial_v_count)
        & (faces[:, 1] < initial_v_count)
        & (faces[:, 2] < initial_v_count)
    )
    faces = faces[valid_face_mask]

    # 2. Remove degenerate / zero-area faces
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    non_degenerate = areas > min_area
    faces = faces[non_degenerate]

    # 3. Remove unreferenced vertices and re-index faces
    unique_v_indices, inverse = np.unique(faces.ravel(), return_inverse=True)
    cleaned_vertices = vertices[unique_v_indices]
    cleaned_faces = inverse.reshape(faces.shape)

    # 4. Check edge-manifoldness
    edge_to_f = build_edge_to_faces(cleaned_faces)
    non_manifold_edges = sum(1 for f_list in edge_to_f.values() if len(f_list) > 2)
    boundary_edges = sum(1 for f_list in edge_to_f.values() if len(f_list) == 1)

    report = {
        "initial_vertices": initial_v_count,
        "initial_faces": initial_f_count,
        "cleaned_vertices": len(cleaned_vertices),
        "cleaned_faces": len(cleaned_faces),
        "removed_faces": initial_f_count - len(cleaned_faces),
        "removed_vertices": initial_v_count - len(cleaned_vertices),
        "is_edge_manifold": (non_manifold_edges == 0),
        "non_manifold_edges_count": non_manifold_edges,
        "boundary_edges_count": boundary_edges,
        "is_closed": (boundary_edges == 0 and non_manifold_edges == 0),
    }

    return cleaned_vertices, cleaned_faces, report


def build_laplacian_matrix(num_vertices: int, faces: np.ndarray) -> sp.csr_matrix:
    """Build sparse symmetric umbrella Laplacian matrix with zero row sums."""
    neighbors = build_vertex_neighbors(num_vertices, faces)
    rows, cols, data = [], [], []

    for v_idx, nbrs in enumerate(neighbors):
        deg = len(nbrs)
        if deg > 0:
            weight = 1.0 / deg
            for nbr in nbrs:
                rows.append(v_idx)
                cols.append(nbr)
                data.append(weight)
            rows.append(v_idx)
            cols.append(v_idx)
            data.append(-1.0)
        else:
            rows.append(v_idx)
            cols.append(v_idx)
            data.append(0.0)

    L = sp.csr_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices), dtype=np.float64)
    return L


def taubin_smoothing(
    vertices: np.ndarray,
    faces: np.ndarray,
    iterations: int = 5,
    lambda_param: float = 0.5,
    mu_param: float = -0.53,
    preserve_boundary: bool = True,
) -> np.ndarray:
    """Apply volume-preserving Taubin smoothing to mesh vertices.

    Iterates between positive diffusion (+lambda * L * V) and negative anti-diffusion (+mu * L * V).
    Condition for stability and volume preservation:
        0 < lambda < -mu, with 1/lambda + 1/mu < 0.

    Args:
        vertices: (V, 3) float64
        faces: (F, 3) int64
        iterations: Number of Taubin cycles
        lambda_param: Positive shrinkage step (typically 0.3 - 0.6)
        mu_param: Negative expansion step (typically -0.31 - -0.63)
        preserve_boundary: Keep boundary vertices pinned

    Returns:
        smoothed_vertices: (V, 3) float64
    """
    if iterations <= 0:
        return vertices.copy()

    smoothed = vertices.astype(np.float64).copy()
    num_verts = len(vertices)

    # Detect boundary vertices if preservation is requested
    boundary_mask = np.zeros(num_verts, dtype=bool)
    if preserve_boundary:
        edge_to_f = build_edge_to_faces(faces)
        for (u, v), f_list in edge_to_f.items():
            if len(f_list) == 1:
                boundary_mask[u] = True
                boundary_mask[v] = True

    L = build_laplacian_matrix(num_verts, faces)

    for _ in range(iterations):
        # Step 1: positive diffusion (shrinkage)
        disp1 = L.dot(smoothed)
        if preserve_boundary:
            disp1[boundary_mask] = 0.0
        smoothed += lambda_param * disp1

        # Step 2: negative anti-diffusion (expansion)
        disp2 = L.dot(smoothed)
        if preserve_boundary:
            disp2[boundary_mask] = 0.0
        smoothed += mu_param * disp2

    return smoothed
