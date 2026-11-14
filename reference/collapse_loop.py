import heapq
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import numpy as np

from importance.collapse_hooks import (
    compute_face_quadric,
    scale_vertex_quadrics,
    add_feature_line_quadrics,
    build_feature_curve_membership,
    reject_edge_collapse,
    merge_vertex_importance,
    CollapseHooksConfig,
)


def compute_optimal_collapse_position_and_cost(
    Q: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Find the optimal merged vertex position and associated quadric error cost.

    Tries solving the linear system for the optimal position; falls back to evaluating
    endpoints v0, v1, and midpoint (v0 + v1)/2 if the matrix is singular or poorly conditioned.
    """
    A = np.zeros((4, 4), dtype=np.float64)
    A[:3, :3] = Q[:3, :3]
    A[:3, 3] = Q[:3, 3]
    A[3, :3] = 0.0
    A[3, 3] = 1.0

    b = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    # Attempt direct inversion if determinant is non-trivial
    try:
        cond = np.linalg.cond(A[:3, :3])
        if cond < 1e5:
            pos = np.linalg.solve(A, b)[:3]
            # Safety check: optimal point should not jump wildly far from edge endpoints
            mid = 0.5 * (v0 + v1)
            edge_len = np.linalg.norm(v1 - v0)
            if np.linalg.norm(pos - mid) < 3.0 * (edge_len + 1e-4):
                v_hom = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)
                cost = float(v_hom.T @ Q @ v_hom)
                return pos, max(0.0, cost)
    except (np.linalg.LinAlgError, ValueError):
        pass

    # Fallback candidates: v0, v1, midpoint
    candidates = [v0, v1, 0.5 * (v0 + v1)]
    best_pos = candidates[0]
    best_cost = float("inf")

    for cand in candidates:
        v_hom = np.array([cand[0], cand[1], cand[2], 1.0], dtype=np.float64)
        c = float(v_hom.T @ Q @ v_hom)
        if c < best_cost:
            best_cost = c
            best_pos = cand

    return best_pos, max(0.0, best_cost)


def reference_qem_decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
    importance: Optional[np.ndarray] = None,
    feature_curves: Optional[List[List[int]]] = None,
    pinned_flags: Optional[np.ndarray] = None,
    config: Optional[CollapseHooksConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference FA-QEM edge collapse decimation loop using the importance hook interface.

    Args:
        vertices: (V, 3) float64
        faces: (F, 3) int64
        target_faces: Desired target polygon count (typically 5k-20k or reduced count)
        importance: Optional (V,) scalar array in [0, 1]
        feature_curves: Optional lists of vertex IDs defining sharp curves
        pinned_flags: Optional (V,) boolean mask
        config: CollapseHooksConfig parameters

    Returns:
        simplified_vertices: (V_out, 3) float64
        simplified_faces: (F_out, 3) int64
        simplified_importance: (V_out,) float64
    """
    if config is None:
        config = CollapseHooksConfig()

    num_verts = len(vertices)
    cur_faces_count = len(faces)
    if cur_faces_count <= target_faces:
        imp = importance.copy() if importance is not None else np.zeros(num_verts)
        return vertices.copy(), faces.copy(), imp

    # Working copies of geometry and attributes
    verts = vertices.astype(np.float64).copy()
    if importance is not None:
        imp = np.clip(importance.astype(np.float64).copy(), 0.0, 1.0)
    else:
        imp = np.zeros(num_verts, dtype=np.float64)

    pinned = pinned_flags.copy() if pinned_flags is not None else np.zeros(num_verts, dtype=bool)
    curves = feature_curves if feature_curves is not None else []
    curve_membership = build_feature_curve_membership(num_verts, curves)

    # 1. Initialize Fundamental Face Quadrics and Accumulate at Vertices
    quadrics = np.zeros((num_verts, 4, 4), dtype=np.float64)
    face_map: Dict[int, List[int]] = {}  # face_id -> [v0, v1, v2]
    vert_to_faces: Dict[int, Set[int]] = defaultdict(set)

    for f_idx, f in enumerate(faces):
        face_map[f_idx] = list(f)
        q_f = compute_face_quadric(verts[f[0]], verts[f[1]], verts[f[2]])
        quadrics[f[0]] += q_f
        quadrics[f[1]] += q_f
        quadrics[f[2]] += q_f
        vert_to_faces[f[0]].add(f_idx)
        vert_to_faces[f[1]].add(f_idx)
        vert_to_faces[f[2]].add(f_idx)

    # 2. Scale Quadrics by Vertex Importance
    if np.any(imp > 0):
        quadrics = scale_vertex_quadrics(quadrics, imp, alpha=config.alpha_importance)

    # 3. Add Feature Line Quadrics
    if curves:
        quadrics = add_feature_line_quadrics(
            quadrics, verts, curves, weight=config.weight_line_quadric
        )

    # Version tracking for lazy heap updates
    vert_versions = np.zeros(num_verts, dtype=np.int32)
    alive_verts = np.ones(num_verts, dtype=bool)

    # Adjacency tracking
    vert_adj: Dict[int, Set[int]] = defaultdict(set)
    for f in faces:
        vert_adj[f[0]].add(f[1])
        vert_adj[f[0]].add(f[2])
        vert_adj[f[1]].add(f[0])
        vert_adj[f[1]].add(f[2])
        vert_adj[f[2]].add(f[0])
        vert_adj[f[2]].add(f[1])

    # 4. Generate Candidate Edges into Priority Queue
    heap: List[Tuple[float, int, int, int, int, np.ndarray]] = []

    def push_edge(u: int, v: int):
        if u == v or not alive_verts[u] or not alive_verts[v]:
            return
        v_u, v_v = min(u, v), max(u, v)
        Q_edge = quadrics[v_u] + quadrics[v_v]
        pos, q_cost = compute_optimal_collapse_position_and_cost(Q_edge, verts[v_u], verts[v_v])
        # Direct geometric feature cost from endpoint importance (Diagram 06)
        feat_cost = config.weight_feature_cost * (imp[v_u] + imp[v_v])
        total_cost = q_cost + feat_cost
        heapq.heappush(
            heap,
            (total_cost, vert_versions[v_u], vert_versions[v_v], v_u, v_v, pos)
        )

    seen_edges: Set[Tuple[int, int]] = set()
    for u in range(num_verts):
        for v in vert_adj[u]:
            e = (min(u, v), max(u, v))
            if e not in seen_edges:
                seen_edges.add(e)
                push_edge(e[0], e[1])

    # 5. Main Collapse Loop
    active_faces = set(face_map.keys())

    while len(active_faces) > target_faces and heap:
        cost, ver_u, ver_v, u, v, new_pos = heapq.heappop(heap)

        # Check for stale heap entries
        if not alive_verts[u] or not alive_verts[v]:
            continue
        if vert_versions[u] != ver_u or vert_versions[v] != ver_v:
            continue

        # Gather incident triangles for rejection predicates
        faces_u = [tuple(face_map[f]) for f in vert_to_faces[u] if f in active_faces]
        faces_v = [tuple(face_map[f]) for f in vert_to_faces[v] if f in active_faces]

        # Rejection rule hook
        if reject_edge_collapse(
            v1=u,
            v2=v,
            new_pos=new_pos,
            vertices=verts,
            incident_faces_v1=faces_u,
            incident_faces_v2=faces_v,
            pinned_flags=pinned,
            curve_membership=curve_membership,
            min_cos_flip=config.normal_flip_min_cos,
            min_angle_deg=config.min_triangle_angle_deg,
        ):
            continue

        # --- Execute Collapse: merge v into u ---
        verts[u] = new_pos
        alive_verts[v] = False
        imp[u] = merge_vertex_importance(imp[u], imp[v])
        pinned[u] = pinned[u] or pinned[v]
        quadrics[u] += quadrics[v]
        vert_versions[u] += 1
        vert_versions[v] += 1

        # Update connectivity: transfer faces containing v to u
        incident_to_v = list(vert_to_faces[v])
        for f_idx in incident_to_v:
            if f_idx not in active_faces:
                continue
            face_verts = face_map[f_idx]
            if u in face_verts:
                # Face has both u and v -> collapses to a line or degenerates; remove it!
                active_faces.remove(f_idx)
                for fv in face_verts:
                    vert_to_faces[fv].discard(f_idx)
            else:
                # Replace v with u in this triangle
                new_face = [u if fv == v else fv for fv in face_verts]
                face_map[f_idx] = new_face
                vert_to_faces[u].add(f_idx)

        vert_to_faces[v].clear()

        # Update neighbor graph
        nbrs_v = list(vert_adj[v])
        for nbr in nbrs_v:
            vert_adj[nbr].discard(v)
            if nbr != u and alive_verts[nbr]:
                vert_adj[nbr].add(u)
                vert_adj[u].add(nbr)
                vert_versions[nbr] += 1
        vert_adj[u].discard(v)
        vert_adj[v].clear()

        # Re-push all edges incident to the updated vertex u
        for nbr in vert_adj[u]:
            push_edge(u, nbr)

    # 6. Re-index remaining active faces and vertices
    remaining_faces = np.array([face_map[f] for f in active_faces], dtype=np.int64)
    if len(remaining_faces) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros(0)

    unique_v_indices, inverse = np.unique(remaining_faces.ravel(), return_inverse=True)
    simplified_verts = verts[unique_v_indices]
    simplified_faces = inverse.reshape(remaining_faces.shape)
    simplified_importance = imp[unique_v_indices]

    return simplified_verts, simplified_faces, simplified_importance
