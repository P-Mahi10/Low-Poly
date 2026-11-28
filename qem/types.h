#pragma once
// Shared types: CGAL mesh typedefs, a tiny Vec3, the Quadric, and the ML hook struct.
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace lp {

using Kernel   = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point3   = Kernel::Point_3;
using Mesh     = CGAL::Surface_mesh<Point3>;
using Vertex   = Mesh::Vertex_index;
using Halfedge = Mesh::Halfedge_index;
using Edge     = Mesh::Edge_index;
using Face     = Mesh::Face_index;

// ---------------------------------------------------------------- Vec3
struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
};
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm2(const Vec3& a) { return dot(a, a); }
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalized(const Vec3& a) {
    double n = norm(a);
    return n > 0 ? a * (1.0 / n) : Vec3();
}
inline Vec3 to_vec(const Point3& p) { return {p.x(), p.y(), p.z()}; }
inline Point3 to_point(const Vec3& v) { return Point3(v.x, v.y, v.z); }

// ---------------------------------------------------------------- Quadric
// Symmetric 4x4 matrix stored as 10 numbers:  [a b c d; b e f g; c f h i; d g i j]
// eval(p) = [p 1] Q [p 1]^T  (a squared-distance-like error, units: length^2)
struct Quadric {
    double a = 0, b = 0, c = 0, d = 0, e = 0, f = 0, g = 0, h = 0, i = 0, j = 0;

    // plane n.x + dd = 0 with unit normal n, scaled by weight w
    static Quadric from_plane(const Vec3& n, double dd, double w = 1.0) {
        Quadric q;
        q.a = w * n.x * n.x; q.b = w * n.x * n.y; q.c = w * n.x * n.z; q.d = w * n.x * dd;
        q.e = w * n.y * n.y; q.f = w * n.y * n.z; q.g = w * n.y * dd;
        q.h = w * n.z * n.z; q.i = w * n.z * dd;
        q.j = w * dd * dd;
        return q;
    }
    Quadric& operator+=(const Quadric& o) {
        a += o.a; b += o.b; c += o.c; d += o.d; e += o.e;
        f += o.f; g += o.g; h += o.h; i += o.i; j += o.j;
        return *this;
    }
    friend Quadric operator+(Quadric x, const Quadric& y) { x += y; return x; }
    Quadric scaled(double s) const {
        Quadric q = *this;
        q.a *= s; q.b *= s; q.c *= s; q.d *= s; q.e *= s;
        q.f *= s; q.g *= s; q.h *= s; q.i *= s; q.j *= s;
        return q;
    }
    double eval(const Vec3& p) const {
        return a * p.x * p.x + 2 * b * p.x * p.y + 2 * c * p.x * p.z + 2 * d * p.x
             + e * p.y * p.y + 2 * f * p.y * p.z + 2 * g * p.y
             + h * p.z * p.z + 2 * i * p.z + j;
    }
    // Minimiser of eval(); false if the 3x3 system is (near) singular.
    bool optimum(Vec3& out) const {
        double det = a * (e * h - f * f) - b * (b * h - f * c) + c * (b * f - e * c);
        double s = std::abs(a) + std::abs(e) + std::abs(h);
        if (!(std::abs(det) > 1e-12 * s * s * s)) return false;
        double inv = 1.0 / det, rx = -d, ry = -g, rz = -i;
        out.x = (rx * (e * h - f * f) - b * (ry * h - f * rz) + c * (ry * f - e * rz)) * inv;
        out.y = (a * (ry * h - f * rz) - rx * (b * h - f * c) + c * (b * rz - ry * c)) * inv;
        out.z = (a * (e * rz - ry * f) - b * (b * rz - ry * c) + rx * (b * f - e * c)) * inv;
        return std::isfinite(out.x) && std::isfinite(out.y) && std::isfinite(out.z);
    }
};

// ---------------------------------------------------------------- ML hook
// Everything here defaults to "no effect". The ML stage can fill these in
// before Simplifier::run(); indices refer to the mesh as handed to the Simplifier.
struct MLHints {
    // Per-vertex importance (size == mesh.num_vertices(), or empty).
    // An edge's cost is multiplied by max(importance[u], importance[v]):
    // values > 1 delay collapses there (keep detail), values < 1 encourage them.
    std::vector<float> vertex_importance;

    // Hard constraints: edges touching a protected vertex / listed edges never collapse.
    std::unordered_set<std::uint32_t> protected_vertices;
    std::unordered_set<std::uint32_t> protected_edges;

    // (Per-view silhouette weights alpha_i live in Camera::weight.)

    double importance(Vertex u, Vertex v) const {
        if (vertex_importance.empty()) return 1.0;
        auto iu = static_cast<std::size_t>(u), iv = static_cast<std::size_t>(v);
        float a = iu < vertex_importance.size() ? vertex_importance[iu] : 1.f;
        float b = iv < vertex_importance.size() ? vertex_importance[iv] : 1.f;
        return std::max(a, b);
    }
    bool is_protected(Vertex u, Vertex v, Edge e) const {
        if (protected_edges.empty() && protected_vertices.empty()) return false;
        return protected_edges.count(static_cast<std::uint32_t>(e)) ||
               protected_vertices.count(static_cast<std::uint32_t>(u)) ||
               protected_vertices.count(static_cast<std::uint32_t>(v));
    }
};

}  // namespace lp