#pragma once
// Cost terms.
//   FAQEMCost      : geometric-feature quadric (base + boundary/sharp-feature) + boundary-area
//                    quadric + normal-deviation penalty. Also picks the collapse target v'.
//   ICostTerm      : interface for extra terms added on top (silhouette now, topology later).
//   SilhouetteCost : multi-view image-silhouette term  lambda_sil * dE_sil(e).
// All terms are in squared-length units so they can be summed directly.
#include "camera.h"
#include "silhouette.h"
#include "types.h"

#include <CGAL/boost/graph/iterator.h>
#include <CGAL/boost/graph/helpers.h>

#include <cmath>
#include <vector>

namespace lp {

// ------------------------------------------------------------ geometry helpers
// CGAL::is_border(vertex, mesh) returns an optional halfedge (not a bool) -> normalise.
inline bool on_border(const Mesh& m, Vertex v) { return static_cast<bool>(CGAL::is_border(v, m)); }
inline bool on_border(const Mesh& m, Edge e) { return static_cast<bool>(CGAL::is_border(e, m)); }

inline Vec3 face_normal_unit(const Mesh& m, Face f) {
    Halfedge h = m.halfedge(f);
    Vec3 a = to_vec(m.point(m.source(h)));
    Vec3 b = to_vec(m.point(m.target(h)));
    Vec3 c = to_vec(m.point(m.target(m.next(h))));
    return normalized(cross(b - a, c - a));
}

inline double tri_quality(const Vec3& a, const Vec3& b, const Vec3& c) {
    double area = 0.5 * norm(cross(b - a, c - a));
    double s = norm2(b - a) + norm2(c - b) + norm2(a - c);
    return s > 0 ? 4.0 * std::sqrt(3.0) * area / s : 0.0;  // 1 = equilateral, 0 = sliver
}

// What happens to the triangles around edge (u,v) if both vertices move to p_new.
struct RingStats {
    double min_cos = 1.0;          // smallest cos(angle between old and new face normal)
    double normal_penalty = 0.0;   // sum_f A_f * (1 - cos_f)            [length^2]
    bool degenerate = false;       // some new triangle collapses to a line/point
    bool bad_quality = false;      // some new triangle is a sliver (and worse than before)
};

inline RingStats analyze_collapse(const Mesh& m, Vertex u, Vertex v, const Vec3& p_new,
                                  double min_quality) {
    RingStats rs;
    auto visit = [&](Vertex w, Vertex other) {
        for (Halfedge h : CGAL::halfedges_around_target(w, m)) {
            if (m.face(h) == Mesh::null_face()) continue;
            Vertex a = m.source(h), c = m.target(m.next(h));
            if (a == other || c == other) continue;  // this face disappears in the collapse
            Vec3 pa = to_vec(m.point(a)), pb = to_vec(m.point(w)), pc = to_vec(m.point(c));
            Vec3 n0 = cross(pb - pa, pc - pa);
            Vec3 n1 = cross(p_new - pa, pc - pa);
            double l0 = norm(n0), l1 = norm(n1);
            if (l1 <= 1e-10 * norm2(pc - pa)) { rs.degenerate = true; continue; }
            if (l0 > 0) {
                double cs = dot(n0, n1) / (l0 * l1);
                rs.min_cos = std::min(rs.min_cos, cs);
                rs.normal_penalty += 0.5 * l0 * (1.0 - cs);
            }
            double q1 = tri_quality(pa, p_new, pc), q0 = tri_quality(pa, pb, pc);
            if (q1 < min_quality && q1 < q0) rs.bad_quality = true;
        }
    };
    visit(u, v);
    visit(v, u);
    return rs;
}

// ------------------------------------------------------------ FA-QEM
struct FAQEMParams {
    double w_area = 10.0;           // scale of the boundary-area quadric Q^A
    double w_feature = 5.0;         // weight of sharp-feature constraint planes (Q_boundary)
    double w_normal = 1.0;          // weight of the normal-preservation term
    double sharp_angle_deg = 40.0;  // dihedral angle above which an edge is a feature edge
    double max_move_factor = 1.5;   // optimal v' must stay within factor*|uv| of the midpoint
    double min_triangle_quality = 0.02;
};

class FAQEMCost {
public:
    struct Eval {
        bool valid = false;
        Vec3 target;
        double geo = 0, area = 0, normal = 0;
        double cheap() const { return geo + area + normal; }
    };

    FAQEMCost(const Mesh& m, FAQEMParams p) : p_(p) { build(m); }

    Eval evaluate(const Mesh& m, Vertex u, Vertex v, Edge e) const {
        Eval ev;
        const bool bu = on_border(m, u), bv = on_border(m, v);
        const bool be = on_border(m, e);
        if (bu && bv && !be) return ev;  // collapsing would pinch the open boundary

        const auto iu = static_cast<std::size_t>(u), iv = static_cast<std::size_t>(v);
        Quadric Q = gf_[iu] + gf_[iv];
        Quadric QA = area_[iu] + area_[iv];
        Quadric Qc = Q + QA.scaled(p_.w_area);

        Vec3 pu = to_vec(m.point(u)), pv = to_vec(m.point(v)), mid = (pu + pv) * 0.5;
        Vec3 cands[4];
        int n = 0;
        if (bu && !bv)      cands[n++] = pu;
        else if (bv && !bu) cands[n++] = pv;
        else {
            cands[n++] = pu; cands[n++] = pv; cands[n++] = mid;
            if (!bu && !bv) {
                Vec3 opt;
                if (Qc.optimum(opt) && norm(opt - mid) <= p_.max_move_factor * norm(pv - pu))
                    cands[n++] = opt;
            }
        }
        int best = 0;
        double best_c = Qc.eval(cands[0]);
        for (int k = 1; k < n; ++k) {
            double c = Qc.eval(cands[k]);
            if (c < best_c) { best_c = c; best = k; }
        }
        ev.target = cands[best];
        ev.geo = std::max(0.0, Q.eval(ev.target));
        ev.area = p_.w_area * std::max(0.0, QA.eval(ev.target));
        RingStats rs = analyze_collapse(m, u, v, ev.target, p_.min_triangle_quality);
        ev.normal = p_.w_normal * rs.normal_penalty;
        ev.valid = std::isfinite(ev.geo + ev.area + ev.normal);
        return ev;
    }

    // Called after a collapse: the surviving vertex inherits both quadrics.
    void merge(Vertex u, Vertex v, Vertex kept) {
        auto iu = static_cast<std::size_t>(u), iv = static_cast<std::size_t>(v);
        Quadric g = gf_[iu] + gf_[iv], a = area_[iu] + area_[iv];
        gf_[static_cast<std::size_t>(kept)] = g;
        area_[static_cast<std::size_t>(kept)] = a;
    }

private:
    FAQEMParams p_;
    std::vector<Quadric> gf_;    // Q_gf = Q_base + Q_boundary(sharp features)
    std::vector<Quadric> area_;  // Q^A    (open-boundary constraint, edge-length weighted)

    void build(const Mesh& m) {
        gf_.assign(m.num_vertices(), Quadric());
        area_.assign(m.num_vertices(), Quadric());

        // mean edge length, used to make the boundary weight dimensionless
        double sum = 0; std::size_t cnt = 0;
        for (Edge e : m.edges()) {
            Halfedge h = m.halfedge(e, 0);
            sum += norm(to_vec(m.point(m.target(h))) - to_vec(m.point(m.source(h))));
            ++cnt;
        }
        const double mean_len = cnt ? sum / cnt : 1.0;

        // Q_base: one plane per incident face
        for (Face f : m.faces()) {
            Halfedge h = m.halfedge(f);
            Vertex a = m.source(h), b = m.target(h), c = m.target(m.next(h));
            Vec3 pa = to_vec(m.point(a));
            Vec3 n = normalized(cross(to_vec(m.point(b)) - pa, to_vec(m.point(c)) - pa));
            if (norm2(n) == 0) continue;
            Quadric q = Quadric::from_plane(n, -dot(n, pa), 1.0);
            for (Vertex w : {a, b, c}) gf_[static_cast<std::size_t>(w)] += q;
        }

        const double cos_sharp = std::cos(p_.sharp_angle_deg * M_PI / 180.0);
        for (Edge e : m.edges()) {
            Halfedge h = m.halfedge(e, 0), ho = m.opposite(h);
            Face f1 = m.face(h), f2 = m.face(ho);
            Vertex a = m.source(h), b = m.target(h);
            Vec3 pa = to_vec(m.point(a)), pb = to_vec(m.point(b));
            double len = norm(pb - pa);
            if (len < 1e-30) continue;
            Vec3 dir = (pb - pa) * (1.0 / len);

            if (f1 == Mesh::null_face() || f2 == Mesh::null_face()) {
                // open boundary -> Q^A : plane through the edge, perpendicular to its face
                Face f = (f1 != Mesh::null_face()) ? f1 : f2;
                Vec3 mn = normalized(cross(dir, face_normal_unit(m, f)));
                if (norm2(mn) == 0) continue;
                Quadric q = Quadric::from_plane(mn, -dot(mn, pa), len / mean_len);
                area_[static_cast<std::size_t>(a)] += q;
                area_[static_cast<std::size_t>(b)] += q;
            } else {
                // sharp feature edge -> Q_boundary part of Q_gf
                Vec3 n1 = face_normal_unit(m, f1), n2 = face_normal_unit(m, f2);
                if (dot(n1, n2) < cos_sharp) {
                    for (const Vec3& n : {n1, n2}) {
                        Vec3 mn = normalized(cross(dir, n));
                        if (norm2(mn) == 0) continue;
                        Quadric q = Quadric::from_plane(mn, -dot(mn, pa), p_.w_feature);
                        gf_[static_cast<std::size_t>(a)] += q;
                        gf_[static_cast<std::size_t>(b)] += q;
                    }
                }
            }
        }
    }
};

// ------------------------------------------------------------ extra terms
struct CollapseContext {
    const Mesh& mesh;
    Vertex u, v;
    Vec3 target;  // proposed v'
};

class ICostTerm {
public:
    virtual ~ICostTerm() = default;
    virtual const char* name() const = 0;
    // Extra cost (>= 0, squared-length units) of this collapse. May be expensive:
    // the Simplifier only calls it lazily, for edges that reach the front of the queue.
    virtual double delta(const CollapseContext& ctx) const = 0;
    // Bookkeeping after a collapse (u and v merged into `kept`).
    virtual void on_collapse(Vertex /*u*/, Vertex /*v*/, Vertex /*kept*/) {}
};

// ------------------------------------------------------------ silhouette term
// E_sil(M) = sum_i alpha_i * sum_{p in silhouette vertices of view i} d_i(p)^2,
// d_i(p) = distance (in world units) from the projection of p to the boundary of mask S_i.
// A vertex is a silhouette vertex in view i if it touches both front- and back-facing
// triangles. dE_sil(e) = E_sil(M') - E_sil(M) is evaluated locally: the two endpoints
// are replaced by v'. Occlusion is ignored; negative deltas are clamped to 0.
class SilhouetteCost : public ICostTerm {
public:
    SilhouetteCost(const Mesh& m, const CameraSet& cams, const SilhouetteSet& sil)
        : cams_(cams), sil_(sil), n_(cams.cams.size()) {
        flags_.assign(m.num_vertices() * n_, 0);
        for (Face f : m.faces()) {
            Halfedge h = m.halfedge(f);
            Vertex a = m.source(h), b = m.target(h), c = m.target(m.next(h));
            Vec3 pa = to_vec(m.point(a)), pb = to_vec(m.point(b)), pc = to_vec(m.point(c));
            Vec3 nrm = cross(pb - pa, pc - pa);
            Vec3 ctr = (pa + pb + pc) * (1.0 / 3.0);
            for (std::size_t i = 0; i < n_; ++i) {
                if (!active(i)) continue;
                std::uint8_t bit = dot(nrm, cams_.cams[i].center - ctr) > 0 ? 1 : 2;
                for (Vertex w : {a, b, c}) flags_[static_cast<std::size_t>(w) * n_ + i] |= bit;
            }
        }
    }

    const char* name() const override { return "silhouette"; }

    double delta(const CollapseContext& c) const override {
        const auto iu = static_cast<std::size_t>(c.u), iv = static_cast<std::size_t>(c.v);
        const Vec3 pu = to_vec(c.mesh.point(c.u)), pv = to_vec(c.mesh.point(c.v));
        double total = 0;
        for (std::size_t i = 0; i < n_; ++i) {
            if (!active(i)) continue;
            std::uint8_t fu = flags_[iu * n_ + i], fv = flags_[iv * n_ + i];
            bool su = fu == 3, sv = fv == 3, sn = (fu | fv) == 3;
            if (!su && !sv && !sn) continue;
            double before = 0, after = 0;
            if (su) before += err(i, pu);
            if (sv) before += err(i, pv);
            if (sn) after = err(i, c.target);
            total += cams_.cams[i].weight * (after - before);
        }
        return std::max(0.0, total);
    }

    void on_collapse(Vertex u, Vertex v, Vertex kept) override {
        const auto iu = static_cast<std::size_t>(u), iv = static_cast<std::size_t>(v);
        const auto ik = static_cast<std::size_t>(kept);
        for (std::size_t i = 0; i < n_; ++i) {
            std::uint8_t f = flags_[iu * n_ + i] | flags_[iv * n_ + i];
            flags_[ik * n_ + i] = f;
        }
    }

private:
    const CameraSet& cams_;
    const SilhouetteSet& sil_;
    std::size_t n_;
    std::vector<std::uint8_t> flags_;  // per vertex & view: bit0 = touches front face, bit1 = back face

    bool active(std::size_t i) const {
        return i < sil_.valid.size() && sil_.valid[i] && cams_.cams[i].weight > 0;
    }
    // squared world-space distance of p's projection to the mask boundary
    double err(std::size_t i, const Vec3& p) const {
        const Camera& cam = cams_.cams[i];
        double u, v, depth, phi;
        if (!cam.project(p, u, v, depth)) return 0;
        if (!sil_.masks[i].sample(u, v, cam.width, cam.height, phi)) return 0;
        double w = phi * depth / (0.5 * (cam.fx + cam.fy));
        return w * w;
    }
};

}  // namespace lp