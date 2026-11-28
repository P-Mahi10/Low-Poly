#pragma once
// Robustness checks run on every candidate collapse right before it is performed.
#include "cost.h"

#include <CGAL/boost/graph/Euler_operations.h>

namespace lp {

struct ValidatorParams {
    double min_normal_cos = 0.1;        // reject if any neighbouring face normal turns by more than ~84 deg
    double min_triangle_quality = 0.02; // reject new sliver triangles
};

class CollapseValidator {
public:
    enum class Result { Ok, Protected, Topology, NormalFlip, Degenerate, Quality };

    CollapseValidator(ValidatorParams p, const MLHints& ml) : p_(p), ml_(ml) {}

    Result check(const Mesh& m, Edge e, Vertex u, Vertex v, const Vec3& target) const {
        if (ml_.is_protected(u, v, e)) return Result::Protected;
        if (on_border(m, u) && on_border(m, v) && !on_border(m, e))
            return Result::Topology;
        if (!CGAL::Euler::does_satisfy_link_condition(e, m)) return Result::Topology;

        RingStats rs = analyze_collapse(m, u, v, target, p_.min_triangle_quality);
        if (rs.degenerate) return Result::Degenerate;
        if (rs.min_cos < p_.min_normal_cos) return Result::NormalFlip;
        if (rs.bad_quality) return Result::Quality;
        return Result::Ok;
    }

    static const char* to_string(Result r) {
        switch (r) {
            case Result::Ok: return "ok";
            case Result::Protected: return "protected";
            case Result::Topology: return "topology";
            case Result::NormalFlip: return "normal-flip";
            case Result::Degenerate: return "degenerate";
            case Result::Quality: return "quality";
        }
        return "?";
    }

private:
    ValidatorParams p_;
    const MLHints& ml_;
};

}  // namespace lp