#pragma once
// Collapse history, kept so appearance (texture/UV) can later be transferred from the
// original mesh to the simplified one by successive mapping.
// Indices refer to the mesh as given to the Simplifier (before collect_garbage()).
#include "types.h"

#include <fstream>
#include <string>
#include <vector>

namespace lp {

struct CollapseRecord {
    std::uint32_t kept = 0, removed = 0;
    Vec3 kept_before, removed_before, new_pos;
    std::uint32_t removed_faces[2] = {0, 0};
    int n_removed_faces = 0;
    double cost = 0;
};

class CollapseHistory {
public:
    std::vector<CollapseRecord> records;

    void add(const CollapseRecord& r) { records.push_back(r); }

    // text: kept removed  kept_before(xyz) removed_before(xyz) new(xyz)  cost  n_faces face_ids...
    bool save(const std::string& path) const {
        std::ofstream out(path);
        if (!out) return false;
        out.precision(9);
        for (const auto& r : records) {
            out << r.kept << ' ' << r.removed << ' '
                << r.kept_before.x << ' ' << r.kept_before.y << ' ' << r.kept_before.z << ' '
                << r.removed_before.x << ' ' << r.removed_before.y << ' ' << r.removed_before.z << ' '
                << r.new_pos.x << ' ' << r.new_pos.y << ' ' << r.new_pos.z << ' '
                << r.cost << ' ' << r.n_removed_faces;
            for (int k = 0; k < r.n_removed_faces; ++k) out << ' ' << r.removed_faces[k];
            out << '\n';
        }
        return true;
    }
};

}  // namespace lp