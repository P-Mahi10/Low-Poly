#pragma once
// The edge-collapse loop (flowchart section 3):
//   candidate edges -> cost (FA-QEM [+ silhouette]) -> priority queue -> lowest cost
//   -> validity check -> collapse -> record history -> refresh local costs -> repeat.
//
// The silhouette term is expensive, so it is evaluated lazily: edges enter the queue
// with their cheap FA-QEM cost (a lower bound, since the extra terms are >= 0). When an
// edge reaches the front, its extra cost is added and it is re-queued; it is only
// collapsed once it is at the front with its full cost.
#include "cost.h"
#include "history.h"
#include "validator.h"

#include <CGAL/boost/graph/Euler_operations.h>

#include <iostream>
#include <memory>
#include <queue>

namespace lp {

struct SimplifyOptions {
    std::size_t target_faces = 5000;
    FAQEMParams faqem;
    ValidatorParams validator;
    double lambda_sil = 1.0;  // weight of the silhouette term
    MLHints ml;               // <-- ML hook: assign importance / protected sets here later
    bool verbose = true;
};

class Simplifier {
public:
    struct Stats {
        std::size_t collapses = 0, rejected = 0, stale = 0;
        std::size_t faces_before = 0, faces_after = 0;
    };

    Simplifier(Mesh& mesh, SimplifyOptions opt)
        : mesh_(mesh), opt_(std::move(opt)), faqem_(mesh, opt_.faqem),
          validator_(opt_.validator, opt_.ml) {}

    // Extra cost term (e.g. SilhouetteCost) with weight lambda.
    void add_term(std::unique_ptr<ICostTerm> term, double weight) {
        terms_.push_back({std::move(term), weight});
    }

    Stats run() {
        stats_ = Stats();
        stats_.faces_before = mesh_.number_of_faces();
        version_.assign(mesh_.num_edges(), 0);
        stamp_.assign(mesh_.num_edges(), 0);

        for (Edge e : mesh_.edges()) push_edge(e);

        std::size_t next_report = 0;
        while (mesh_.number_of_faces() > opt_.target_faces && !pq_.empty()) {
            Entry top = pq_.top();
            pq_.pop();
            Edge e(top.edge);
            if (mesh_.is_removed(e) || version_[top.edge] != top.version) { ++stats_.stale; continue; }

            Halfedge h = mesh_.halfedge(e, 0);
            Vertex u = mesh_.source(h), v = mesh_.target(h);

            // lazy evaluation of the expensive terms
            if (!top.full) {
                double extra = 0;
                CollapseContext ctx{mesh_, u, v, top.target};
                for (auto& t : terms_) extra += t.weight * t.term->delta(ctx);
                top.full = true;
                if (extra > 0) {
                    top.cost += opt_.ml.importance(u, v) * extra;
                    if (!pq_.empty() && top.cost > pq_.top().cost) { pq_.push(top); continue; }
                }
            }

            auto res = validator_.check(mesh_, e, u, v, top.target);
            if (res != CollapseValidator::Result::Ok) { ++stats_.rejected; continue; }

            collapse(e, h, u, v, top);

            if (opt_.verbose && mesh_.number_of_faces() + 5000 <= next_report) next_report = 0;
            if (opt_.verbose && stats_.collapses % 20000 == 0)
                std::cout << "  faces: " << mesh_.number_of_faces() << "\r" << std::flush;
        }
        stats_.faces_after = mesh_.number_of_faces();
        if (opt_.verbose)
            std::cout << "\nSimplifier: " << stats_.faces_before << " -> " << stats_.faces_after
                      << " faces (" << stats_.collapses << " collapses, " << stats_.rejected
                      << " rejected)\n";
        if (stats_.faces_after > opt_.target_faces)
            std::cerr << "Warning: stopped above target; no valid collapses left "
                         "(relax min_normal_cos / min_triangle_quality, or check protected sets).\n";
        return stats_;
    }

    const CollapseHistory& history() const { return history_; }

private:
    struct Entry {
        double cost;
        std::uint32_t edge, version;
        bool full;
        Vec3 target;
    };
    struct Cmp {
        bool operator()(const Entry& a, const Entry& b) const { return a.cost > b.cost; }
    };
    struct Term { std::unique_ptr<ICostTerm> term; double weight; };

    Mesh& mesh_;
    SimplifyOptions opt_;
    FAQEMCost faqem_;
    CollapseValidator validator_;
    std::vector<Term> terms_;
    std::priority_queue<Entry, std::vector<Entry>, Cmp> pq_;
    std::vector<std::uint32_t> version_, stamp_;
    std::uint32_t stamp_gen_ = 0;
    CollapseHistory history_;
    Stats stats_;

    void push_edge(Edge e) {
        Halfedge h = mesh_.halfedge(e, 0);
        Vertex u = mesh_.source(h), v = mesh_.target(h);
        if (opt_.ml.is_protected(u, v, e)) return;
        auto ev = faqem_.evaluate(mesh_, u, v, e);
        if (!ev.valid) return;
        double cost = ev.cheap() * opt_.ml.importance(u, v);
        pq_.push({cost, static_cast<std::uint32_t>(e), version_[static_cast<std::size_t>(e)],
                  terms_.empty(), ev.target});
    }

    void collapse(Edge e, Halfedge h, Vertex u, Vertex v, const Entry& top) {
        CollapseRecord rec;
        rec.removed_before = to_vec(mesh_.point(u));
        rec.kept_before = to_vec(mesh_.point(v));
        rec.new_pos = top.target;
        rec.cost = top.cost;
        for (Face f : {mesh_.face(h), mesh_.face(mesh_.opposite(h))})
            if (f != Mesh::null_face()) rec.removed_faces[rec.n_removed_faces++] = static_cast<std::uint32_t>(f);

        Vertex kept = CGAL::Euler::collapse_edge(e, mesh_);
        Vertex removed = (kept == u) ? v : u;
        mesh_.point(kept) = to_point(top.target);

        rec.kept = static_cast<std::uint32_t>(kept);
        rec.removed = static_cast<std::uint32_t>(removed);
        if (kept == u) std::swap(rec.kept_before, rec.removed_before);  // keep names truthful
        history_.add(rec);

        faqem_.merge(u, v, kept);
        for (auto& t : terms_) t.term->on_collapse(u, v, kept);
        ++stats_.collapses;

        // refresh every edge around the surviving vertex and its neighbours
        std::vector<Vertex> ring{kept};
        for (Vertex w : CGAL::vertices_around_target(kept, mesh_)) ring.push_back(w);
        ++stamp_gen_;
        for (Vertex w : ring)
            for (Halfedge hh : CGAL::halfedges_around_target(w, mesh_)) {
                Edge ee = mesh_.edge(hh);
                auto ie = static_cast<std::size_t>(ee);
                if (stamp_[ie] == stamp_gen_) continue;
                stamp_[ie] = stamp_gen_;
                ++version_[ie];
                push_edge(ee);
            }
    }
};

}  // namespace lp