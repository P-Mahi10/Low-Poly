// lowpoly: FA-QEM (+ optional multi-view silhouette term) mesh simplification.
//
//   lowpoly in.ply out.ply --faces 5000
//           [--colmap DIR_WITH_cameras.txt_images.txt --masks MASK_DIR]
//           [--lambda-sil 1.0] [--sharp-angle 40] [--min-component N] [--history file.txt]
#include "simplifier.h"

#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

namespace PMP = CGAL::Polygon_mesh_processing;
using namespace lp;

struct Args {
    std::string in, out, colmap, masks, history;
    std::size_t faces = 5000, min_component = 0;
    double lambda_sil = 1.0, sharp_angle = 40.0;
};

static bool parse(int argc, char** argv, Args& a) {
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto val = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::cerr << "missing value for " << name << "\n"; std::exit(1); }
            return argv[++i];
        };
        if (s == "--faces") a.faces = std::stoull(val("--faces"));
        else if (s == "--colmap") a.colmap = val("--colmap");
        else if (s == "--masks") a.masks = val("--masks");
        else if (s == "--lambda-sil") a.lambda_sil = std::stod(val("--lambda-sil"));
        else if (s == "--sharp-angle") a.sharp_angle = std::stod(val("--sharp-angle"));
        else if (s == "--min-component") a.min_component = std::stoull(val("--min-component"));
        else if (s == "--history") a.history = val("--history");
        else pos.push_back(s);
    }
    if (pos.size() < 2) return false;
    a.in = pos[0]; a.out = pos[1];
    return true;
}

static bool load_and_clean(const std::string& path, Mesh& mesh, std::size_t min_component) {
    if (!CGAL::IO::read_polygon_mesh(path, mesh) || mesh.is_empty()) {
        std::cerr << "Cannot read mesh " << path << "\n";
        return false;
    }
    if (!CGAL::is_triangle_mesh(mesh)) PMP::triangulate_faces(mesh);

    PMP::remove_isolated_vertices(mesh);
    // drop tiny floating components (Poisson leaves some); keep big ones
    std::size_t thr = min_component ? min_component : std::max<std::size_t>(20, mesh.number_of_faces() / 1000);
    std::size_t removed = PMP::keep_large_connected_components(mesh, thr);
    if (removed) std::cout << "Removed " << removed << " small components (< " << thr << " faces)\n";
    PMP::remove_isolated_vertices(mesh);
    mesh.collect_garbage();  // dense indices from here on
    return !mesh.is_empty();
}

int main(int argc, char** argv) {
    Args args;
    if (!parse(argc, argv, args)) {
        std::cerr << "usage: " << argv[0] << " in.ply out.ply --faces N [--colmap DIR --masks DIR]\n"
                  << "        [--lambda-sil X] [--sharp-angle DEG] [--min-component N] [--history FILE]\n";
        return 1;
    }
    auto t0 = std::chrono::steady_clock::now();

    Mesh mesh;
    if (!load_and_clean(args.in, mesh, args.min_component)) return 1;
    std::cout << "Loaded " << mesh.number_of_vertices() << " vertices, " << mesh.number_of_faces() << " faces\n";

    SimplifyOptions opt;
    opt.target_faces = args.faces;
    opt.faqem.sharp_angle_deg = args.sharp_angle;
    opt.lambda_sil = args.lambda_sil;

    // ---- ML hook (not used yet) ----
    // opt.ml.vertex_importance = ...;   // size == mesh.num_vertices(), >1 keeps detail
    // opt.ml.protected_vertices = ...;  // edges touching these never collapse
    // cameras.cams[i].weight = ...;     // per-view silhouette importance alpha_i

    CameraSet cameras;
    SilhouetteSet silhouettes;
    Simplifier simp(mesh, opt);

    if (!args.colmap.empty() && !args.masks.empty()) {
#ifdef HAVE_OPENCV
        if (!cameras.load_colmap(args.colmap)) { std::cerr << "Failed to read COLMAP model\n"; return 1; }
        std::size_t n = silhouettes.load(args.masks, cameras);
        std::cout << "Cameras: " << cameras.cams.size() << ", masks loaded: " << n << "\n";
        if (n > 0) {
            simp.add_term(std::make_unique<SilhouetteCost>(mesh, cameras, silhouettes), opt.lambda_sil);
            std::cout << "Silhouette term ON (lambda = " << opt.lambda_sil << ")\n";
        } else {
            std::cerr << "No masks matched camera image names; silhouette term OFF\n";
        }
#else
        std::cerr << "Built without OpenCV: silhouette term OFF\n";
#endif
    } else {
        std::cout << "Silhouette term OFF (no --colmap/--masks)\n";
    }

    simp.run();

    mesh.collect_garbage();
    if (!args.history.empty() && !simp.history().save(args.history))
        std::cerr << "Could not write history " << args.history << "\n";
    if (!CGAL::IO::write_polygon_mesh(args.out, mesh, CGAL::parameters::stream_precision(17))) {
        std::cerr << "Cannot write " << args.out << "\n";
        return 1;
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "Wrote " << args.out << " (" << mesh.number_of_vertices() << " vertices, "
              << mesh.number_of_faces() << " faces) in " << secs << " s\n";
    return 0;
}