#pragma once
// Reference silhouettes S_i. Each binary mask is turned into a signed distance
// field phi (pixels; < 0 inside, > 0 outside, 0 on the boundary) so that the
// discrepancy D(.) of a projected point is one bilinear lookup.
#include "camera.h"

#include <filesystem>
#include <iostream>
#include <vector>

#ifdef HAVE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace lp {

class SilhouetteMask {
public:
    bool load(const std::string& path) {
#ifdef HAVE_OPENCV
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) return false;
        cv::Mat bin, inv, inside, outside;
        cv::threshold(img, bin, 127, 255, cv::THRESH_BINARY);
        cv::bitwise_not(bin, inv);
        cv::distanceTransform(bin, inside, 2, 5);   // fg pixel -> dist to bg
        cv::distanceTransform(inv, outside, 2, 5);  // bg pixel -> dist to fg
        cv::Mat phi = outside - inside;
        w_ = phi.cols; h_ = phi.rows;
        phi_.assign(phi.begin<float>(), phi.end<float>());
        return true;
#else
        (void)path;
        return false;
#endif
    }

    // (u,v): camera pixel coords of a camera of size cam_w x cam_h.
    // Returns signed distance in *camera* pixels; false if outside the image.
    bool sample(double u, double v, double cam_w, double cam_h, double& phi_cam_px) const {
        if (phi_.empty()) return false;
        double sx = w_ / cam_w, sy = h_ / cam_h;
        double x = u * sx - 0.5, y = v * sy - 0.5;
        if (x < 0 || y < 0 || x > w_ - 1 || y > h_ - 1) return false;
        int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
        int x1 = std::min(x0 + 1, w_ - 1), y1 = std::min(y0 + 1, h_ - 1);
        double fx = x - x0, fy = y - y0;
        double p00 = phi_[(size_t)y0 * w_ + x0], p10 = phi_[(size_t)y0 * w_ + x1];
        double p01 = phi_[(size_t)y1 * w_ + x0], p11 = phi_[(size_t)y1 * w_ + x1];
        double p = (p00 * (1 - fx) + p10 * fx) * (1 - fy) + (p01 * (1 - fx) + p11 * fx) * fy;
        phi_cam_px = p / (0.5 * (sx + sy));
        return true;
    }

private:
    int w_ = 0, h_ = 0;
    std::vector<float> phi_;
};

// One mask per camera (index-aligned with CameraSet::cams).
class SilhouetteSet {
public:
    std::vector<SilhouetteMask> masks;
    std::vector<char> valid;

    // Looks for <dir>/<image name>, <name>.png, <stem>.png, <stem>.jpg.
    // Cameras without a mask get weight 0 (they are ignored).
    std::size_t load(const std::filesystem::path& dir, CameraSet& cams) {
        namespace fs = std::filesystem;
        masks.assign(cams.cams.size(), SilhouetteMask());
        valid.assign(cams.cams.size(), 0);
        std::size_t ok = 0;
        for (std::size_t i = 0; i < cams.cams.size(); ++i) {
            const std::string& name = cams.cams[i].name;
            std::string stem = fs::path(name).stem().string();
            fs::path candidates[] = {dir / name, dir / (name + ".png"),
                                     dir / (stem + ".png"), dir / (stem + ".jpg")};
            for (auto& c : candidates) {
                if (fs::exists(c) && masks[i].load(c.string())) { valid[i] = 1; ++ok; break; }
            }
            if (!valid[i]) {
                cams.cams[i].weight = 0.0;
                std::cerr << "  no mask for " << name << " (view ignored)\n";
            }
        }
        return ok;
    }
};

}  // namespace lp