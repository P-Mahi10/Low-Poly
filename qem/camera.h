#pragma once
// COLMAP text-model cameras (cameras.txt + images.txt), pinhole projection.
#include "types.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace lp {

struct Camera {
    int id = 0;
    std::string name;
    int width = 0, height = 0;
    double fx = 1, fy = 1, cx = 0, cy = 0;
    double R[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // world -> camera
    double t[3] = {0, 0, 0};
    Vec3 center;          // camera centre in world coordinates
    double weight = 1.0;  // alpha_i, importance of this view (ML can change it)

    Vec3 to_cam(const Vec3& p) const {
        return {R[0] * p.x + R[1] * p.y + R[2] * p.z + t[0],
                R[3] * p.x + R[4] * p.y + R[5] * p.z + t[1],
                R[6] * p.x + R[7] * p.y + R[8] * p.z + t[2]};
    }
    // pixel coordinates follow COLMAP: pixel centre of the first pixel is (0.5, 0.5)
    bool project(const Vec3& p, double& u, double& v, double& depth) const {
        Vec3 c = to_cam(p);
        if (c.z <= 1e-9) return false;
        depth = c.z;
        u = fx * c.x / c.z + cx;
        v = fy * c.y / c.z + cy;
        return true;
    }
};

class CameraSet {
public:
    std::vector<Camera> cams;

    // dir must contain cameras.txt and images.txt
    // (colmap model_converter --input_path sparse/0 --output_path txt --output_type TXT)
    bool load_colmap(const std::filesystem::path& dir) {
        std::map<int, Camera> intr;
        if (!read_cameras(dir / "cameras.txt", intr)) return false;
        return read_images(dir / "images.txt", intr);
    }

private:
    static bool next_data_line(std::istream& in, std::string& line) {
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            return true;
        }
        return false;
    }

    static bool read_cameras(const std::filesystem::path& p, std::map<int, Camera>& out) {
        std::ifstream in(p);
        if (!in) { std::cerr << "Cannot open " << p << "\n"; return false; }
        std::string line;
        while (next_data_line(in, line)) {
            std::istringstream ss(line);
            Camera c;
            std::string model;
            ss >> c.id >> model >> c.width >> c.height;
            std::vector<double> prm;
            double x;
            while (ss >> x) prm.push_back(x);

            bool single_focal = (model == "SIMPLE_PINHOLE" || model == "SIMPLE_RADIAL" ||
                                 model == "RADIAL");
            if (single_focal) {
                if (prm.size() < 3) return false;
                c.fx = c.fy = prm[0]; c.cx = prm[1]; c.cy = prm[2];
            } else {
                if (prm.size() < 4) return false;
                c.fx = prm[0]; c.fy = prm[1]; c.cx = prm[2]; c.cy = prm[3];
            }
            if (model != "PINHOLE" && model != "SIMPLE_PINHOLE")
                std::cerr << "Warning: camera model " << model
                          << " has distortion; ignored (use undistorted images/model).\n";
            out[c.id] = c;
        }
        return !out.empty();
    }

    bool read_images(const std::filesystem::path& p, const std::map<int, Camera>& intr) {
        std::ifstream in(p);
        if (!in) { std::cerr << "Cannot open " << p << "\n"; return false; }
        std::string line;
        while (next_data_line(in, line)) {  // line 1: pose
            std::istringstream ss(line);
            int image_id, cam_id;
            double qw, qx, qy, qz, tx, ty, tz;
            std::string name;
            ss >> image_id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cam_id >> name;
            std::string points_line;
            std::getline(in, points_line);  // line 2: 2D points (may be empty), skip

            auto it = intr.find(cam_id);
            if (it == intr.end()) continue;
            Camera c = it->second;
            c.name = name;

            double n = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
            qw /= n; qx /= n; qy /= n; qz /= n;
            c.R[0] = 1 - 2 * (qy * qy + qz * qz); c.R[1] = 2 * (qx * qy - qz * qw); c.R[2] = 2 * (qx * qz + qy * qw);
            c.R[3] = 2 * (qx * qy + qz * qw); c.R[4] = 1 - 2 * (qx * qx + qz * qz); c.R[5] = 2 * (qy * qz - qx * qw);
            c.R[6] = 2 * (qx * qz - qy * qw); c.R[7] = 2 * (qy * qz + qx * qw); c.R[8] = 1 - 2 * (qx * qx + qy * qy);
            c.t[0] = tx; c.t[1] = ty; c.t[2] = tz;
            // centre = -R^T t
            c.center = {-(c.R[0] * tx + c.R[3] * ty + c.R[6] * tz),
                        -(c.R[1] * tx + c.R[4] * ty + c.R[7] * tz),
                        -(c.R[2] * tx + c.R[5] * ty + c.R[8] * tz)};
            cams.push_back(c);
        }
        return !cams.empty();
    }
};

}  // namespace lp