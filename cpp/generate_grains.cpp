#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <omp.h>

constexpr double OLIVINE_DENSITY = 3222.0; // kg/m^3
constexpr double PI = 3.14159265358979323846;

struct Point3D {
    double x, y, z;
};

struct Face {
    int v0, v1, v2;
};

struct Grain {
    double dmax_class_um;
    double max_width_um;
    double area_m2;
    double volume_m3;
    double mass_kg;
    double a_spec_m2_kg;
};

Point3D cross(const Point3D& a, const Point3D& b) {
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

Point3D sub(const Point3D& a, const Point3D& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

double norm(const Point3D& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

void compute_convex_hull_3d(const std::vector<Point3D>& pts, std::vector<Face>& hull_faces, std::vector<int>& hull_vertex_indices) {
    int n = static_cast<int>(pts.size());
    if (n < 4) return;
    std::vector<bool> in_hull(n, false);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = j + 1; k < n; ++k) {
                Point3D v1 = sub(pts[j], pts[i]);
                Point3D v2 = sub(pts[k], pts[i]);
                Point3D normal = cross(v1, v2);

                if (norm(normal) < 1e-9) continue;

                int pos = 0, neg = 0;
                for (int l = 0; l < n; ++l) {
                    if (l == i || l == j || l == k) continue;
                    Point3D v3 = sub(pts[l], pts[i]);
                    double dot = normal.x * v3.x + normal.y * v3.y + normal.z * v3.z;
                    if (dot > 1e-7) pos++;
                    if (dot < -1e-7) neg++;
                }

                if (pos == 0 || neg == 0) {
                    hull_faces.push_back({ i, j, k });
                    in_hull[i] = true; in_hull[j] = true; in_hull[k] = true;
                }
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        if (in_hull[i]) hull_vertex_indices.push_back(i);
    }
}

void compute_hull_properties(const std::vector<Point3D>& pts, const std::vector<Face>& faces, double& area, double& volume, double& max_dist) {
    area = 0.0;
    volume = 0.0;
    max_dist = 0.0;

    for (size_t i = 0; i < pts.size(); ++i) {
        for (size_t j = i + 1; j < pts.size(); ++j) {
            double d = norm(sub(pts[i], pts[j]));
            if (d > max_dist) max_dist = d;
        }
    }

    for (const auto& f : faces) {
        Point3D p0 = pts[f.v0], p1 = pts[f.v1], p2 = pts[f.v2];
        Point3D v1 = sub(p1, p0), v2 = sub(p2, p0);
        area += 0.5 * norm(cross(v1, v2));

        double scalar_triple = p0.x * (p1.y * p2.z - p1.z * p2.y)
                             - p0.y * (p1.x * p2.z - p1.z * p2.x)
                             + p0.z * (p1.x * p2.y - p1.y * p2.x);
        volume += scalar_triple;
    }
    volume = std::abs(volume) / 6.0;
}

Grain generate_grain(double target_dmax_um, double axis_a, double axis_b, double axis_c, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);
    std::uniform_real_distribution<double> dist_angle(0.0, 2.0 * PI);
    std::uniform_real_distribution<double> dist_cos(-1.0, 1.0);
    std::uniform_int_distribution<int> dist_n(11, 19);

    int num_points = dist_n(rng);
    std::vector<Point3D> points;
    points.reserve(num_points);

    for (int i = 0; i < num_points; ++i) {
        double r = std::cbrt(dist_01(rng));
        double phi = dist_angle(rng);
        double costheta = dist_cos(rng);
        double sintheta = std::sqrt(1.0 - costheta * costheta);

        double x = axis_a * r * sintheta * std::cos(phi);
        double y = axis_b * r * sintheta * std::sin(phi);
        double z = axis_c * r * costheta;

        points.push_back({ x, y, z });
    }

    std::vector<Face> faces;
    std::vector<int> vertex_indices;
    compute_convex_hull_3d(points, faces, vertex_indices);

    double raw_area = 0.0, raw_volume = 0.0, raw_dmax = 0.0;
    compute_hull_properties(points, faces, raw_area, raw_volume, raw_dmax);

    double S = (raw_dmax > 0.0) ? (target_dmax_um / raw_dmax) : 1.0;

    double area_um2 = (S * S) * raw_area;
    double volume_um3 = (S * S * S) * raw_volume;

    double area_m2 = area_um2 * 1e-12;
    double volume_m3 = volume_um3 * 1e-18;
    double mass_kg = OLIVINE_DENSITY * volume_m3;

    return Grain{ target_dmax_um, target_dmax_um, area_m2, volume_m3, mass_kg, area_m2 / mass_kg };
}

void export_grains_to_json(const std::string& filename, const std::vector<Grain>& grains, double total_mass, double a, double b, double c) {
    std::ofstream out(filename);
    if (!out.is_open()) return;

    double total_area = 0.0;
    for (const auto& g : grains) total_area += g.area_m2;

    out << std::scientific << std::setprecision(8);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"aspect_ratio\": {\"a\": " << a << ", \"b\": " << b << ", \"c\": " << c << "},\n";
    out << "    \"total_mass_kg\": " << total_mass << ",\n";
    out << "    \"total_area_m2\": " << total_area << ",\n";
    out << "    \"bulk_a_spec_m2_kg\": " << (total_area / total_mass) << ",\n";
    out << "    \"num_grains\": " << grains.size() << "\n";
    out << "  },\n";
    out << "  \"grains\": [\n";
    for (size_t i = 0; i < grains.size(); ++i) {
        const auto& g = grains[i]; // <-- Hier stond per ongeluk een ':' i.p.v. '='
        out << "    {\"dmax_class_um\": " << g.dmax_class_um << ", \"area_m2\": " << g.area_m2 
            << ", \"volume_m3\": " << g.volume_m3 << ", \"mass_kg\": " << g.mass_kg 
            << ", \"a_spec_m2_kg\": " << g.a_spec_m2_kg << "}" << (i + 1 < grains.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
}

int main(int argc, char* argv[]) {
    double axis_a = 1.5, axis_b = 1.2, axis_c = 1.0;
    std::string out_filename = "initial_grains.json";

    if (argc >= 4) {
        axis_a = std::stod(argv[1]);
        axis_b = std::stod(argv[2]);
        axis_c = std::stod(argv[3]);
    }
    if (argc >= 5) {
        out_filename = argv[4];
    }

    struct SizeClass { double dmax_um; double target_mass_kg; };
    std::vector<SizeClass> classes = {
        {2000.0, 55.0}, {1410.0, 109.0}, {1000.0, 144.0}, {707.0, 243.0},
        {420.0, 199.0}, {250.0, 102.0}, {150.0, 49.0}, {88.0, 33.0},
        {50.0, 37.0}, {16.0, 13.0}, {8.0, 16.0}
    };

    std::vector<Grain> all_grains;
    double total_sim_mass = 0.0;

    for (size_t c = 0; c < classes.size(); ++c) {
        const auto& cls = classes[c];
        std::vector<Grain> class_grains;
        double accumulated_mass = 0.0;

        // Genereer representatieve deeltjes per klasse (max 100 per klasse voor de Monte Carlo benadering)
        constexpr int GRAINS_PER_CLASS = 100;
        
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            std::mt19937_64 rng(1337 + c * 100 + thread_id);
            
            #pragma omp for
            for (int i = 0; i < GRAINS_PER_CLASS; ++i) {
                Grain g = generate_grain(cls.dmax_um, axis_a, axis_b, axis_c, rng);
                #pragma omp critical
                {
                    class_grains.push_back(g);
                    accumulated_mass += g.mass_kg;
                }
            }
        }

        // Schaal de deeltjes per klasse exact op naar de doelmassa van Tabel 3.1
        if (accumulated_mass > 0) {
            double scale_factor = cls.target_mass_kg / accumulated_mass;
            for (auto& g : class_grains) {
                g.mass_kg *= scale_factor;
                g.area_m2 *= scale_factor; // Behoudt exact dezelfde A_spec!
            }
        }

        all_grains.insert(all_grains.end(), class_grains.begin(), class_grains.end());
        total_sim_mass += cls.target_mass_kg;
    }

    export_grains_to_json(out_filename, all_grains, total_sim_mass, axis_a, axis_b, axis_c);
    return 0;
}