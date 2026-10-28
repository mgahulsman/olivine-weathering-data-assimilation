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

constexpr double OLIVINE_DENSITY = 3222.0;
constexpr double PI = 3.14159265358979323846;

struct Point3D {
    double x, y, z;
};

struct Grain {
    double dmax_class_um;
    double max_width_um;
    double area_m2;
    double volume_m3;
    double mass_kg;
    double a_spec_m2_kg;
};

// -----------------------------------------------------------------------------
// 3D Convex Hull & Geometry Helpers
// -----------------------------------------------------------------------------

struct Face {
    int v0, v1, v2;
};

// Helper: Vector cross product
Point3D cross(const Point3D& a, const Point3D& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Helper: Vector subtraction
Point3D sub(const Point3D& a, const Point3D& b) {
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

// Helper: Vector length
double norm(const Point3D& v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

// Compute 3D Convex Hull for small point clouds
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

                if (norm(normal) < 1e-9) continue; // Collinear

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
                    in_hull[i] = true;
                    in_hull[j] = true;
                    in_hull[k] = true;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        if (in_hull[i]) hull_vertex_indices.push_back(i);
    }
}

// Computes exact surface area and volume of the Convex Hull
void compute_hull_properties(const std::vector<Point3D>& pts, const std::vector<Face>& faces, double& area, double& volume) {
    area = 0.0;
    volume = 0.0;

    for (const auto& f : faces) {
        Point3D p0 = pts[f.v0];
        Point3D p1 = pts[f.v1];
        Point3D p2 = pts[f.v2];

        Point3D v1 = sub(p1, p0);
        Point3D v2 = sub(p2, p0);
        area += 0.5 * norm(cross(v1, v2));

        double scalar_triple = p0.x * (p1.y * p2.z - p1.z * p2.y)
                             - p0.y * (p1.x * p2.z - p1.z * p2.x)
                             + p0.z * (p1.x * p2.y - p1.y * p2.x);
        volume += scalar_triple;
    }
    volume = std::abs(volume) / 6.0;
}

// -----------------------------------------------------------------------------
// Grain Generation Algorithm
// -----------------------------------------------------------------------------

Grain generate_grain(double target_dmax_um, std::mt19937_64& rng) {
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

        double x = 1.5 * r * sintheta * std::cos(phi);
        double y = 1.2 * r * sintheta * std::sin(phi);
        double z = 1.0 * r * costheta;

        points.push_back({ x, y, z });
    }

    std::vector<Face> faces;
    std::vector<int> vertex_indices;
    compute_convex_hull_3d(points, faces, vertex_indices);

    double max_dist_sq = 0.0;
    for (size_t i = 0; i < vertex_indices.size(); ++i) {
        for (size_t j = i + 1; j < vertex_indices.size(); ++j) {
            Point3D d = sub(points[vertex_indices[i]], points[vertex_indices[j]]);
            double dist_sq = d.x * d.x + d.y * d.y + d.z * d.z;
            if (dist_sq > max_dist_sq) max_dist_sq = dist_sq;
        }
    }
    double raw_dmax = std::sqrt(max_dist_sq);

    double S = target_dmax_um / raw_dmax;
    double raw_area = 0.0, raw_volume = 0.0;
    compute_hull_properties(points, faces, raw_area, raw_volume);

    double area_um2 = (S * S) * raw_area;
    double volume_um3 = (S * S * S) * raw_volume;

    double area_m2 = area_um2 * 1e-12;
    double volume_m3 = volume_um3 * 1e-18;
    double mass_kg = OLIVINE_DENSITY * volume_m3;

    return Grain{
        target_dmax_um,
        target_dmax_um,
        area_m2,
        volume_m3,
        mass_kg,
        area_m2 / mass_kg
    };
}

// -----------------------------------------------------------------------------
// JSON Export Helper Function
// -----------------------------------------------------------------------------

void export_grains_to_json(const std::string& filename, const std::vector<Grain>& grains, double total_mass, double target_dmax_um) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error opening file for writing JSON: " << filename << "\n";
        return;
    }

    double total_area = 0.0;
    for (const auto& g : grains) {
        total_area += g.area_m2;
    }
    double bulk_a_spec = total_area / total_mass;

    out << std::scientific << std::setprecision(8);
    out << "{\n";
    out << "  \"metadata\": {\n";
    out << "    \"target_dmax_um\": " << target_dmax_um << ",\n";
    out << "    \"total_mass_kg\": " << total_mass << ",\n";
    out << "    \"total_area_m2\": " << total_area << ",\n";
    out << "    \"bulk_a_spec_m2_kg\": " << bulk_a_spec << ",\n";
    out << "    \"density_kg_m3\": " << OLIVINE_DENSITY << ",\n";
    out << "    \"num_grains\": " << grains.size() << "\n";
    out << "  },\n";
    out << "  \"grains\": [\n";

    for (size_t i = 0; i < grains.size(); ++i) {
        const auto& g = grains[i];
        out << "    {\n";
        out << "      \"dmax_class_um\": " << g.dmax_class_um << ",\n";
        out << "      \"area_m2\": " << g.area_m2 << ",\n";
        out << "      \"volume_m3\": " << g.volume_m3 << ",\n";
        out << "      \"mass_kg\": " << g.mass_kg << ",\n";
        out << "      \"a_spec_m2_kg\": " << g.a_spec_m2_kg << "\n";
        out << "    }" << (i + 1 < grains.size() ? "," : "") << "\n";
    }

    out << "  ]\n";
    out << "}\n";
    out.close();

    std::cout << "\n[JSON Export] Successfully saved " << grains.size() 
              << " grains to file: " << filename << "\n";
}

// -----------------------------------------------------------------------------
// Main Execution
// -----------------------------------------------------------------------------

int main() {
    // Definieer de 11 deeltjesklassen uit Tabel 3.1
    struct SizeClass { double dmax_um; double mass_kg; };
    std::vector<SizeClass> classes = {
        {2000.0, 55.0}, {1410.0, 109.0}, {1000.0, 144.0}, 
        {707.0, 243.0}, {420.0, 199.0}, {250.0, 102.0}, 
        {150.0, 49.0}, {88.0, 33.0}, {50.0, 37.0}, 
        {16.0, 13.0}, {8.0, 16.0}
    };

    std::vector<Grain> all_grains;
    all_grains.reserve(500'000); // Reserveer genoeg geheugen
    
    double total_sim_mass = 0.0;
    std::mt19937_64 rng(1337); // Simpele, vaste seed voor reproduceerbaarheid

    // Loop over elke klasse uit Tabel 3.1
    for (const auto& cls : classes) {
        double current_class_mass = 0.0;
        std::cout << "Genereren van klasse " << cls.dmax_um << " um (Doel: " << cls.mass_kg << " kg)...\n";
        
        while (current_class_mass < cls.mass_kg) {
            Grain g = generate_grain(cls.dmax_um, rng);
            
            // Vermijd overshoot door het laatste deeltje te controleren (optioneel, maar houdt massa accuraat)
            if (current_class_mass + g.mass_kg > cls.mass_kg) {
                // Overslaan of accepteren met lichte afwijking; we accepteren hem hier simpelweg
            }
            
            all_grains.push_back(g);
            current_class_mass += g.mass_kg;
            total_sim_mass += g.mass_kg;
        }
    }

    std::cout << "\nGeneratie voltooid! Totaal gegenereerde massa: " << total_sim_mass << " kg\n";
    export_grains_to_json("initial_grains.json", all_grains, total_sim_mass, 2000.0);

    return 0;
}