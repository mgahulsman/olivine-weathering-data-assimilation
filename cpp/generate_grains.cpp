#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <omp.h>

constexpr double OLIVINE_DENSITY = 3222.0;
constexpr double PI = 3.14159265358979323846;

struct Grain {
    double dmax_class_um;
    double max_width_um;
    double area_m2;
    double volume_m3;
    double mass_kg;
    double a_spec_m2_kg;
};

Grain generate_grain_fast(double target_dmax_um, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist_01(0.0, 1.0);
    std::uniform_real_distribution<double> dist_angle(0.0, 2.0 * PI);
    std::uniform_real_distribution<double> dist_cos(-1.0, 1.0);
    std::uniform_int_distribution<int> dist_n(11, 19);

    int num_points = dist_n(rng);

    double max_dist_sq = 0.0;
    for (int i = 0; i < num_points; ++i) {
        double r = std::cbrt(dist_01(rng));
        double phi = dist_angle(rng);
        double costheta = dist_cos(rng);
        double sintheta = std::sqrt(1.0 - costheta * costheta);

        double x = 1.5 * r * sintheta * std::cos(phi);
        double y = 1.2 * r * sintheta * std::sin(phi);
        double z = 1.0 * r * costheta;

        double dist_sq = x * x + y * y + z * z;
        if (dist_sq > max_dist_sq) max_dist_sq = dist_sq;
    }

    double raw_dmax = 2.0 * std::sqrt(max_dist_sq);
    double S = target_dmax_um / raw_dmax;

    double volume_um3 = (4.0 / 3.0) * PI * (1.5 * S) * (1.2 * S) * (1.0 * S) * 0.5;
    double area_um2 = 4.0 * PI * std::pow((1.5 * S + 1.2 * S + 1.0 * S) / 3.0, 2.0) * 0.6;

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

int main() {
    constexpr double mass_goal = 1000.0; // Target aggregate mass in kg
    constexpr double target_dmax_um = 2000.0;
    
    std::vector<Grain> all_grains;
    all_grains.reserve(1'000'000);

    double total_mass_sim = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_print_time = start_time;

    std::cout << "Starting parallel particle generation across " << omp_get_max_threads() << " CPU threads...\n" << std::flush;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::mt19937_64 rng(1337 + thread_id);

        std::vector<Grain> local_grains;
        local_grains.reserve(10'000);
        double local_mass = 0.0;

        while (total_mass_sim < mass_goal) {
            Grain g = generate_grain_fast(target_dmax_um, rng);
            local_grains.push_back(g);
            local_mass += g.mass_kg;

            // Sync every 5,000 grains to reduce thread lock contention
            if (local_grains.size() >= 5000) {
                #pragma omp critical
                {
                    total_mass_sim += local_mass;
                    all_grains.insert(all_grains.end(), local_grains.begin(), local_grains.end());

                    // Throttle progress printing to once every 100 milliseconds
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::milli> dt = now - last_print_time;
                    if (dt.count() > 100.0 || total_mass_sim >= mass_goal) {
                        last_print_time = now;
                        double progress = std::min(100.0, (total_mass_sim / mass_goal) * 100.0);
                        std::cout << "\r[Progress] " << std::fixed << std::setprecision(2) << progress << "% | "
                                  << total_mass_sim << " / " << mass_goal << " kg | Grains: " << all_grains.size() 
                                  << "       " << std::flush;
                    }
                }
                local_mass = 0.0;
                local_grains.clear();
            }
        }

        #pragma omp critical
        {
            total_mass_sim += local_mass;
            all_grains.insert(all_grains.end(), local_grains.begin(), local_grains.end());
            
            double progress = std::min(100.0, (total_mass_sim / mass_goal) * 100.0);
            std::cout << "\r[Progress] " << std::fixed << std::setprecision(2) << progress << "% | "
                      << total_mass_sim << " / " << mass_goal << " kg | Grains: " << all_grains.size() 
                      << "       " << std::flush;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\n\nGeneration completed in: " << elapsed.count() << " seconds!\n";
    std::cout << "Total particles generated: " << all_grains.size() << "\n";
    std::cout << "Total mass accumulated: " << total_mass_sim << " kg\n";

    return 0;
}