import numpy as np
import matplotlib
matplotlib.use('Agg')  # Voorkomt dat Tkinter/Tcl gebruikt wordt

import matplotlib.pyplot as plt
from forward_model import OlivineDissolutionModel
from particle_filter import ParticleFilterDA


def run_synthetic_experiment():
    # 1. Setup Ground Truth Target (e.g., true aspect ratio a = 2.0, b = 1.4, c = 1.0)
    true_a, true_b, true_c = 2.0, 1.4, 1.0
    ground_truth_json = "ground_truth.json"
    
    # Generate ground truth initial grains via C++ binary
    import subprocess
    subprocess.run(["cpp/generate_grains.exe", str(true_a), str(true_b), str(true_c), ground_truth_json], check=True)
    
    gt_model = OlivineDissolutionModel(ground_truth_json)
    
    # 2. Simulation Setup
    total_years = 20.0
    dt_years = 1.0
    dt_seconds = dt_years * 365.25 * 86400.0
    n_steps = int(total_years / dt_years) + 1
    sigma_obs = 2.0  # 2.0 kg Gaussian observation noise

    # Generate synthetic observations y_t
    t_years = np.linspace(0, total_years, n_steps)
    true_masses = []
    
    for _ in range(n_steps):
        true_masses.append(gt_model.get_state_vector()["m_tot"])
        gt_model.step(dt_seconds)
        
    true_masses = np.array(true_masses)
    # Add measurement noise epsilon ~ N(0, sigma_obs^2)
    observations = true_masses + np.random.normal(0.0, sigma_obs, size=n_steps)

    # 3. Initialize Particle Filter Data Assimilation
    pf = ParticleFilterDA(
        n_particles=100,         # Increased particle count for smoother posteriors
        sigma_obs=2.0,           # Matches Eq (7) in methodology
        a_lw=0.98,               # Discount factor a_LW in (0.95, 0.99)
        cpp_generator_path="./cpp/generate_grains.exe"
    )
    pf.initialize_ensemble(a_prior_mean=1.3, a_prior_std=0.4)

    

    # Parameter tracking arrays
    estimated_a_mean = []
    estimated_a_std = []

    print("Starting Sequential Monte Carlo Assimilation Loop...")
    for t_idx in range(n_steps):
        y_obs = observations[t_idx]
        
        # 1. Update weights based on current observation
        pf.update_weights(y_obs)
        
        # 2. Compute posterior statistics
        a_particles = np.array([p["a"] for p in pf.particles])
        weighted_mean_a = np.sum(pf.weights * a_particles)
        weighted_var_a = np.sum(pf.weights * (a_particles - weighted_mean_a) ** 2)
        
        estimated_a_mean.append(weighted_mean_a)
        estimated_a_std.append(np.sqrt(weighted_var_a))

        print(f"Year {t_years[t_idx]:4.1f} | Obs Mass: {y_obs:7.2f} kg | Est. Aspect Ratio 'a': {weighted_mean_a:.4f} ± {np.sqrt(weighted_var_a):.4f} | N_eff: {pf.compute_n_eff():.1f}")

        # 3. Resample if degenerate
        pf.resample_if_needed()
        
        # 4. Propagate ensemble and apply Liu-West smoothing to parameters
        if t_idx < n_steps - 1:
            pf.apply_liu_west_propagation(dt_seconds)

    # Cleanup temporary particle JSON files
    pf.clean_temp_files()

    # 4. Plot Parameter Recovery Progress
    plt.figure(figsize=(8, 5))
    plt.axhline(true_a, color='r', linestyle='--', label=f'True $a = {true_a}$')
    plt.errorbar(t_years, estimated_a_mean, yerr=estimated_a_std, fmt='-o', color='b', ecolor='gray', capsize=3, label='Assimilated Posterior $a$')
    plt.xlabel("Time (Years)")
    plt.ylabel("Primary Aspect Ratio $a$")
    plt.title("Parameter Estimation via SMC Particle Filter (Liu-West Filter)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("poging5000.png")


if __name__ == "__main__":
    run_synthetic_experiment()