import os
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from forward_model import OlivineDissolutionModel


class ParticleSmootherAssimilation:
    def __init__(self, num_particles=40, true_aspect_ratio=(1.8, 1.3, 1.0), obs_noise_std=2.0, sigma_jitter=0.02):
        self.num_particles = num_particles
        self.true_aspect_ratio = true_aspect_ratio
        self.obs_noise_std = obs_noise_std
        self.sigma_jitter = sigma_jitter

        # 1. Generate Ground Truth Data
        print("[1] Generating Ground Truth trajectory...")
        self.generate_cpp_grains(true_aspect_ratio, "data/ground_truth.json")
        gt_model = OlivineDissolutionModel("data/ground_truth.json")
        self.times, self.true_masses, _ = gt_model.run_simulation(total_years=20.0, dt_years=1.0)
        
        # Synthetic observations with additive noise
        self.observations = self.true_masses + np.random.normal(0, self.obs_noise_std, size=len(self.true_masses))

        # 2. Initialize Ensemble Parameters (Prior)
        print(f"[2] Initializing {num_particles} particles...")
        self.particles_params = []
        for i in range(num_particles):
            a_prior = np.random.uniform(1.1, 2.5)
            b_prior = np.random.uniform(1.0, a_prior)
            c_prior = 1.0
            self.particles_params.append([a_prior, b_prior, c_prior])

    def generate_cpp_grains(self, aspect_ratio, output_json):
        a, b, c = aspect_ratio
        os.makedirs("data", exist_ok=True)
        project_root = Path(__file__).resolve().parent.parent
        exe_path = project_root / "cpp" / ("generate_grains.exe" if os.name == 'nt' else "generate_grains")
        cmd = f'"{exe_path}" {a} {b} {c} {output_json}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

    def likelihood(self, predicted_mass, observed_mass):
        diff = predicted_mass - observed_mass
        return np.exp(-0.5 * (diff / self.obs_noise_std) ** 2) + 1e-300

    def systematic_resample(self, weights):
        """Systematic resampling routine."""
        N = len(weights)
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def rebuild_model_and_catchup(self, param_triple, target_t_idx, dt_seconds):
        """Rebuilds a model with updated parameters and integrates from t=0 to target_t_idx."""
        json_path = f"data/temp_p_{np.random.randint(0, int(1e8))}.json"
        self.generate_cpp_grains(param_triple, json_path)
        model = OlivineDissolutionModel(json_path)
        
        # Catch-up integration from t=0 to current step
        for _ in range(target_t_idx):
            model.step(dt_seconds)
            
        if os.path.exists(json_path):
            os.remove(json_path)
            
        return model

    def run_smoother_simulation(self):
        print("\n[3] Starting Forward Filtering (SIR) & Backward Smoothing (FFBS)...")
        dt_seconds = 1.0 * 365.25 * 86400.0
        n_years = len(self.times)

        # Storage matrices
        mass_trajectories = np.zeros((self.num_particles, n_years))
        a_param_trajectories = np.zeros((self.num_particles, n_years))
        forward_weights = np.zeros((self.num_particles, n_years))

        # Active models initialization
        models = []
        for i in range(self.num_particles):
            json_path = f"data/particle_{i}.json"
            self.generate_cpp_grains(self.particles_params[i], json_path)
            models.append(OlivineDissolutionModel(json_path))
            
            mass_trajectories[i, 0] = models[i].get_state_vector()["m_tot"]
            a_param_trajectories[i, 0] = self.particles_params[i][0]
            
        forward_weights[:, 0] = 1.0 / self.num_particles

        # -------------------------------------------------------------
        # STEP 1: FORWARD PASS (Sequential Importance Resampling Filter)
        # -------------------------------------------------------------
        for t_idx in range(1, n_years):
            obs_t = self.observations[t_idx]
            
            # 1a. Forecast Step
            for i in range(self.num_particles):
                models[i].step(dt_seconds)
                mass_trajectories[i, t_idx] = models[i].get_state_vector()["m_tot"]
                a_param_trajectories[i, t_idx] = self.particles_params[i][0]

            # 1b. Measurement Update (Likelihood)
            weights = np.zeros(self.num_particles)
            for i in range(self.num_particles):
                weights[i] = forward_weights[i, t_idx - 1] * self.likelihood(mass_trajectories[i, t_idx], obs_t)

            weights /= np.sum(weights)
            forward_weights[:, t_idx] = weights

            # 1c. Resampling & Jittering Check (N_eff < N / 2)
            n_eff = 1.0 / np.sum(weights ** 2)
            if n_eff < self.num_particles / 2.0 and t_idx < n_years - 1:
                print(f"  -> Resampling triggered at t = {t_idx} years (N_eff = {n_eff:.2f})")
                resample_indices = self.systematic_resample(weights)
                
                new_params = []
                new_models = []
                for i in range(self.num_particles):
                    parent_idx = resample_indices[i]
                    base_a, base_b, _ = self.particles_params[parent_idx]
                    
                    # Apply Gaussian jitter to parameter a
                    new_a = max(1.01, base_a + np.random.normal(0, self.sigma_jitter))
                    new_b = min(base_b, new_a)  # Maintain physical constraint b <= a
                    new_param = [new_a, new_b, 1.0]
                    
                    new_params.append(new_param)
                    # Catch-up integration from t=0 to t_idx
                    new_models.append(self.rebuild_model_and_catchup(new_param, t_idx, dt_seconds))

                self.particles_params = new_params
                models = new_models
                
                # Reset weights post-resampling
                forward_weights[:, t_idx] = 1.0 / self.num_particles

        # -------------------------------------------------------------
        # STEP 2: BACKWARD PASS (Batch Smoother with Gaussian Kernel)
        # -------------------------------------------------------------
        smoothed_weights = np.zeros_like(forward_weights)
        smoothed_weights[:, -1] = forward_weights[:, -1]
        sigma_kernel = 0.05 

        for t_idx in range(n_years - 2, -1, -1):
            for i in range(self.num_particles):
                a_i = a_param_trajectories[i, t_idx]
                a_next = a_param_trajectories[:, t_idx + 1]
                
                # Transition density
                transition_dens = np.exp(-0.5 * ((a_i - a_next) / sigma_kernel) ** 2)
                
                denom = np.sum(forward_weights[:, t_idx] * transition_dens)
                if denom > 0:
                    integral = np.sum((smoothed_weights[:, t_idx + 1] * transition_dens) / denom)
                else:
                    integral = 1.0
                    
                smoothed_weights[i, t_idx] = forward_weights[i, t_idx] * integral

            sum_sw = np.sum(smoothed_weights[:, t_idx])
            if sum_sw > 0:
                smoothed_weights[:, t_idx] /= sum_sw
            else:
                smoothed_weights[:, t_idx] = 1.0 / self.num_particles

        # -------------------------------------------------------------
        # PLOTTING RESULTS
        # -------------------------------------------------------------
        filter_a_mean = np.sum(a_param_trajectories * forward_weights, axis=0)
        smoother_a_mean = np.sum(a_param_trajectories * smoothed_weights, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for i in range(self.num_particles):
            ax1.plot(self.times, mass_trajectories[i, :], color='gray', alpha=0.2, linewidth=1)
        ax1.plot(self.times, self.true_masses, 'k--', label='True Ground Truth', linewidth=2)
        ax1.scatter(self.times, self.observations, color='red', s=20, zorder=5, label='Observations')
        ax1.set_ylabel("Total Mass (kg)")
        ax1.set_title("Particle Smoother: Ensemble Mass Trajectories")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.times, filter_a_mean, 'b--', label='Particle Filter Estimate (Online SIR)', linewidth=2)
        ax2.plot(self.times, smoother_a_mean, 'g-', label='Particle Smoother Estimate (Batch FFBS)', linewidth=2.5)
        ax2.axhline(self.true_aspect_ratio[0], color='red', linestyle=':', label=f'True a = {self.true_aspect_ratio[0]}')
        ax2.set_xlabel("Time (Years)")
        ax2.set_ylabel("Estimated a-parameter")
        ax2.set_title("Comparison: Sequential SIR Filtering vs. Batch Smoothing")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("smoottthhh.png")


if __name__ == "__main__":
    smoother = ParticleSmootherAssimilation(
        num_particles=40,
        true_aspect_ratio=(1.8, 1.3, 1.0),
        obs_noise_std=2.0
    )
    smoother.run_smoother_simulation()