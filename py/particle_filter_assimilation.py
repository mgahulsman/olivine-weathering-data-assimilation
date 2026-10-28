import os
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class ParticleFilterAssimilation:
    def __init__(self, num_particles=40, true_aspect_ratio=(1.8, 1.3, 1.0), 
                 obs_noise_std=2.0, a_LW=0.98):
        self.num_particles = num_particles
        self.true_aspect_ratio = true_aspect_ratio
        self.obs_noise_std = obs_noise_std
        self.a_LW = a_LW

        # 1. Ground Truth Generation
        print("[1] Generating Ground Truth Trajectory...")
        m_gt, a_gt = self.generate_and_get_initial_state(true_aspect_ratio, "data/ground_truth.json")
        self.times, self.true_masses = self.run_ground_truth_sim(m_gt, a_gt, total_years=20.0, dt_years=1.0)
        
        # Synthetic observations: y_t = M_total_true(t) + \epsilon_t
        self.observations = self.true_masses + np.random.normal(0, self.obs_noise_std, size=len(self.true_masses))

        # 2. Prior Ensemble Initialization
        print(f"[2] Initializing Ensemble (N = {num_particles})...")
        self.particles_a = np.random.uniform(1.1, 2.5, size=num_particles)
        
        self.particles_mass = np.zeros(num_particles)
        self.particles_area = np.zeros(num_particles)
        
        # Store initial geometry baselines per particle for scaling: A_spec_0 = A0 / M0
        self.particles_aspec_0 = np.zeros(num_particles)

        for i in range(num_particles):
            m0, a0 = self.generate_and_get_initial_state(
                (self.particles_a[i], np.random.uniform(1.0, max(1.001, self.particles_a[i])), 1.0),
                f"data/particle_{i}.json"
            )
            self.particles_mass[i] = m0
            self.particles_area[i] = a0
            self.particles_aspec_0[i] = a0 / m0

    def generate_cpp_grains(self, aspect_ratio, output_json):
        a, b, c = aspect_ratio
        os.makedirs("data", exist_ok=True)
        project_root = Path(__file__).resolve().parent.parent
        exe_path = project_root / "cpp" / ("generate_grains.exe" if os.name == 'nt' else "generate_grains")
        cmd = f'"{exe_path}" {a} {b} {c} "{output_json}"'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

    def generate_and_get_initial_state(self, aspect_ratio, output_json):
        self.generate_cpp_grains(aspect_ratio, output_json)
        with open(output_json, 'r') as f:
            data = json.load(f)
        return data["metadata"]["total_mass_kg"], data["metadata"]["total_area_m2"]

    def forward_step(self, mass_t, area_t, dt):
        """Explicit numerical integration matching Equations 3 & 4."""
        if mass_t <= 0:
            return 0.0, 0.0

        M_MOLAR = 0.1407
        R_RATE = 6.69e-10

        # Eq 3: Numerical Integration
        dm = M_MOLAR * area_t * R_RATE * dt
        mass_next = max(0.0, mass_t - dm)

        # Eq 4: Isometric Surface Area Update
        if mass_t > 0 and mass_next > 0:
            area_next = area_t * (mass_next / mass_t) ** (2.0 / 3.0)
        else:
            area_next = 0.0

        return mass_next, area_next

    def run_ground_truth_sim(self, m0, a0, total_years=20.0, dt_years=1.0):
        n_steps = int(total_years / dt_years) + 1
        times = np.linspace(0, total_years, n_steps)
        dt_sec = dt_years * 365.25 * 86400.0
        
        masses = np.zeros(n_steps)
        m_curr, a_curr = m0, a0
        masses[0] = m_curr

        for t in range(1, n_steps):
            m_curr, a_curr = self.forward_step(m_curr, a_curr, dt_sec)
            masses[t] = m_curr

        return times, masses

    def stochastic_universal_resampling(self, weights):
        N = len(weights)
        cumulative_sum = np.cumsum(weights)
        positions = (np.arange(N) + np.random.uniform(0, 1)) / N
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def run_assimilation(self):
        print("\n[3] Executing Recursive Sequential Monte Carlo Particle Filter...")
        dt_seconds = 1.0 * 365.25 * 86400.0
        n_years = len(self.times)

        mass_trajectories = np.zeros((self.num_particles, n_years))
        a_posterior_history = np.zeros((self.num_particles, n_years))
        weights_history = np.zeros((self.num_particles, n_years))

        # Initial state recording
        mass_trajectories[:, 0] = self.particles_mass
        a_posterior_history[:, 0] = self.particles_a
        
        weights = np.ones(self.num_particles) / self.num_particles
        weights_history[:, 0] = weights

        for t_idx in range(1, n_years):
            obs_t = self.observations[t_idx]

            # ---------------------------------------------------------
            # 1. Forward Integration (Physics Step)
            # ---------------------------------------------------------
            for i in range(self.num_particles):
                m_next, a_next = self.forward_step(
                    self.particles_mass[i], self.particles_area[i], dt_seconds
                )
                self.particles_mass[i] = m_next
                self.particles_area[i] = a_next
                mass_trajectories[i, t_idx] = m_next

            # ---------------------------------------------------------
            # 2. Gaussian Likelihood Evaluation
            # ---------------------------------------------------------
            likelihoods = np.exp(-0.5 * ((obs_t - self.particles_mass) / self.obs_noise_std) ** 2)
            weights = weights * likelihoods
            weights_sum = np.sum(weights)
            if weights_sum > 0:
                weights /= weights_sum
            else:
                weights = np.ones(self.num_particles) / self.num_particles

            # ---------------------------------------------------------
            # 3. Resampling & Continuous Liu-West Kernel Perturbation
            # ---------------------------------------------------------
            N_eff = 1.0 / np.sum(weights ** 2)
            if N_eff <= 0.8 * self.num_particles:
                print(f"  [Resample Triggered] t = {t_idx} years | N_eff = {N_eff:.2f}")
                resample_indices = self.stochastic_universal_resampling(weights)
                
                # Duplicate states and parameters
                self.particles_a = self.particles_a[resample_indices]
                self.particles_mass = self.particles_mass[resample_indices]
                self.particles_area = self.particles_area[resample_indices]
                self.particles_aspec_0 = self.particles_aspec_0[resample_indices]

                # --- Liu & West Kernel Smoothing ---
                bar_theta = np.mean(self.particles_a)
                V_param = np.var(self.particles_a)

                # Maintain minimum variance buffer
                V_param = max(V_param, 1e-3)

                kernel_mean = self.a_LW * self.particles_a + (1.0 - self.a_LW) * bar_theta
                kernel_std = np.sqrt((1.0 - self.a_LW ** 2) * V_param)
                
                self.particles_a = np.clip(
                    np.random.normal(kernel_mean, kernel_std), a_min=1.01, a_max=3.0
                )

                # --- Corrected Area Update ---
                # Recalculate baseline specific area for updated parameter 'a'
                for i in range(self.num_particles):
                    m0_new, a0_new = self.generate_and_get_initial_state(
                        (self.particles_a[i], np.random.uniform(1.0, max(1.001, self.particles_a[i])), 1.0),
                        f"data/particle_{i}.json"
                    )
                    self.particles_aspec_0[i] = a0_new / m0_new
                    
                    # Update current area using current mass and updated baseline A_spec_0:
                    # A(t) = A_spec_0 * M0 * (M(t)/M0)^(2/3) = A_spec_0 * M0^(1/3) * M(t)^(2/3)
                    self.particles_area[i] = self.particles_aspec_0[i] * (m0_new ** (1.0 / 3.0)) * (self.particles_mass[i] ** (2.0 / 3.0))

                weights = np.ones(self.num_particles) / self.num_particles

            a_posterior_history[:, t_idx] = self.particles_a
            weights_history[:, t_idx] = weights

        self.plot_results(mass_trajectories, a_posterior_history, weights_history)

    def plot_results(self, mass_trajectories, a_posterior_history, weights_history):
        a_mean_trajectory = np.sum(a_posterior_history * weights_history, axis=0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        for i in range(self.num_particles):
            ax1.plot(self.times, mass_trajectories[i, :], color='gray', alpha=0.25, linewidth=1)
        ax1.plot(self.times, self.true_masses, 'k--', label='Ground Truth Trajectory', linewidth=2)
        ax1.scatter(self.times, self.observations, color='red', s=25, zorder=5, label=r'Observations $y_t$ ($\sigma_{\mathrm{obs}}=2.0$ kg)')
        ax1.set_ylabel("Aggregate Mass $M_{\mathrm{total}}$ (kg)")
        ax1.set_title("Sequential Monte Carlo (Particle Filter): Ensemble State Tracking")
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2.plot(self.times, a_mean_trajectory, color='blue', linestyle='-', linewidth=2, label=r'Posterior Mean Parameter $\bar{a}_t$')
        ax2.axhline(self.true_aspect_ratio[0], color='red', linestyle='--', linewidth=1.5, label=f'True Aspect Ratio $a = {self.true_aspect_ratio[0]}$')
        
        p16 = np.percentile(a_posterior_history, 16, axis=0)
        p84 = np.percentile(a_posterior_history, 84, axis=0)
        ax2.fill_between(self.times, p16, p84, color='blue', alpha=0.15, label=r'$1\sigma$ Parameter Ensemble Spread')

        ax2.set_xlabel("Time (Years)")
        ax2.set_ylabel(r"Primary Aspect Ratio Parameter $a$")
        ax2.set_title("Parameter Estimation via Liu-West Kernel Smoothing")
        ax2.legend()
        ax2.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.savefig("particle_filter_assimilation_fixed.png", dpi=300)
        print("\n[Done] Corrected plot generated and saved as 'particle_filter_assimilation_fixed.png'.")


if __name__ == "__main__":
    pf = ParticleFilterAssimilation(
        num_particles=40,
        true_aspect_ratio=(1.8, 1.3, 1.0),
        obs_noise_std=2.0,
        a_LW=0.98
    )
    pf.run_assimilation()