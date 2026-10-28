import os
import subprocess
import numpy as np
from forward_model import OlivineDissolutionModel


class ParticleFilterDA:
    def __init__(
        self,
        n_particles: int = 100,
        sigma_obs: float = 2.0,            # Section 3.2: sigma_obs = 2.0 kg
        a_lw: float = 0.98,                  # Section 3.4: a_LW in (0.95, 0.99)
        resample_threshold_ratio: float = 0.8, # Section 3.4: N_eff <= 0.8 N
        cpp_generator_path: str = "./cpp/generate_grains.exe"
    ):
        self.N = n_particles
        self.sigma_obs = sigma_obs
        self.a_lw = a_lw
        self.resample_threshold = resample_threshold_ratio * self.N
        self.cpp_path = cpp_generator_path
        
        self.weights = np.ones(self.N) / self.N
        self.particles = []

    def initialize_ensemble(self, a_prior_mean: float = 1.3, a_prior_std: float = 0.4):
        """Section 3.1: Initialize augmented state vector z_0 = [M_tot, A_tot, a]^T."""
        self.particles = []
        for i in range(self.N):
            # Sample primary aspect ratio a >= 1.0
            a_val = max(1.01, np.random.normal(a_prior_mean, a_prior_std))
            # Section 3.1: 1.0 <= b <= a, c = 1.0
            b_val = np.random.uniform(1.0, a_val)
            c_val = 1.0
            
            json_file = f"particle_{i}.json"
            subprocess.run(
                [self.cpp_path, str(a_val), str(b_val), str(c_val), json_file],
                check=True,
                stdout=subprocess.DEVNULL
            )
            
            model = OlivineDissolutionModel(json_file)
            state = model.get_state_vector()
            
            self.particles.append({
                "id": i,
                "a": a_val,
                "b": b_val,
                "c": c_val,
                "json_path": json_file,
                "model": model,
                "M_total": state["m_tot"],
                "A_spec": state["A_spec_aggregate"]
            })
            
        self.weights = np.ones(self.N) / self.N

    def update_weights(self, observation_y: float):
        """Section 3.2 & 3.3: Gaussian Likelihood evaluation f(y_t | z_t)."""
        simulated_masses = np.array([p["M_total"] for p in self.particles])
        
        # Log-likelihood formulation for numerical stability
        log_likelihoods = -0.5 * ((observation_y - simulated_masses) / self.sigma_obs) ** 2
        max_log_lik = np.max(log_likelihoods)
        
        unnormalized = self.weights * np.exp(log_likelihoods - max_log_lik)
        self.weights = unnormalized / np.sum(unnormalized)

    def compute_n_eff(self) -> float:
        """Section 3.4: Effective ensemble size N_eff = 1 / sum(w_i^2)."""
        return 1.0 / np.sum(self.weights ** 2)

    def apply_liu_west_propagation(self, dt_seconds: float):
        """
        Section 3.4: Continuous Liu-West kernel smoothing on parameter 'a'
        and forward step propagation.
        """
        a_vals = np.array([p["a"] for p in self.particles])
        
        # Weighted sample mean and variance (Section 3.4)
        weighted_mean = np.sum(self.weights * a_vals)
        weighted_var = np.sum(self.weights * (a_vals - weighted_mean) ** 2)
        std_noise = np.sqrt((1.0 - self.a_lw ** 2) * max(weighted_var, 1e-4))

        for p in self.particles:
            old_a = p["a"]
            
            # 1. Kernel smoothing parameter evolution (Eq. 13)
            mu_a = self.a_lw * old_a + (1.0 - self.a_lw) * weighted_mean
            new_a = max(1.01, np.random.normal(mu_a, std_noise))
            
            p["a"] = new_a
            p["b"] = min(p["b"], new_a)  # Enforce 1.0 <= b <= a
            
            # 2. Update reactive surface area based on aspect ratio hypothesis change
            # Surface area scales with primary aspect ratio modification
            area_factor = (new_a / old_a) ** (1.0 / 3.0)
            p["model"].current_areas *= area_factor

            # 3. Step kinetic dissolution forward by dt_seconds (Section 3.2)
            p["model"].step(dt_seconds)
            
            state = p["model"].get_state_vector()
            p["M_total"] = state["m_tot"]
            p["A_spec"] = state["A_spec_aggregate"]

    def stochastic_universal_resampling(self):
        """Section 3.4: Low-variance Stochastic Universal Resampling (SUR)."""
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0
        
        ancestor_indices = np.zeros(self.N, dtype=int)
        step = 1.0 / self.N
        start = np.random.uniform(0, step)
        pointers = start + np.arange(self.N) * step
        
        i, j = 0, 0
        while i < self.N:
            if pointers[i] < cdf[j]:
                ancestor_indices[i] = j
                i += 1
            else:
                j += 1

        # Deep clone particles from ancestors
        new_particles = []
        for new_idx, parent_idx in enumerate(ancestor_indices):
            parent = self.particles[parent_idx]
            
            model = OlivineDissolutionModel(parent["json_path"])
            model.current_masses = parent["model"].current_masses.copy()
            model.current_areas = parent["model"].current_areas.copy()
            
            state = model.get_state_vector()
            new_particles.append({
                "id": new_idx,
                "a": parent["a"],
                "b": parent["b"],
                "c": parent["c"],
                "json_path": parent["json_path"],
                "model": model,
                "M_total": state["m_tot"],
                "A_spec": state["A_spec_aggregate"]
            })

        self.particles = new_particles
        self.weights = np.ones(self.N) / self.N  # Reset equal weights (1/N)

    def resample_if_needed(self):
        """Section 3.4: Conditional resampling when N_eff <= 0.8 N."""
        if self.compute_n_eff() <= self.resample_threshold:
            self.stochastic_universal_resampling()
            return True
        return False

    def clean_temp_files(self):
        for p in self.particles:
            if os.path.exists(p["json_path"]):
                try:
                    os.remove(p["json_path"])
                except OSError:
                    pass