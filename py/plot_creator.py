import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Voorkomt dat Tkinter/Tcl gebruikt wordt

import matplotlib.pyplot as plt
from forward_model import OlivineDissolutionModel
from particle_filter import ParticleFilterDA


class PlotCreator:
    def __init__(
        self,
        true_a: float = 2.0,
        true_b: float = 1.4,
        true_c: float = 1.0,
        total_years: float = 20.0,
        dt_years: float = 1.0,
        sigma_obs: float = 2.0,
        n_particles: int = 50,
        cpp_path: str = "./cpp/generate_grains.exe",
        output_dir: str = os.path.join("data", "plots")
    ):
        self.true_a = true_a
        self.true_b = true_b
        self.true_c = true_c
        self.total_years = total_years
        self.dt_years = dt_years
        self.dt_seconds = dt_years * 365.25 * 86400.0
        self.n_steps = int(total_years / dt_years) + 1
        self.t_years = np.linspace(0, total_years, self.n_steps)
        self.sigma_obs = sigma_obs
        self.n_particles = n_particles
        self.cpp_path = cpp_path
        self.gt_json = "ground_truth.json"
        self.output_dir = output_dir

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        self.true_masses = None
        self.true_aspecs = None
        self.observations = None
        self.da_results = None

    def _get_save_path(self, filename: str) -> str:
        """Helper method to ensure files are saved under self.output_dir."""
        return os.path.join(self.output_dir, os.path.basename(filename))

    def generate_ground_truth(self, seed: int = 42):
        """Run synthetic ground truth simulation and construct observation vector."""
        subprocess.run(
            [self.cpp_path, str(self.true_a), str(self.true_b), str(self.true_c), self.gt_json],
            check=True,
            stdout=subprocess.DEVNULL
        )
        gt_model = OlivineDissolutionModel(self.gt_json)

        masses, aspecs = [], []
        for _ in range(self.n_steps):
            state = gt_model.get_state_vector()
            masses.append(state["m_tot"])
            aspecs.append(state["A_spec_aggregate"])
            gt_model.step(self.dt_seconds)

        self.true_masses = np.array(masses)
        self.true_aspecs = np.array(aspecs)

        np.random.seed(seed)
        self.observations = self.true_masses + np.random.normal(0.0, self.sigma_obs, size=self.n_steps)

    def run_da_experiment(self):
        """Execute Particle Filter Data Assimilation loop and log step diagnostics."""
        pf = ParticleFilterDA(
            n_particles=self.n_particles,
            sigma_obs=self.sigma_obs,
            a_lw=0.98,
            cpp_generator_path=self.cpp_path
        )
        # Set prior mean to match ground truth true_a (2.0)
        pf.initialize_ensemble(a_prior_mean=self.true_a, a_prior_std=0.4)

        history = {
            "a_mean": [], "a_std": [],
            "mass_mean": [], "mass_std": [],
            "aspec_mean": [], "aspec_std": [],
            "n_eff": [], "resampled": []
        }

        for t_idx in range(self.n_steps):
            y_obs = self.observations[t_idx]
            pf.update_weights(y_obs)

            a_vals = np.array([p["a"] for p in pf.particles])
            mass_vals = np.array([p["M_total"] for p in pf.particles])
            aspec_vals = np.array([p["A_spec"] for p in pf.particles])
            w = pf.weights

            mean_a = np.sum(w * a_vals)
            std_a = np.sqrt(np.sum(w * (a_vals - mean_a) ** 2))
            mean_m = np.sum(w * mass_vals)
            std_m = np.sqrt(np.sum(w * (mass_vals - mean_m) ** 2))
            mean_as = np.sum(w * aspec_vals)
            std_as = np.sqrt(np.sum(w * (aspec_vals - mean_as) ** 2))

            history["a_mean"].append(mean_a)
            history["a_std"].append(std_a)
            history["mass_mean"].append(mean_m)
            history["mass_std"].append(std_m)
            history["aspec_mean"].append(mean_as)
            history["aspec_std"].append(std_as)
            history["n_eff"].append(pf.compute_n_eff())

            was_resampled = pf.resample_if_needed()
            history["resampled"].append(was_resampled)

            if t_idx < self.n_steps - 1:
                pf.apply_liu_west_propagation(self.dt_seconds)

        pf.clean_temp_files()
        self.da_results = history

    def _generate_spaghetti_plot(self, n_ensemble: int, a_std: float, base_output_path: str):
        """Helper to generate a single spaghetti plot for a given standard deviation."""
        fig, ax = plt.subplots(figsize=(9, 5.5))

        for i in range(n_ensemble):
            a_val = max(self.true_b, np.random.normal(self.true_a, a_std))
            b_val = self.true_b
            c_val = self.true_c

            json_path = f"temp_spag_{i}.json"
            subprocess.run([self.cpp_path, str(a_val), str(b_val), str(c_val), json_path], check=True, stdout=subprocess.DEVNULL)

            model = OlivineDissolutionModel(json_path)
            mass_traj = []
            for _ in range(self.n_steps):
                mass_traj.append(model.get_state_vector()["m_tot"])
                model.step(self.dt_seconds)

            if os.path.exists(json_path):
                os.remove(json_path)

            ax.plot(self.t_years, mass_traj, color="tab:blue", alpha=0.3, linewidth=1.2, label="Prior Model Trajectory" if i == 0 else "")

        ax.plot(self.t_years, self.true_masses, color="crimson", linewidth=2.5, label=f"True Mass ($a_{{true}}={self.true_a}$)")
        ax.scatter(self.t_years, self.observations, color="black", s=25, zorder=5, label="Noisy Observations ($y_t$)")

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Total Mass $M_{\\mathrm{tot}}$ (kg)")
        ax.set_title(f"Open-Loop Model Forecasts (Spaghetti Plot, $\\sigma_a={a_std}$)")
        ax.grid(True, linestyle=":")
        ax.legend(loc="upper right")
        plt.tight_layout()

        filename, ext = os.path.splitext(base_output_path)
        final_filename = f"{filename}_std_{a_std}{ext}"
        plt.savefig(self._get_save_path(final_filename), dpi=300)
        plt.close()

    def plot_spaghetti_no_da(self, n_ensemble: int = 30, output_path: str = "spaghetti_no_da.png"):
        """Plot un-assimilated open-loop model forecasts against ground truth for various std values."""
        for a_std in [0.4, 2.0]:
            self._generate_spaghetti_plot(n_ensemble=n_ensemble, a_std=a_std, base_output_path=output_path)

    def plot_assimilated_mass_trajectory(self, output_path: str = "assimilated_mass.png"):
        """Plot assimilated state vector mass decay uncertainty vs. ground truth."""
        plt.figure(figsize=(9, 5.5))
        m_mean = np.array(self.da_results["mass_mean"])
        m_std = np.array(self.da_results["mass_std"])

        plt.fill_between(self.t_years, m_mean - 2 * m_std, m_mean + 2 * m_std, color="tab:blue", alpha=0.3, label="Posterior 95% CI")
        plt.plot(self.t_years, m_mean, color="tab:blue", linewidth=2, label="Assimilated Ensemble Mean")
        plt.plot(self.t_years, self.true_masses, color="crimson", linewidth=2.5, linestyle="--", label=f"True Mass ($a_{{true}}={self.true_a}$)")
        plt.scatter(self.t_years, self.observations, color="black", s=25, zorder=5, label="Observations ($y_t$)")

        plt.xlabel("Time (Years)")
        plt.ylabel("Total Mass $M_{\\mathrm{tot}}$ (kg)")
        plt.title("Assimilated Mass Decay vs. Ground Truth")
        plt.grid(True, linestyle=":")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self._get_save_path(output_path), dpi=300)
        plt.close()

    def plot_parameter_convergence(self, output_path: str = "parameter_convergence.png"):
        """Plot structural aspect ratio parameter recovery trajectory."""
        plt.figure(figsize=(9, 5.5))
        a_mean = np.array(self.da_results["a_mean"])
        a_std = np.array(self.da_results["a_std"])

        plt.axhline(self.true_a, color='crimson', linestyle='--', linewidth=2.5, label=f'True $a = {self.true_a}$')
        plt.errorbar(self.t_years, a_mean, yerr=a_std, fmt='-o', color='tab:blue', ecolor='lightgray', capsize=3, linewidth=1.5, label='Assimilated Posterior $a$')

        plt.xlabel("Time (Years)")
        plt.ylabel("Primary Aspect Ratio $a$")
        plt.title("Aspect Ratio Parameter Recovery (SMC Particle Filter)")
        plt.grid(True, linestyle=":")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self._get_save_path(output_path), dpi=300)
        plt.close()

    def plot_effective_sample_size(self, output_path: str = "effective_sample_size.png"):
        """Plot effective sample size N_eff and SUR resampling triggers."""
        fig, ax = plt.subplots(figsize=(9, 5))
        n_eff = np.array(self.da_results["n_eff"])
        resampled_mask = np.array(self.da_results["resampled"])
        threshold = 0.8 * self.n_particles

        ax.plot(self.t_years, n_eff, color="tab:purple", linewidth=2, label="$N_{\\mathrm{eff}}$")
        ax.axhline(threshold, color="tab:red", linestyle="--", label=f"Resampling Threshold ($0.8 N = {threshold:.0f}$)")

        for idx, is_resampled in enumerate(resampled_mask):
            if is_resampled:
                ax.axvline(self.t_years[idx], color="tab:orange", linestyle=":", alpha=0.7, label="SUR Resample Event" if idx == resampled_mask.tolist().index(True) else "")

        ax.set_xlabel("Time (Years)")
        ax.set_ylabel("Effective Ensemble Size $N_{\\mathrm{eff}}$")
        ax.set_title("Particle Filter Diagnostics: $N_{\\mathrm{eff}}$ and Resampling Events")
        ax.grid(True, linestyle=":")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self._get_save_path(output_path), dpi=300)
        plt.close()

    def plot_aspec_evolution(self, output_path: str = "aspec_evolution.png"):
        """Plot bulk specific surface area evolution over time."""
        plt.figure(figsize=(9, 5.5))
        as_mean = np.array(self.da_results["aspec_mean"])
        as_std = np.array(self.da_results["aspec_std"])

        plt.fill_between(self.t_years, as_mean - 2 * as_std, as_mean + 2 * as_std, color="tab:green", alpha=0.3, label="Posterior 95% CI")
        plt.plot(self.t_years, as_mean, color="tab:green", linewidth=2, label="Assimilated $A_{\\mathrm{spec}}$ Mean")
        plt.plot(self.t_years, self.true_aspecs, color="crimson", linewidth=2.5, linestyle="--", label="True $A_{\\mathrm{spec}}$")

        plt.xlabel("Time (Years)")
        plt.ylabel("Specific Surface Area $A_{\\mathrm{spec}}$ (m$^2$/kg)")
        plt.title("Aggregate Specific Surface Area Evolution")
        plt.grid(True, linestyle=":")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(self._get_save_path(output_path), dpi=300)
        plt.close()

    def generate_all_plots(self):
        """Run full experiment and output all 5 high-resolution plots."""
        self.generate_ground_truth()
        self.run_da_experiment()

        self.plot_spaghetti_no_da()
        self.plot_assimilated_mass_trajectory()
        self.plot_parameter_convergence()
        self.plot_effective_sample_size()
        self.plot_aspec_evolution()


if __name__ == "__main__":
    creator = PlotCreator()
    creator.generate_all_plots()