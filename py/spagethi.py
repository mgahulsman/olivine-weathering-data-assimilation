import os
import json
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Voorkomt dat Tkinter/Tcl gebruikt wordt

import matplotlib.pyplot as plt
from pathlib import Path
from forward_model import OlivineDissolutionModel


class ParticleFilterAssimilation:
    def __init__(self, num_particles=40, true_aspect_ratio=(1.8, 1.3, 1.0), obs_noise_std=2.0):
        self.num_particles = num_particles
        self.true_aspect_ratio = true_aspect_ratio
        self.obs_noise_std = obs_noise_std  # Ingesteld op 2.0 kg ruis voor balans
        
        # 1. Genereer Ground Truth Data (True: a=1.8)
        print("[1] Genereren van Ground Truth (Echte Traject)...")
        self.generate_cpp_grains(true_aspect_ratio, "data/ground_truth.json")
        gt_model = OlivineDissolutionModel("data/ground_truth.json")
        self.times, self.true_masses, _ = gt_model.run_simulation(total_years=20.0, dt_years=1.0)
        
        # Voeg synthetische ruis toe aan de TOTALE MASSA
        self.observations = self.true_masses + np.random.normal(0, self.obs_noise_std, size=len(self.true_masses))

        # 2. Initialiseer Deeltjes (Particle Filter Ensemble)
        print(f"[2] Initialiseren van {num_particles} filter-deeltjes...")
        self.particles_params = []
        self.weights = np.ones(num_particles) / num_particles
        self.models = []

        for i in range(num_particles):
            # Prior verdeling voor a tussen 1.1 en 2.5
            a_prior = np.random.uniform(1.1, 2.5)
            b_prior = np.random.uniform(1.0, a_prior)
            c_prior = 1.0
            
            json_path = f"data/particle_{i}.json"
            self.generate_cpp_grains((a_prior, b_prior, c_prior), json_path)
            
            self.particles_params.append([a_prior, b_prior, c_prior])
            self.models.append(OlivineDissolutionModel(json_path))

    def generate_cpp_grains(self, aspect_ratio, output_json):
        a, b, c = aspect_ratio
        os.makedirs("data", exist_ok=True)
        
        project_root = Path(__file__).resolve().parent.parent
        if os.name == 'nt':
            exe_path = project_root / "cpp" / "generate_grains.exe"
        else:
            exe_path = project_root / "cpp" / "generate_grains"
            
        cmd = f'"{exe_path}" {a} {b} {c} {output_json}'
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL)

    def likelihood(self, predicted_mass, observed_mass):
        diff = predicted_mass - observed_mass
        return np.exp(-0.5 * (diff / self.obs_noise_std) ** 2) + 1e-300

    def resample(self, current_t_idx):
        """
        Resamplet deeltjes en haalt de nieuwe modellen direct in tot het huidige jaar t_idx.
        Dit voorkomt dat geresamplede modellen terugvallen op hun t=0 beginmassa.
        """
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=self.weights)
        dt_seconds = 1.0 * 365.25 * 86400.0
        
        new_params = []
        new_models = []

        for idx, i in enumerate(indices):
            a_old, b_old, c_old = self.particles_params[i]
            
            # Voeg kleine jitter/variatie toe aan de parameters
            a_new = max(1.1, a_old + np.random.normal(0, 0.02))
            b_new = max(1.0, min(a_new, b_old + np.random.normal(0, 0.02)))
            
            json_path = f"data/particle_{idx}.json"
            self.generate_cpp_grains((a_new, b_new, c_old), json_path)
            
            # Maak nieuw model en haal direct in tot de huidige stap!
            model = OlivineDissolutionModel(json_path)
            for _ in range(current_t_idx):
                model.step(dt_seconds)
                
            new_params.append([a_new, b_new, c_old])
            new_models.append(model)
            
        self.particles_params = new_params
        self.models = new_models
        self.weights = np.ones(self.num_particles) / self.num_particles

    def run_assimilation_with_spaghetti_plot(self):
        print("\n[3] Starten van Data Assimilatie met Spaghetti-generatie...")
        dt_seconds = 1.0 * 365.25 * 86400.0
        n_years = len(self.times)
        
        # Matrices om alle trajecten voor de spaghetti plot op te slaan
        mass_trajectories = np.zeros((self.num_particles, n_years))
        a_param_trajectories = np.zeros((self.num_particles, n_years))

        # Bewaar beginstatus (Jaar 0.0)
        for i in range(self.num_particles):
            mass_trajectories[i, 0] = self.models[i].get_state_vector()["m_tot"]
            a_param_trajectories[i, 0] = self.particles_params[i][0]

        for t_idx in range(1, n_years):
            t_years = self.times[t_idx]

            # Stap A: Forward step voor elk deeltje
            for i in range(self.num_particles):
                self.models[i].step(dt_seconds)
                mass_trajectories[i, t_idx] = self.models[i].get_state_vector()["m_tot"]
                a_param_trajectories[i, t_idx] = self.particles_params[i][0]

            # Stap B: Likelihood & Gewichten-update
            obs_t = self.observations[t_idx]
            for i in range(self.num_particles):
                self.weights[i] *= self.likelihood(mass_trajectories[i, t_idx], obs_t)

            # Normaliseren van gewichten
            sum_w = np.sum(self.weights)
            if sum_w > 0 and not np.isnan(sum_w):
                self.weights /= sum_w
            else:
                self.weights = np.ones(self.num_particles) / self.num_particles

            # Gewogen gemiddelde berekenen voor console-output
            avg_a = np.sum([self.particles_params[i][0] * self.weights[i] for i in range(self.num_particles)])
            print(f"Jaar {t_years:4.1f} | Obs Massa: {obs_t:7.1f} kg | Geschat a-parameter: {avg_a:.3f} (True: {self.true_aspect_ratio[0]})")

            # Stap C: Resampling bij deeltjessterfte
            N_eff = 1.0 / np.sum(self.weights ** 2)
            if N_eff < self.num_particles / 2.0:
                print(f"   --> Resampling getriggerd op jaar {t_years:.1f} (N_eff = {N_eff:.1f})")
                self.resample(current_t_idx=t_idx)

        # -------------------------------------------------------------
        # PLOTTEN VAN DE SPAGHETTI PLOT
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Bovenste Grafiek: Massa Trajecten
        for i in range(self.num_particles):
            ax1.plot(self.times, mass_trajectories[i, :], color='gray', alpha=0.3, linewidth=1)
        ax1.plot(self.times, self.true_masses, 'k--', label='True Ground Truth', linewidth=2)
        ax1.scatter(self.times, self.observations, color='red', s=20, zorder=5, label='Observaties (met ruis)')
        ax1.set_ylabel("Totale Massa (kg)")
        ax1.set_title("Spaghetti Plot: Massa-trajecten van Filter-deeltjes")
        ax1.legend()
        ax1.grid(True)

        # Onderste Grafiek: Convergentie Parameter a
        for i in range(self.num_particles):
            ax2.plot(self.times, a_param_trajectories[i, :], alpha=0.4, linewidth=1)
        ax2.axhline(self.true_aspect_ratio[0], color='red', linestyle='--', label=f'True a = {self.true_aspect_ratio[0]}')
        ax2.set_xlabel("Tijd (Jaar)")
        ax2.set_ylabel("Geschatte a-parameter")
        ax2.set_title("Spaghetti Plot: Convergentie van Vormparameter a")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("spagthiiii.png")


if __name__ == "__main__":
    pf = ParticleFilterAssimilation(
        num_particles=40,
        true_aspect_ratio=(1.8, 1.3, 1.0),
        obs_noise_std=2.0
    )
    pf.run_assimilation_with_spaghetti_plot()