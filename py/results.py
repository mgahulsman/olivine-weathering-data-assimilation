import json
import numpy as np
import matplotlib.pyplot as plt
from forward_model import OlivineDissolutionModel

# ==============================================================================
# 1. SETUP EN EXPERIMENT CONFIGURATIE
# ==============================================================================
np.random.seed(42)  # Voor herhaalbaarheid (reproducibility)

N_P = 50                      # Aantal particles in het ensemble
TOTAL_YEARS = 40.0            # Simulatieduur in jaren
DT_YEARS = 0.5                # Tijdstap van 0.5 jaar (halfjaarlijks)
PH_VALUE = 6.0                # Constante pH
SIGMA_OBS = 15.0              # Observatiefout standaardafwijking in kg (massa)
N_EFF_THRESHOLD = 0.5 * N_P   # Resampling drempelwaarde

# ==============================================================================
# 2. STEP 1: GENEREREN VAN DE "SYNTHETIC TRUTH" & OBSERVATIES
# ==============================================================================
print("[1/4] Genereren van de Synthetic Truth & Observaties...")
truth_model = OlivineDissolutionModel("data/initial_grains.json")
years, true_mass, true_aspec = truth_model.run_simulation(
    total_years=TOTAL_YEARS, 
    dt_years=DT_YEARS, 
    pH=PH_VALUE
)

n_steps = len(years)

# Observatie-intervallen (bijv. elke 2 jaar = elke 4 tijdstappen)
obs_indices = np.arange(4, n_steps, 4)
obs_years = years[obs_indices]

# Observaties met Gaussische meetruis: y = Hx + epsilon
obs_mass = true_mass[obs_indices] + np.random.normal(0, SIGMA_OBS, size=len(obs_indices))

# ==============================================================================
# 3. STEP 2: PRIOR ENSEMBLE GENERATIE & SPAGHETTI PLOT
# ==============================================================================
print("[2/4] Genereren van het Prior Ensemble...")
prior_trajectories = np.zeros((N_P, n_steps))

for i in range(N_P):
    p_model = OlivineDissolutionModel("data/initial_grains.json")
    # Ruis toe te voegen aan de initiële massa van de particles
    p_model.current_masses *= np.random.uniform(0.8, 1.2)
    _, m_hist, _ = p_model.run_simulation(
        total_years=TOTAL_YEARS, 
        dt_years=DT_YEARS, 
        pH=PH_VALUE
    )
    prior_trajectories[i, :] = m_hist

# Visualisatie: Figure 1 - Prior Spaghetti Plot
plt.figure(figsize=(9, 4.5))
for i in range(N_P):
    plt.plot(years, prior_trajectories[i, :], linewidth=1)
plt.plot(years, true_mass, 'k--', label='Synthetic Truth', linewidth=2)
plt.scatter(obs_years, obs_mass, color='red', marker='*', s=80, zorder=5, label='Synthetische Metingen')
plt.xlabel('Tijd (jaren)')
plt.ylabel('Totale Massa (kg)')
plt.title('Prior Ensemble Trajectorieën (Spaghetti Plot) vs. Observaties')
plt.legend()
plt.tight_layout()
plt.savefig('figure_prior_spaghetti.png', dpi=300)
plt.close()

# ==============================================================================
# 4. STEP 3: SEQUENTIAL PARTICLE FILTER ASSIMILATIE
# ==============================================================================
print("[3/4] Uitvoeren van het Particle Filter...")

weights = np.ones(N_P) / N_P
posterior_trajectories = prior_trajectories.copy()

N_eff_history = []
resample_events = []

def systematic_resample(w):
    N = len(w)
    positions = (np.arange(N) + np.random.uniform(0, 1)) / N
    indexes = np.zeros(N, dtype=int)
    cumulative_sum = np.cumsum(w)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

# Tijdstapafhankelijke assimilatieloop
for idx, obs_step in enumerate(obs_indices):
    y_val = obs_mass[idx]
    predictions = posterior_trajectories[:, obs_step]
    
    # 1. Residuals & Gaussian Likelihood
    residuals = y_val - predictions
    likelihood = np.exp(-0.5 * (residuals / SIGMA_OBS) ** 2)
    
    # 2. Update & Normalisatie van Gewichten
    weights *= likelihood
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights /= w_sum
    else:
        weights = np.ones(N_P) / N_P
        
    # 3. Filter Health (Effective Sample Size Neff)
    N_eff = 1.0 / np.sum(weights ** 2)
    N_eff_history.append((years[obs_step], N_eff))
    
    # 4. Systematic Resampling bij Degeneratie
    if N_eff < N_EFF_THRESHOLD:
        resample_events.append(years[obs_step])
        idx_resampled = systematic_resample(weights)
        posterior_trajectories = posterior_trajectories[idx_resampled, :]
        weights = np.ones(N_P) / N_P

# ==============================================================================
# 5. STEP 4: VISUALISATIE VAN DE RESULTATEN & METRIEKEN
# ==============================================================================
print("[4/4] Genereren van Resultaatgrafieken & Metrieken...")

# Figure 2: Posterior Trajectories
plt.figure(figsize=(9, 4.5))
for i in range(N_P):
    plt.plot(years, posterior_trajectories[i, :], color='royalblue', alpha=0.3, linewidth=1)
plt.plot(years, true_mass, 'k--', label='Synthetic Truth', linewidth=2)
plt.scatter(obs_years, obs_mass, color='red', marker='*', s=80, zorder=5, label='Synthetische Metingen')
plt.xlabel('Tijd (jaren)')
plt.ylabel('Totale Massa (kg)')
plt.title('Posterior Trajectorieën Ná Particle Filter Data-Assimilatie')
plt.legend()
plt.tight_layout()
plt.savefig('figure_posterior_trajectories.png', dpi=300)
plt.close()

# Figure 3: Filter Health (Neff over de tijd)
t_neff, val_neff = zip(*N_eff_history)
plt.figure(figsize=(8, 3.5))
plt.plot(t_neff, val_neff, 'b-o', label=r'$N_{\mathrm{eff}}$')
plt.axhline(N_EFF_THRESHOLD, color='r', linestyle='--', label='Resampling Drempelwaarde')
plt.xlabel('Tijd (jaren)')
plt.ylabel(r'Effective Sample Size ($N_{\mathrm{eff}}$)')
plt.title(r'Filter Gezondheid: $N_{\mathrm{eff}}$ en Resampling Events')
plt.legend()
plt.tight_layout()
plt.savefig('figure_filter_health.png', dpi=300)
plt.close()

# Kwantitatieve evaluatie: RMSE Berekening
prior_mean = np.mean(prior_trajectories, axis=0)
posterior_mean = np.mean(posterior_trajectories, axis=0)

prior_rmse = np.sqrt(np.mean((prior_mean - true_mass) ** 2))
posterior_rmse = np.sqrt(np.mean((posterior_mean - true_mass) ** 2))

print("\n" + "="*50)
print(" EXPERIMENT METRIEKEN & RESULTATEN")
print("="*50)
print(f"Prior Ensemble RMSE:     {prior_rmse:.4f} kg")
print(f"Posterior Ensemble RMSE: {posterior_rmse:.4f} kg")
print(f"Foutreductie (RMSE):     {((prior_rmse - posterior_rmse) / prior_rmse) * 100:.2f}%")
print(f"Aantal Resampling Events: {len(resample_events)}")
print("="*50)