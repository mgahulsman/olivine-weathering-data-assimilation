import numpy as np
import matplotlib.pyplot as plt
from forward_model import OlivineDissolutionModel

# ==============================================================================
# 1. SETUP EN EXPERIMENT CONFIGURATIE
# ==============================================================================
np.random.seed(42)

TOTAL_YEARS = 40.0
DT_YEARS = 0.5
PH_VALUE = 6.0
V_FLUID = 1000.0       # Volume in liters (nodig voor C_Mg berekening)
M_MOLAR = 0.1406       # Molaire massa forsteriet (kg/mol)

# Ruisniveau (sigma) ingesteld op 5% (volgens methodologie testbereik 1% - 20%)
SIGMA_RELATIVE = 0.05  

# ==============================================================================
# 2. GENEREREN VAN DE "SYNTHETIC TRUTH" (REFERENCE RUN)
# ==============================================================================
print("Genereren van de Synthetic Truth...")

# Let op: zorg dat 'area_multiplier' in je forward_model.py is geaccepteerd!
truth_model = OlivineDissolutionModel("data/initial_ground_truth.json", V_fluid=V_FLUID)

# We genereren de waarheid met een vaste vormparameter (theta_morph = 1.0)
years, true_mass, true_aspec = truth_model.run_simulation(
    total_years=TOTAL_YEARS, 
    dt_years=DT_YEARS, 
    pH=PH_VALUE,
    area_multiplier=1.0 
)

# Bereken C_Mg (Opgelost Magnesium in mol/L) direct uit de massaverandering
# Forsteriet bevat 2 mol Mg per mol mineraal
dissolved_mass = true_mass[0] - true_mass
true_cmg = (dissolved_mass / M_MOLAR) * 2.0 / V_FLUID

# ==============================================================================
# 3. GENEREREN VAN OBSERVATIES MET MEETRUIS (H OPERATOR)
# ==============================================================================
print("Genereren van Observaties (met Gaussische meetruis op Aspec en C_Mg)...")

# Observatie-intervallen (bijv. elke 2 jaar = elke 4 tijdstappen)
obs_indices = np.arange(4, len(years), 4)
obs_years = years[obs_indices]

# Observaties voor Specifieke Oppervlakte (A_spec)
# y_t = H*x_t + epsilon, waarbij epsilon ~ N(0, R)
sigma_aspec = true_aspec[obs_indices] * SIGMA_RELATIVE
obs_aspec = true_aspec[obs_indices] + np.random.normal(0, sigma_aspec)

# Observaties voor Magnesium Concentratie (C_Mg)
sigma_cmg = true_cmg[obs_indices] * SIGMA_RELATIVE
obs_cmg = true_cmg[obs_indices] + np.random.normal(0, sigma_cmg)

# ==============================================================================
# 4. PLOTTEN VAN DE SYNTHETISCHE OBSERVATIES
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot C_Mg
ax1.plot(years, true_cmg, 'k-', linewidth=2, label='Synthetic Truth ($C_{Mg}$)')
ax1.scatter(obs_years, obs_cmg, color='red', marker='*', s=80, label='Noisy Observations (5% error)')
ax1.set_ylabel('Concentratie $C_{Mg}$ (mol/L)')
ax1.set_title('Synthetische Observaties: Opgelost Magnesium ($C_{Mg}$)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot A_spec
ax2.plot(years, true_aspec, 'k-', linewidth=2, label='Synthetic Truth ($A_{spec}$)')
ax2.scatter(obs_years, obs_aspec, color='blue', marker='o', s=50, label='Noisy Observations (5% error)')
ax2.set_ylabel('Specifieke Oppervlakte (m$^2$/kg)')
ax2.set_xlabel('Tijd (jaren)')
ax2.set_title('Synthetische Observaties: Bulk Specifieke Oppervlakte ($A_{spec, aggregate}$)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('synthetic_observations_methodology.png', dpi=300)
plt.show()