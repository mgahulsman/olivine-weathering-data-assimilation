import json
import logging
import numpy as np

# Force matplotlib to use a non-interactive backend (prevents Tcl/Tk errors)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_config(file_path: str ="config.json") -> ?:
    """Loads configuration and observations from the JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise ValueError(f"Configuration file {file_path} not found.")


def calculate_dissolution_rate(ph: float) -> float:
    """
    Calculates the dissolution rate (log r) based on pH using Olsen's equations[cite: 4].
    """
    if ph < 6.0:
        log_r = -0.48 * ph - 6.9
    else:
        log_r = -0.18 * ph - 8.8
    return 10**log_r

def run_ensemble_simulation(num_runs=50):
    """
    Runs multiple model simulations by perturbing initial conditions 
    to create data for a spaghetti plot.
    """
    config = load_config()
    
    years = np.array(config["observations_table_2"]["years"])
    base_ph = config["initial_state_variables"]["initial_ph"]
    base_rainfall = config["environmental_parameters"]["rainfall_mm_per_year"]
    
    total_initial_mass = sum(sc["original_mass_kg"] for sc in config["observations_table_2"]["size_classes"])
    
    molar_mass_olivine = 140.7
    molar_mass_co2 = 44.01
    co2_capture_ratio = (4 * molar_mass_co2) / molar_mass_olivine
    
    logging.info(f"Starting ensemble run with {num_runs} iterations...")
    
    plt.figure(figsize=(10, 6))
    
    for _ in range(num_runs):
        perturbed_ph = np.random.normal(base_ph, 0.3)         # Standard deviation of 0.3 pH units
        perturbed_rainfall = np.random.normal(base_rainfall, 50) # Standard deviation of 50 mm/year
        
        # Simple dynamic simulation over the years
        rate = calculate_dissolution_rate(perturbed_ph)
        base_rate = calculate_dissolution_rate(base_ph)
        scaling_factor = rate / base_rate
        
        # Calculate trajectory over time
        trajectory = []
        for year in years:
            if year == 0:
                trajectory.append(0.0)
            else:
                fraction_dissolved = min(1.0, (year / 40.0) ** 0.6 * 0.48 * scaling_factor * (perturbed_rainfall / base_rainfall))
                co2_captured = fraction_dissolved * total_initial_mass * co2_capture_ratio
                trajectory.append(co2_captured)

        # Line plot for every itteration
        plt.plot(years, trajectory, color="blue", alpha=0.15, linewidth=1.5)

    # Plot formatting
    plt.title("Ensemble Forward Run: CO2 Sequestration Uncertainty (Spaghetti Plot)")
    plt.xlabel("Years of Weathering")
    plt.ylabel("Cumulative CO2 Capture (kg)")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save plot to file
    output_filename = "owcs_spaghetti_plot.png"
    plt.savefig(output_filename, dpi=300)
    logging.info(f"Spaghetti plot successfully saved as '{output_filename}'")
    plt.close()

if __name__ == "__main__":
    run_ensemble_simulation(num_runs=50)