import logging

from matplotlib import pyplot as plt
from pydantic import BaseModel
from molmass import Formula

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class Environment(BaseModel):
    rainfall: float = 750  # mm per year, 
    average_temperature: float = 10.7 # Celsius 
    atmospheric_CO2: int = 410 # ppm 
    mixing_depth: float = 0.5 # m
    soil_bulk_density: float = 1.7 # kg per liter
    porosity: float = 0.3


class Model:
    # TODO: implement true values
    def __init__(self, environment: Environment, true_values=None):
        self.environment = environment
        self.true_values = true_values

        # State variables
        self.ph: float = 6.0
        self.co2: int = ... # ppm
        self.diam_distribution: list[float] = [8, 16, 50, 88, 150, 250, 420, 707, 1000, 1410, 2000]

        # Model settings
        self.time: int = 40 # year
        self.co2_capture_ratio = 4 * Formula("CO2").mass / Formula("Mg2SiO4").mass # See reaction formula page 2

        # Initialize plot results
        plt.figure(figsize=(10, 6))
    

    @property
    def dissolution_rate(self):
        """Berekent automatisch de dissolution rate op basis van de huidige pH."""
        if self.ph < 6.0:
            log_r = -0.48 * self.ph - 6.9
        else:
            log_r = -0.18 * self.ph - 8.8
        return 10**log_r

    def spaghetti_plot(self):
        plt.title("Ensemble Forward Run: CO2 Sequestration Uncertainty (Spaghetti Plot)")
        plt.xlabel("Years of Weathering")
        plt.ylabel("Cumulative CO2 Capture (kg)")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Save plot to file
        output_filename = "owcs_spaghetti_plot.png"
        plt.savefig(output_filename, dpi=300)
        logging.info(f"Spaghetti plot successfully saved as '{output_filename}'")
        plt.close()

    def step(self):
        """
        Update the state parameters
        """
        ...

    def run_model(self):
        for current_year in range(1, self.time + 1):
            logging.info(f"Running step year {current_year} of {self.time} years.")
            self.step()

        self.spaghetti_plot()


if __name__ == "__main__":
    env = Environment()
    model = Model(environment=env, true_values=None)
    model.run_model()