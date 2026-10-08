import logging

from matplotlib import pyplot as plt
from pydantic import BaseModel
from molmass import Formula

from true_values import TRUE_VALUES, Observation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class Environment(BaseModel):
    rainfall: float = 750  # mm per year, 
    average_temperature: float = 10.7 # Celsius 
    atmospheric_CO2: int = 410 # ppm 
    mixing_depth: float = 0.5 # m
    soil_bulk_density: float = 1.7 # kg per liter
    porosity: float = 0.3


class Model:
    def __init__(self, environment: Environment, true_values: list[Observation]):
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
    def dissolution_rate(self) -> float:
        """Calculates dissolution rate accourding to the formula in section 2"""
        if self.ph < 6.0:
            log_r = -0.48 * self.ph - 6.9
        else:
            log_r = -0.18 * self.ph - 8.8
        return 10**log_r

    @staticmethod
    def calculate_shrunk_diameter(current_diam: float, dissolution_rate: float) -> float:
        """Calculates the new diameter of an olivine particle after one year."""
        # TODO: find prove of these formule and refrence
        # This might be the original paper: https://pdf.sciencedirectassets.com/276245/1-s2.0-S0082078455X80026/1-s2.0-S0082078455800331/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEF0aCXVzLWVhc3QtMSJHMEUCIDQ%2FEe5zFet2HU9X6Mc8n5bavpiGGvMN1LLQGZFLsEgcAiEA9VNIqaWRKT1tsesRfVbL8F2tsC8qqmgejnatrEKIZ%2BIqswUIJhAFGgwwNTkwMDM1NDY4NjUiDPDSE6S5P7Xk80Yp4iqQBU164g0seJM7YC4iMQSCPhpbffzg9bDpV6l2nT4fVrCtms6cPv1K%2ByYNG5LA0gTLLXPKEKglOxAw%2F53U5%2FAGTfXoj8ioDHuIrk%2Bbi26pq3gN5p7nMejjQR7MjzUOMFQSDN9T3B4%2BhuAHRrBdQuQBzW97myt%2FGP%2Bp1Auyp8HBPk%2Ff%2FWOIHgpa4tBRYC3g9mkAoHm%2BlU%2B3cyTKLcUqVwyvwEyov0lxieYEbxPc2ZZYrbZcCpfEESG9yquYlf8iLJWvVBNXv6L8TDAbpLGM2zho6fzTw2pqGrlx6sMwrJSJ9HABta0aIuPQUu7vEodGtMxIVKOR9vpjwmnWBsc3O2mSRSLdJSAJuejaqXdul5piQHSZLjuFK3wJq%2BGoWv70c0Z0vRHybWiyOOvogIjnODxSJIkizUAI722nk3TCJneTF5%2BAnKlZbfx9g5KXnJeak25eu83GEIaE5ma1CXvOUJRr2Y58Fw6EV%2ByInhxO5ApjJmb6FHobvO5F88jT5lLQ1sLRzuYUbXxUjVBuiJ7t6O16%2F7W84SeqCE7%2BX7QLDGyBEcFRneo%2FWbfkw5gtwlwPmt%2FTD%2BkuXC0o3oJl3PACO7u2ukVTSznePt42qLV5PDwQfv%2FIwtHRpc550DkkNrUwghn9FoDlvNW8Vq1wme5LU3p4GGtiwws4X%2BoJbsuWqGcMvg6wvKAzUhq5NFzdFKsuDNP3vfQ2yshDK%2BLumh53kFGwhyWwghFt3WrPX7x2GcaY7%2FV%2F2hTbPM%2FoQHIrHkuL7Vk%2FLL50q4IwklRQwfwb1fkEhe9yfRWVH7LRj1gos8W7CRfNbm22j%2FdMmW67jY6Dena9iofD%2F8X%2FtOCaobkFTPu9lzt8P%2BLkvMVLMD2NaRQyWPi5MKyontYGOrEBhOf22GLUtkWnC84t%2FzoEubBwOPBFV6vOuOtSKnJO%2F9VUqNPrRPeN5ozrvTdvNeAJs1ODL2xi3BdXdTKxJfVEj0zKfl5vQBZMmnipEB2te37nc8w0K2fX3D7u43ojmAn0V%2F%2Fx%2Byinmq7ojpGWUTsVONkeRIDiQCwuy6M0IZaX6FN68ZvizzVLcTGA2xHwM6CvL3TMV4bwXwG2fA3p81HXo9d8m6qj4AEoTsugzHeQ9iJF&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20261008T134749Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQT5652W7%2F20261008%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e7db6805785ce81fdaa4d95d17f91ac6e6b5b1ed4d9674630be2f61f7125e449&hash=987e68738c1fdf73fe4c418497838a2f9acfb1abecdae9ebe8a107de8ed5b116&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0082078455800331&tid=spdf-866f3cbe-3ce0-45ed-84d6-4b67e29bcebd&sid=7d9e34c218e98849f12841918dc7ab7a8e58gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0b0a025053545d565706&rr=a475997d7e689f6c&cc=nl
        if current_diam <= 0:
             return 0.0

        density = 3.3 * 10**6  # g/m^3
        vm = Formula("Mg2SiO4").mass / density  # m^3 / mol

        seconds_per_year = 31536000 # TODO: Fix the schrikkeljaren
        diam_reduction = 2 * vm * dissolution_rate * seconds_per_year * 10**6
        
        return max(0.0, current_diam - diam_reduction)

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
        # TODO: next step is this
        ...

    def run_model(self):
        for current_year in range(1, self.time + 1):
            logging.info(f"Running step year {current_year} of {self.time} years.")
            self.step()

        self.spaghetti_plot()


if __name__ == "__main__":
    env = Environment()
    model = Model(environment=env, true_values=TRUE_VALUES)
    model.run_model()