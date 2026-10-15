import logging
import math

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from pydantic import BaseModel, model_validator
from molmass import Formula

from true_values import TRUE_VALUES, TrueObservation

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class Environment(BaseModel):
    rainfall: float = 750  # mm per year, 
    average_temperature: float = 10.7 # Celsius 
    atmospheric_CO2: int = 410 # ppm 
    mixing_depth: float = 0.5 # m
    soil_bulk_density: float = 1.7 # kg per liter
    porosity: float = 0.3


class ModelObservation(BaseModel):
    initial_diam: float
    original_mass: float
    diam_per_step: list[float] = []

    # TODO: implement these
    # mass_per_step: list[float] = []
    # percentage_dissolved_per_step: list[float] = []

    @model_validator(mode="after")
    def initialize_lists(self) -> "ModelObservation":
        if not self.diam_per_step:
            self.diam_per_step = [self.initial_diam]
        # if not self.mass_per_step:
        #             self.mass_per_step = [self.orginal_mass]
        # if not self.percentage_dissolved_per_step:
        #     self.percentage_dissolved_per_step = [0.0]
        return self


class Model:
    def __init__(self, environment: Environment, model_observations: list[ModelObservation], true_values: list[TrueObservation]):
        self.environment = environment
        self.model_observations = model_observations
        self.true_values = true_values

        # State variables
        self.ph: float = 6.0

        # Model settings
        self.time: int = 40 # year
        self.co2_capture_ratio = 4 * Formula("CO2").mass / Formula("Mg2SiO4").mass # See reaction formula page 2

        # Initialize plot results
        plt.figure(figsize=(10, 6))

    def __str__(self) -> str:
        summary = [
            f"\n\n=== Model Summary ===\n",
            f"Environment: Temp={self.environment.average_temperature}°C, Rainfall={self.environment.rainfall}mm/year",
            f"State: pH={self.ph}, Time steps={self.time} years",
            "Observations (Initial -> Final Diameter):"
        ]
        for obs in self.model_observations:
            initial = obs.diam_per_step[0]
            final = obs.diam_per_step[-1]
            summary.append(f"  - {initial} µm -> {final:.2f} µm (Steps recorded: {len(obs.diam_per_step)})")

        summary.append("\n\n")
        
        return "\n".join(summary)
    

    # @property
    # def dissolution_rate(self) -> float:
    #     """Implemented a rate which gives the right answers, but it should give """        
    #     return 6.69e-10

    @property
    def dissolution_rate(self) -> float:
        """De pH (Olsen) en Temperatuur (Arrhenius) formules uit het paper van Knops & Vink"""
        # 1. pH afhankelijkheid (Olsen)
        if self.ph < 6:
            log_r = -0.48 * self.ph - 6.9
        else:
            log_r = -0.18 * self.ph - 8.8
            
        r_ph_factor = 10 ** log_r
        
        # 2. Temperatuur afhankelijkheid (Arrhenius functie uit paper)
        R = 8.314          # Universele gasconstante (J / (K * mol))
        T = self.environment.average_temperature + 273.15  # Absolute temperatuur in Kelvin
        T_r = 283.7        # Referentietemperatuur uit het paper (K)
        k_tr = 7.43e-11    # Referentie snelheidskonstante (mol * m^-2 * s^-1)
        E_a = 79000        # Activatie-energie voor forsteriet (J/mol)
        
        # Arrhenius temperatuurfactor
        arrhenius_factor = math.exp(-(E_a / R) * ((1 / T) - (1 / T_r)))
        
        # Gecombineerde dissolutiesnelheid
        return k_tr * arrhenius_factor * r_ph_factor

    # @property
    # def dissolution_rate(self) -> float:
    #     """Gecorrigeerde pH (Olsen) en Temperatuur (Arrhenius) dissoluratiesnelheid"""
    #     # 1. pH afhankelijkheid (Olsen formule geeft direct de basissnelheid)
    #     if self.ph < 6:
    #         log_r = -0.48 * self.ph - 6.9
    #     else:
    #         log_r = -0.18 * self.ph - 8.8
            
    #     r_base = 10 ** log_r  # Dit is de absolute snelheid op basis van pH
        
    #     # 2. Temperatuur afhankelijkheid (Arrhenius correctie)
    #     R = 8.314          # Universele gasconstante (J / (K * mol))
    #     T = self.environment.average_temperature + 273.15  # Kelvin
    #     T_r = 283.7        # Referentietemperatuur uit het paper (K)
    #     E_a = 79000        # Activatie-energie (J/mol)
        
    #     arrhenius_factor = math.exp(-(E_a / R) * ((1 / T) - (1 / T_r)))
        
    #     # 3. Kalibratiefactor zodat het bij de standaard pH=6.0 en T=10.7°C 
    #     # exact uitkomt op jouw werkende waarde (6.69e-10)
    #     # Bij pH=6.0 is r_base ongeveer 1.32e-10. Om naar 6.69e-10 te gaan is de factor ~5.06
    #     calibration_factor = 6.69e-10 / 1.32e-10 
        
    #     return r_base * arrhenius_factor * calibration_factor


    @staticmethod
    def calculate_shrunk_diameter(current_diam: float, dissolution_rate: float) -> float:
        """Calculates the new diameter of an olivine particle after one year."""
        if current_diam <= 0:
             return 0.0

        density = 3.3 * 10**6  # g/m^3
        vm = Formula("Mg2SiO4").mass / density  # m^3 / mol

        seconds_per_year = 31536000 # TODO: Fix the schrikkeljaren
        diam_reduction = 2 * vm * dissolution_rate * seconds_per_year * 10**6
        
        return max(0.0, current_diam - diam_reduction)

    @staticmethod
    def increase_ph():
        """Increases the pH since olivine is basic"""
        ...

    def plot_model(self):
        """Plott de modelresultaten op basis van de theoretische formules tegen de ware observaties."""
        plt.figure(figsize=(12, 8))
        colors = ['b', 'g', 'r']
        
        for idx, obs in enumerate(self.model_observations):
            init_d = obs.initial_diam
            years = list(range(len(obs.diam_per_step)))
            c = colors[idx % len(colors)]
            
            # Plot de model simulatie met theoretische formules
            plt.plot(years, obs.diam_per_step, label=f"Model (Theo pH/T) ({init_d} µm)", linestyle="-", color=c)
            
            # Plot de werkelijke waarden ter vergelijking
            matching_true = next((t for t in self.true_values if t.initial_diam == init_d), None)
            if matching_true:
                true_years = sorted(matching_true.diam_over_years.keys())
                true_diams = [matching_true.diam_over_years[y] for y in true_years]
                plt.plot(true_years, true_diams, label=f"True ({init_d} µm)", linestyle="--", marker="o", alpha=0.7, color=c)

        plt.title("Model vs True Values: pH & Temp Dependent Olivine Weathering")
        plt.xlabel("Years of Weathering")
        plt.ylabel(r"Particle Diameter ($\mu$m)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()

        plt.savefig("olivine_model_comparison_theoretical.png", dpi=300)
        plt.close()
        logging.info("Model vergelijkingsplot succesvol opgeslagen.")
        

    def spaghetti_plot(self):
        ...

    def step(self):
        """Update de deeltjesdiameter voor elke observatie op basis van de theoretische dissolution rate"""
        for obs in self.model_observations:
            current_diam = obs.diam_per_step[-1]
            new_diam = self.calculate_shrunk_diameter(current_diam, self.dissolution_rate)
            obs.diam_per_step.append(new_diam)

    def run_model(self):
        for current_year in range(1, self.time + 1):
            logging.info(f"Running step year {current_year} of {self.time} years.")
            self.step()

        # Print an overview of what happened
        logging.info(self)

        self.plot_model()


if __name__ == "__main__":
    environment = Environment()
    model_observations = [
        ModelObservation(initial_diam=1000, original_mass=144),
        ModelObservation(initial_diam=707, original_mass=243), 
        ModelObservation(initial_diam=50, original_mass=37)       
    ]
    model = Model(environment=environment, model_observations=model_observations, true_values=TRUE_VALUES)
    model.run_model()