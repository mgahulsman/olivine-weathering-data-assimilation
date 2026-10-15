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
    rainfall: float = 750  # mm per year
    average_temperature: float = 10.7  # Celsius
    atmospheric_CO2: int = 410  # ppm
    mixing_depth: float = 0.5  # m
    soil_bulk_density: float = 1.7  # kg per liter
    porosity: float = 0.3


class ModelObservation(BaseModel):
    initial_diam: float  # µm
    original_mass: float  # kg
    diam_per_step: list[float] = []

    @model_validator(mode="after")
    def initialize_lists(self) -> "ModelObservation":
        if not self.diam_per_step:
            self.diam_per_step = [self.initial_diam]
        return self


class Model:
    def __init__(
        self,
        environment: Environment,
        model_observations: list[ModelObservation],
        true_values: list[TrueObservation],
    ):
        self.environment = environment
        self.model_observations = model_observations
        self.true_values = true_values

        # State variables
        self.ph: float = 6.0

        # Model settings
        self.time: int = 40  # years
        self.co2_capture_ratio = (
            4 * Formula("CO2").mass / Formula("Mg2SiO4").mass
        )  # Reactieverhouding (pagina 2)

    def __str__(self) -> str:
        summary = [
            "\n=== Model Summary ===",
            f"Environment: Temp={self.environment.average_temperature}°C, Rainfall={self.environment.rainfall}mm/year",
            f"State: pH={self.ph}, Simulation Time={self.time} years",
            "Observations (Initial -> Final Diameter):",
        ]
        for obs in self.model_observations:
            initial = obs.diam_per_step[0]
            final = obs.diam_per_step[-1]
            summary.append(
                f"  - {initial} µm -> {final:.2f} µm (Steps: {len(obs.diam_per_step)})"
            )
        return "\n".join(summary)

    @property
    def dissolution_rate(self) -> float:
        """
        Gecombineerde pH (Olsen) en Temperatuur (Arrhenius) dissolutiesnelheid.
        Gekalibreerd op k_fitted = 6.69e-10 mol*m^-2*s^-1 bij standaardcondities (pH=6.0, T=10.7°C),
        hetgeen exact de ~1.8 µm/jaar krimpsnelheid uit Tabel 2 oplevert.
        """
        # 1. pH afhankelijkheid (Olsen)
        if self.ph < 6:
            log_r = -0.48 * self.ph - 6.9
        else:
            log_r = -0.18 * self.ph - 8.8

        r_ph_factor = 10**log_r
        r_ph_ref = 10 ** (-0.18 * 6.0 - 8.8)  # Referentiewaarde bij pH = 6.0
        ph_scaling = r_ph_factor / r_ph_ref

        # 2. Temperatuur afhankelijkheid (Arrhenius)
        R = 8.314  # Gasconstante (J / (K * mol))
        T = self.environment.average_temperature + 273.15  # Kelvin
        T_r = 10.7 + 273.15  # Referentietemperatuur (10.7°C in Kelvin)
        E_a = 79000  # Activatie-energie (J/mol)

        arrhenius_factor = math.exp(-(E_a / R) * ((1 / T) - (1 / T_r)))

        # 3. Gekalibreerde basissnelheid
        k_fitted = 6.69e-10

        return k_fitted * arrhenius_factor * ph_scaling

    @staticmethod
    def calculate_shrunk_diameter(
        current_diam: float, dissolution_rate: float
    ) -> float:
        """Berekent de nieuwe deeltjesdiameter na één jaar verwering (Shrinking Core Model)."""
        if current_diam <= 0:
            return 0.0

        density = 3.3 * 10**6  # g/m^3
        vm = Formula("Mg2SiO4").mass / density  # m^3 / mol

        seconds_per_year = 31536000
        diam_reduction = 2 * vm * dissolution_rate * seconds_per_year * 10**6

        return max(0.0, current_diam - diam_reduction)

    def step(self):
        """Voert één tijdstap (1 jaar) uit voor alle observaties."""
        rate = self.dissolution_rate
        for obs in self.model_observations:
            current_d = obs.diam_per_step[-1]
            new_d = self.calculate_shrunk_diameter(current_d, rate)
            obs.diam_per_step.append(new_d)

    def run_model(self):
        """Runt de simulatie over 40 jaar en plot het resultaat."""
        for year in range(1, self.time + 1):
            self.step()

        logging.info(self)
        self.plot_model()

    def plot_model(self):
        """Plot de modelresultaten (diameterreductie) direct tegen de werkelijke observaties."""
        plt.figure(figsize=(12, 8))

        # Dynamisch kleurenpalet voor de verschillende grootteklassen
        cmap = plt.get_cmap("tab10")

        for idx, obs in enumerate(self.model_observations):
            init_d = obs.initial_diam
            years = list(range(len(obs.diam_per_step)))
            color = cmap(idx % 10)

            # Modellijn (doorgetrokken)
            plt.plot(
                years,
                obs.diam_per_step,
                label=f"Model ({init_d} µm)",
                linestyle="-",
                color=color,
                linewidth=2,
            )

            # Zoek bijbehorende werkelijke waarden uit true_values.py
            matching_true = next(
                (t for t in self.true_values if t.initial_diam == init_d), None
            )
            if matching_true:
                true_years = sorted(matching_true.diam_over_years.keys())
                true_diams = [
                    matching_true.diam_over_years[y] for y in true_years
                ]
                plt.plot(
                    true_years,
                    true_diams,
                    label=f"True ({init_d} µm)",
                    linestyle="--",
                    marker="o",
                    alpha=0.7,
                    color=color,
                )

        plt.title(
            "Shrinking Core Model vs True Values (Vink & Knops Table 2)",
            fontsize=14,
        )
        plt.xlabel("Years of Weathering", fontsize=12)
        plt.ylabel(r"Particle Diameter ($\mu$m)", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small"
        )
        plt.tight_layout()

        plt.savefig("olivine_model_vs_true_values.png", dpi=300)
        plt.close()
        logging.info(
            "Grafiek succesvol opgeslagen als 'olivine_model_vs_true_values.png'"
        )


if __name__ == "__main__":
    environment = Environment()

    # Alle of geselecteerde grootteklassen uit Tabel 2
    model_observations = [
        ModelObservation(initial_diam=1000, original_mass=144),
        ModelObservation(initial_diam=707, original_mass=243),
        ModelObservation(initial_diam=50, original_mass=37),
    ]

    model = Model(
        environment=environment,
        model_observations=model_observations,
        true_values=TRUE_VALUES,
    )
    model.run_model()