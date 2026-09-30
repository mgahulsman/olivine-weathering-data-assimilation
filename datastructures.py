from pydantic import BaseModel, Field
from typing import List, Optional

# --- New: Physical Constants (previously loose in the script) ---
class PhysicalConstants(BaseModel):
    molar_mass_olivine: float = 140.7
    molar_mass_co2: float = 44.01
    
    @property
    def co2_capture_ratio(self) -> float:
        """Automatically calculates the ratio based on the constants."""
        return (4 * self.molar_mass_co2) / self.molar_mass_olivine

# --- 1. Model Info ---
class ModelInfo(BaseModel):
    model_name: str = "OWCS V6.3"
    reference_paper: str = "Vink, J.P.M., & Knops, P. (2023). Size-Fractionated Weathering of Olivine, Its CO2 Sequestration Rate, and Ecotoxicological Risk Assessment of Nickel Release. Minerals, 13(2), 235."

# --- 2. Initial State Variables ---
class InitialStateVariables(BaseModel):
    initial_olivine_mass_kg_ha: float = 1000.0
    initial_ph: float = 6.0
    initial_co2_amount: Optional[float] = None

# --- 3. Environmental Parameters ---
class EnvironmentalParameters(BaseModel):
    rainfall_mm_per_year: float = 750.0
    average_temperature_celsius: float = 10.7
    absolute_temperature_kelvin: float = 295.0
    reference_temperature_kelvin: float = 283.7
    atmospheric_co2_ppm: float = 410.0
    mixing_depth_m: float = 0.5
    soil_bulk_density_kg_l: float = 1.7
    porosity: float = 0.3

# --- 4. BLM Water Chemistry ---
class BlmWaterChemistry(BaseModel):
    dissolved_organic_carbon_mg_l: float = 5.0
    calcium_mg_l: float = 40.0
    eqs_surface_water_ni_ug_l: float = 4.0
    eqs_groundwater_ni_ug_l: float = 2.1

# --- 5. Geochemical Composition ---
class OxidesMassPercentage(BaseModel):
    SiO2: float = 40.86
    MgO: float = 35.11
    Fe2O3: float = 7.89
    Al2O3: float = 2.82
    MnO: float = 0.13
    Na2O: float = 0.06
    TiO2: float = 0.05
    CaO: float = 2.15
    K2O: float = 0.08
    P2O5: float = 0.01

class ElementsMgKg(BaseModel):
    Cr: float = 2281.0
    Ni: float = 1301.0
    Sr: float = 32.2
    Zr: float = 158.8
    Ba: float = 35.8

class NickelSorptionDistribution(BaseModel):
    iron_manganese_oxides: float = 28.0
    carbonates: float = 23.0
    organic_matter: float = 20.0
    exchangeable_phase: float = 29.0

class GeochemicalComposition(BaseModel):
    oxides_mass_percentage: OxidesMassPercentage = Field(default_factory=OxidesMassPercentage)
    elements_mg_kg: ElementsMgKg = Field(default_factory=ElementsMgKg)
    nickel_sorption_distribution_percentage: NickelSorptionDistribution = Field(default_factory=NickelSorptionDistribution)

# --- 6. Dissolution Kinetics ---
class OlsenEquations(BaseModel):
    low_ph_range: str = "log r = -0.48 * pH - 6.9 (for pH < 6)"
    high_ph_range: str = "log r = -0.18 * pH - 8.8 (for pH > 6)"

class DissolutionKinetics(BaseModel):
    rate_constant_ln_kT: float = 7.43e-11
    olsen_equations: OlsenEquations = Field(default_factory=OlsenEquations)

# --- 7. Experiment Settings ---
class ExperimentSettings(BaseModel):
    weathering_period_years: int = 40
    original_total_mass_kg: float = 1000.0

# --- 8. Observations Table 2 (Keep this dynamic) ---
class SizeClass(BaseModel):
    initial_diameter_um: int
    diameters_per_year: List[float]  # Changed to float in case of decimals
    original_mass_kg: float
    remaining_mass_kg: float
    dissolved_percentage: float

class ObservationsTable2(BaseModel):
    years: List[int]
    size_classes: List[SizeClass]

# --- MAIN MODEL: AppConfig ---
class AppConfig(BaseModel):
    # All nested models get themselves as default factory, except observations
    constants: PhysicalConstants = Field(default_factory=PhysicalConstants)
    model_info: ModelInfo = Field(default_factory=ModelInfo)
    initial_state_variables: InitialStateVariables = Field(default_factory=InitialStateVariables)
    environmental_parameters: EnvironmentalParameters = Field(default_factory=EnvironmentalParameters)
    blm_water_chemistry: BlmWaterChemistry = Field(default_factory=BlmWaterChemistry)
    geochemical_composition: GeochemicalComposition = Field(default_factory=GeochemicalComposition)
    dissolution_kinetics: DissolutionKinetics = Field(default_factory=DissolutionKinetics)
    experiment_settings: ExperimentSettings = Field(default_factory=ExperimentSettings)
    
    # Observations have no default, as they must be loaded from JSON!
    observations_table_2: ObservationsTable2