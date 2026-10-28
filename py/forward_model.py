import json
import numpy as np

class OlivineDissolutionModel:
    def __init__(self, json_config_path: str, V_fluid: float = 1000.0):
        self.M_MOLAR = 0.1407  # Molaire massa forsteriet (kg/mol)
        self.V_fluid = V_fluid
        
        with open(json_config_path, "r") as f:
            self.config = json.load(f)

        self.initial_masses = np.array([g["mass_kg"] for g in self.config["grains"]])
        self.initial_areas = np.array([g["area_m2"] for g in self.config["grains"]])
        
        self.current_masses = self.initial_masses.copy()
        self.current_areas = self.initial_areas.copy()

    def step(self, dt_seconds: float):
        r_sp = 6.69e-10

        # Equation 3: M(t + dt) = max(0, M(t) - M_molar * A(t) * r * dt)
        dm = self.M_MOLAR * self.current_areas * r_sp * dt_seconds
        next_masses = np.maximum(0.0, self.current_masses - dm)

        # Equation 4: A(t + dt) = A(t) * (M(t + dt) / M(t))^(2/3)
        with np.errstate(divide='ignore', invalid='ignore'):
            mass_ratios = np.where(self.current_masses > 0, next_masses / self.current_masses, 0.0)
            next_areas = self.current_areas * (mass_ratios ** (2.0 / 3.0))

        # Update state variables for next timestep
        self.current_masses = next_masses
        self.current_areas = next_areas

    def get_state_vector(self) -> dict:
        total_mass = float(np.sum(self.current_masses))
        A_spec = float(np.sum(self.current_areas) / total_mass) if total_mass > 0 else 0.0
        
        return {
            "m_tot": total_mass,
            "A_spec_aggregate": A_spec
        }

    def run_simulation(self, total_years: float = 20.0, dt_years: float = 1.0):
        n_steps = int(total_years / dt_years) + 1
        time_years = np.linspace(0, total_years, n_steps)
        dt_seconds = dt_years * 365.25 * 86400.0

        mass_history = np.zeros(n_steps)
        aspec_history = np.zeros(n_steps)

        # Reset to initial conditions
        self.current_masses = self.initial_masses.copy()
        self.current_areas = self.initial_areas.copy()

        for t_idx in range(n_steps):
            state = self.get_state_vector()
            mass_history[t_idx] = state["m_tot"]
            aspec_history[t_idx] = state["A_spec_aggregate"]
            if t_idx < n_steps - 1:
                self.step(dt_seconds)

        return time_years, mass_history, aspec_history