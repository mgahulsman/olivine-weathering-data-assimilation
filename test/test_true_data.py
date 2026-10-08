import pytest

from oop_implementation import Environment, Model
from true_values import TRUE_VALUES


def test_full_model():
    env = Environment()
    model = Model(environment=env, true_values=TRUE_VALUES)
    model.run_model()

    for obs in TRUE_VALUES:
        initial_d = obs.initial_diam
        # TODO: unclear whether this history things exists
        model_diameters = model.history.get(initial_d)
        
        for year, expected_diam in obs.diam_over_years.items():
            if model_diameters and year < len(model_diameters):
                actual_diam = model_diameters[year]

                # TODO: Maak de tolerantie kleiner en kleiner
                assert actual_diam == pytest.approx(expected_diam, abs=100)

# TODO, after test is implemented
#   1. Fix that the calculate_shrunk_diameter gets implemented
#   2. Implement a small pH increament since the olivine increases pH
#   3. Introduce a factor for the other unknown things which you can optimize