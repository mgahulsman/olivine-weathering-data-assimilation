from pydantic import BaseModel

class Diameter(BaseModel):
    original_diameter: float # micro meter
    original_mass: float # kg
    remaining_mass: float # kg
    dissolved: float # percentage


class Observation(BaseModel):
    year: int
    diameters: list[float]
    total_remaining_mass: float | None = None # Totale resterende massa in kg

class TrueData(BaseModel):
    original_masses: list[float] # De startmassa per fractie (in kg)
    observations: list[Observation]

# Data from Table 2
true_data = TrueData(
    original_masses=[55, 109, 144, 243, 199, 102, 49, 33, 37, 13, 16],
    observations=[
        Observation(
            year=0,
            diameters=[2000, 1410, 1000, 707, 420, 250, 150, 88, 50, 16, 8],
            total_remaining_mass=1000.0
        ),
        Observation(
            year=1,
            diameters=[2000, 1410, 999, 706, 419, 249, 149, 87, 49, 15, 7],
            total_remaining_mass=None
        ),
        Observation(
            year=3,
            diameters=[1993, 1407, 993, 700, 413, 243, 143, 81, 43, 9, 1],
            total_remaining_mass=None
        ),
        Observation(
            year=5,
            diameters=[1989, 1403, 989, 696, 409, 239, 139, 77, 39, 5, 0],
            total_remaining_mass=None
        ),
        Observation(
            year=10,
            diameters=[1982, 1398, 982, 689, 402, 232, 132, 70, 32, 0, 0],
            total_remaining_mass=None
        ),
        Observation(
            year=15,
            diameters=[1971, 1394, 971, 678, 391, 221, 121, 59, 21, 0, 0],
            total_remaining_mass=None
        ),
        Observation(
            year=30,
            diameters=[1946, 1377, 946, 653, 366, 196, 96, 34, 0, 0, 0],
            total_remaining_mass=None
        ),
        Observation(
            year=40,
            diameters=[1928, 1361, 928, 635, 348, 178, 78, 16, 0, 0, 0],
            total_remaining_mass=519.0
        )
    ]
)