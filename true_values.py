from pydantic import BaseModel

    
class Observation(BaseModel):
    initial_diam: int # micro meter (We don't need this since it's in diam_over_years, but we might want to use it as id)
    original_mass: int # kg
    remaining_mass: float # kg
    percentage_dissolved: float # %
    diam_over_years: dict[int, int] # key: years, value: diameters (micro meter)


TRUE_VALUES: list[Observation] = [
    Observation(
        initial_diam=2000,
        original_mass=55,
        remaining_mass=48.0,
        percentage_dissolved=13.0,
        diam_over_years={
            0: 2000,
            1: 2000,
            3: 1993,
            5: 1989,
            10: 1982,
            15: 1971,
            30: 1946,
            40: 1928,
        }
    ),
    Observation(
        initial_diam=1410,
        original_mass=109,
        remaining_mass=90.0,
        percentage_dissolved=17.0,
        diam_over_years={
            0: 1410,
            1: 1410,
            3: 1407,
            5: 1403,
            10: 1398,
            15: 1394,
            30: 1377,
            40: 1361,
        }
    ),
    Observation(
        initial_diam=1000,
        original_mass=144,
        remaining_mass=110.0,
        percentage_dissolved=24.0,
        diam_over_years={
            0: 1000,
            1: 999,
            3: 993,
            5: 989,
            10: 982,
            15: 971,
            30: 946,
            40: 928,
        }
    ),
    Observation(
        initial_diam=707,
        original_mass=243,
        remaining_mass=156.0,
        percentage_dissolved=36.0,
        diam_over_years={
            0: 707,
            1: 706,
            3: 700,
            5: 696,
            10: 689,
            15: 678,
            30: 653,
            40: 635,
        }
    ),
    Observation(
        initial_diam=420,
        original_mass=199,
        remaining_mass=90.0,
        percentage_dissolved=55.0,
        diam_over_years={
            0: 420,
            1: 419,
            3: 413,
            5: 409,
            10: 402,
            15: 391,
            30: 366,
            40: 348,
        }
    ),
    Observation(
        initial_diam=250,
        original_mass=102,
        remaining_mass=23.0,
        percentage_dissolved=77.0,
        diam_over_years={
            0: 250,
            1: 249,
            3: 243,
            5: 239,
            10: 232,
            15: 221,
            30: 196,
            40: 178,
        }
    ),
    Observation(
        initial_diam=150,
        original_mass=49,
        remaining_mass=2.0,
        percentage_dissolved=97.0,
        diam_over_years={
            0: 150,
            1: 149,
            3: 143,
            5: 139,
            10: 132,
            15: 121,
            30: 96,
            40: 78,
        }
    ),
    Observation(
        initial_diam=88,
        original_mass=33,
        remaining_mass=0.0,
        percentage_dissolved=100.0,
        diam_over_years={
            0: 88,
            1: 87,
            3: 81,
            5: 77,
            10: 70,
            15: 59,
            30: 34,
            40: 16,
        }
    ),
    Observation(
        initial_diam=50,
        original_mass=37,
        remaining_mass=0.0,
        percentage_dissolved=100.0,
        diam_over_years={
            0: 50,
            1: 49,
            3: 43,
            5: 39,
            10: 32,
            15: 21,
            30: 0,
            40: 0,
        }
    ),
    Observation(
        initial_diam=16,
        original_mass=13,
        remaining_mass=0.0,
        percentage_dissolved=100.0,
        diam_over_years={
            0: 16,
            1: 15,
            3: 9,
            5: 5,
            10: 0,
            15: 0,
            30: 0,
            40: 0,
        }
    ),
    Observation(
        initial_diam=8,
        original_mass=16,
        remaining_mass=0.0,
        percentage_dissolved=100.0,
        diam_over_years={
            0: 8,
            1: 7,
            3: 1,
            5: 0,
            10: 0,
            15: 0,
            30: 0,
            40: 0,
        }
    )
]