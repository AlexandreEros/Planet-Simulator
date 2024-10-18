import numpy as np
from celestial_body import CelestialBody

class Star(CelestialBody):
    def __init__(self, name: str, body_type: str, position: np.ndarray, velocity: np.ndarray, mass: float,
                 power: float, color: str,
                 orbital_period: float = None, eccentricity: float = 0.0, year_percentage: float = 0.0,
                 argument_of_perihelion_deg: float = 0.0):

        super().__init__(name=name, body_type=body_type, position=position, velocity=velocity, mass=mass, color=color,
                 orbital_period=orbital_period, eccentricity=eccentricity, year_percentage=year_percentage,
                 argument_of_perihelion_deg=argument_of_perihelion_deg)

        self.power = power