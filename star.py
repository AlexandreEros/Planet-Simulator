import numpy as np
from celestial_body import CelestialBody

class Star(CelestialBody):
    def __init__(self, name: str, body_type: str, mass: float,
                 power: float, color: str, orbital_data: dict):

        super().__init__(name=name, body_type=body_type, mass=mass, color=color,
                         orbital_data=orbital_data)

        self.power = power