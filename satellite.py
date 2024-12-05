import numpy as np
from scipy import constants

from celestial_body import CelestialBody
from star import Star
from planet import Planet

class Satellite(Planet):
    def __init__(self, name: str, body_type: str, mass: float, sidereal_day: float,
                 axial_tilt_deg: float, season_reference_axis_deg: float, color: str, surface_data: dict,
                 orbital_data: dict, planet: Planet, parent: CelestialBody | None = None):
        self.planet = planet
        self.parent = self.planet if parent is None else parent
        super().__init__(name, body_type, mass, color,
                         sidereal_day, axial_tilt_deg, season_reference_axis_deg,
                         surface_data, orbital_data,
                         self.planet.star, parent=self.planet)
