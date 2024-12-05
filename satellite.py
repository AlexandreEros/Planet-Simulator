import numpy as np
from scipy import constants

from celestial_body import CelestialBody
from star import Star
from planet import Planet

class Satellite(Planet):
    def __init__(self, name: str, body_type: str, radius: float, mass: float, sidereal_day: float,
                 axial_tilt_deg: float, season_reference_axis_deg: float, color: str, surface_data: dict,
                 orbital_data: dict, planet: Planet, parent: CelestialBody | None = None):
        self.planet = planet
        self.parent = self.planet if parent is None else parent
        super().__init__(name, body_type, radius, mass, sidereal_day, axial_tilt_deg, season_reference_axis_deg, color,
                         surface_data, orbital_data, self.planet.star, parent=self.planet)

        # self.bond_albedo = surface_data['albedo']
        # self.blackbody_temperature = ((1 - self.bond_albedo) * self.planet.star.power /
        #                               (16 * np.pi * constants.Stefan_Boltzmann * self.planet.semi_major_axis ** 2)) ** (1/4)
        # self.surface.blackbody_temperature = self.blackbody_temperature * np.cos(np.radians(0.99*self.surface.coordinates[:, 0])) ** (1 / 4)
        # self.surface.subsurface_temperature = np.full((len(self.surface.vertices), self.surface.n_layers),
        #                                               self.surface.blackbody_temperature[:,None], dtype=np.float64)
