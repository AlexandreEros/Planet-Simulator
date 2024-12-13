import numpy as np

from .celestial_body import CelestialBody
from .planet import Planet

class Satellite(Planet):
    def __init__(self, name: str, body_type: str, mass: float, color: str,
                 orbital_data: dict, rotation_data: dict, surface_data: dict, atmosphere_data: dict,
                 planet: Planet, parent: CelestialBody | None = None):
        self.planet = planet
        self.parent = self.planet if parent is None else parent

        # Converting phase into "true anomaly":
        planet_angular_position_deg = np.rad2deg(self.planet.true_anomaly + self.planet.argument_of_perihelion)
        orbital_data['true_anomaly_deg'] = orbital_data['phase_deg'] + (planet_angular_position_deg + 180) % 360 - orbital_data['argument_of_perihelion_deg']
        # Assuming all moons are tidally locked by default:
        if 'sidereal_day' not in rotation_data: rotation_data['sidereal_day'] = orbital_data['orbital_period']
        # Assuming longitude 0 is at the center of the face visible from the planet:
        rotation_data['subsolar_point_longitude'] = 0.0

        super().__init__(name, body_type, mass, color,
                         orbital_data, rotation_data, surface_data, atmosphere_data,
                         self.planet.star, parent=self.planet)
