import numpy as np
from scipy import constants

from .math_utils.vector_utils import deg2rad, rotate_vector, normalize, rotation_mat_x, rotation_mat_y, rotation_mat_z
from .celestial_body import CelestialBody
from .star import Star
from .components.surface import Surface
from .components.atmosphere import Atmosphere

class Planet(CelestialBody):
    def __init__(self, name: str, body_type: str, mass: float, color: str,
                 orbital_data: dict, rotation_data: dict, surface_data: dict, atmosphere_data: dict,
                 star: Star, parent: CelestialBody | None = None):

        self.star = star
        self.parent = self.star if parent is None else parent

        super().__init__(name=name, body_type=body_type, mass=mass, color=color,
                         orbital_data=orbital_data, parent_mass=self.parent.mass,
                         parent_position=self.parent.position, parent_velocity=self.parent.velocity)

        self.sidereal_day = rotation_data['sidereal_day']
        self.axial_tilt = deg2rad(rotation_data['axial_tilt_deg'])
        self.ecliptic_longitude_of_north_pole = 0.0 if 'ecliptic_longitude_of_north_pole_deg' not in rotation_data else deg2rad(rotation_data['ecliptic_longitude_of_north_pole_deg'])
        self.initial_season_rad = self.ecliptic_longitude_of_north_pole - self.argument_of_perihelion
        initial_longitude = 0.0 if 'subsolar_point_longitude' not in rotation_data else rotation_data['subsolar_point_longitude']
        self.current_angle = np.pi + self.initial_season_rad - deg2rad(initial_longitude)

        self.bond_albedo = surface_data['bond_albedo']
        semi_major_axis = self.semi_major_axis if self.body_type=='planet' else self.parent.semi_major_axis
        self.blackbody_temperature = ((1 - self.bond_albedo) * self.star.power /
                                      (16 * np.pi * constants.Stefan_Boltzmann * semi_major_axis**2)) ** (1/4)
        surface_data['blackbody_temperature'] = self.blackbody_temperature

        self.surface = Surface(**surface_data)

        self.radius = self.surface.radius

        self.is_airless = True
        if 'surface_pressure' in atmosphere_data and atmosphere_data['surface_pressure'] > 0.0:
            self.is_airless = False
            self.atmosphere = Atmosphere(self.surface, self.mass, atmosphere_data)
            self.surface.f_GH = self.atmosphere.air_data.f_GH

        self.rotation_rate = 2*np.pi / self.sidereal_day
        axial_tilt_matrix = rotation_mat_y(self.axial_tilt)
        axial_tilt_matrix = np.dot(rotation_mat_z(self.ecliptic_longitude_of_north_pole), axial_tilt_matrix)
        self.axial_tilt_matrix = np.dot(self.inclination_matrix, axial_tilt_matrix)
        self.rotation_axis = np.dot(self.axial_tilt_matrix, np.array([0, 0, 1], dtype = np.float64))
        self.rotation_axis /= np.linalg.norm(self.rotation_axis)  # Just to be sure
        self.angular_velocity = self.rotation_rate * self.rotation_axis

        self.sunlight = self.position / np.linalg.norm(self.position)


    def update_sunlight(self, delta_t: float, star: Star):
        self.current_angle = (self.current_angle + self.rotation_rate * delta_t) % (2*np.pi)
        try:
            absolute_sunlight_unit_vector = normalize(self.position - star.position)
        except RuntimeWarning:
            raise ZeroDivisionError("Division by zero in `Planet.update_sunlight`; attribute `Planet.position` has "
                                    "a magnitude of zero, and an attempt was made to normalize it.")

        r = np.linalg.norm(self.position - star.position)
        solar_flux = star.power / (4 * np.pi * r ** 2)
        absolute_sunlight_vector = solar_flux * absolute_sunlight_unit_vector

        # Convert to the planet's rotating frame of reference
        self.sunlight = rotate_vector(absolute_sunlight_vector, self.rotation_axis, -self.current_angle)
        self.sunlight = np.dot(self.sunlight, self.axial_tilt_matrix)

        self.surface.update_irradiance(self.sunlight)


    def update_temperature(self, delta_t: float):
        self.surface.update_temperature(delta_t)
        if not self.is_airless:
            self.atmosphere.update(delta_t)
