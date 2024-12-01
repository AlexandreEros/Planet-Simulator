import numpy as np

from vector_utils import deg2rad, rotate_vector, normalize, rotation_mat_x, rotation_mat_y, rotation_mat_z
from celestial_body import CelestialBody
from surface import Surface
from star import Star

class Planet(CelestialBody):
    def __init__(self, name: str, body_type: str, radius: float, mass: float,
                 sidereal_day: float, axial_tilt_deg: float, season_reference_axis_deg: float, color: str,
                 surface_data: dict, orbital_data: dict,
                 parent_mass: float, parent_position = np.zeros(3), parent_velocity = np.zeros(3), parent = None):

        try:
            super().__init__(name=name, body_type=body_type, mass=mass, color=color,
                             orbital_data=orbital_data, parent_mass=parent_mass,
                             parent_position=parent_position, parent_velocity=parent_velocity)
            self.parent = parent

            self.sidereal_day = sidereal_day
            self.axial_tilt = deg2rad(axial_tilt_deg)
            self.season_reference_axis = deg2rad(season_reference_axis_deg)

            self.radius = radius
            self.surface = Surface(self.radius, **surface_data)

            self.rotation_rate = 2*np.pi / self.sidereal_day
            self.axial_tilt_matrix = np.dot(rotation_mat_z(self.season_reference_axis), rotation_mat_x(self.axial_tilt))
            self.axial_tilt_matrix = np.dot(self.inclination_matrix, self.axial_tilt_matrix)
            self.rotation_axis = np.dot(self.axial_tilt_matrix, np.array([0, 0, 1], dtype = np.float64))
            self.rotation_axis /= np.linalg.norm(self.rotation_axis)  # Just to be sure
            self.angular_velocity = self.rotation_rate * self.rotation_axis
            self.current_angle = 0.0

            self.sunlight = self.position / np.linalg.norm(self.position)

        except Exception as err:
            raise Exception(f"Error in the constructor of `Planet`:\n{err}")


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
