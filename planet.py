import numpy as np

from vector_utils import deg2rad, rotate_vector, rotation_mat_x, rotation_mat_y, rotation_mat_z
from celestial_body import CelestialBody
from surface import Surface

class Planet(CelestialBody):
    def __init__(self, name: str, body_type: str, radius: float, position: np.ndarray, velocity: np.ndarray, mass: float,
                 sidereal_day: float, axial_tilt_deg: float, longitude_ascending_node_deg: float, color: str,
                 surface_data: dict,
                 orbital_period: float = None, eccentricity: float = 0.0, year_percentage: float = 0.0,
                 argument_of_perihelion_deg: float = 0.0,):
                 #resolution: int = 0, noise_scale: float = 1.0, noise_octaves: int = 4, noise_amplitude: float = 0.05):

        try:
            super().__init__(name=name, body_type=body_type, position=position, velocity=velocity, mass=mass, color=color,
                     orbital_period=orbital_period, eccentricity=eccentricity, year_percentage=year_percentage,
                     argument_of_perihelion_deg=argument_of_perihelion_deg)

            self.sidereal_day = sidereal_day
            self.axial_tilt = deg2rad(axial_tilt_deg)
            self.longitude_ascending_node = deg2rad(longitude_ascending_node_deg)

            self.radius = radius
            self.surface = Surface(self.radius, **surface_data)

            self.rotation_rate = 2*np.pi / self.sidereal_day
            self.axial_tilt_matrix = np.dot(rotation_mat_z(self.longitude_ascending_node), rotation_mat_x(self.axial_tilt))
            self.rotation_axis = np.dot(self.axial_tilt_matrix, np.array([0, 0, 1], dtype = np.float64))
            self.rotation_axis /= np.linalg.norm(self.rotation_axis)  # Just to be sure
            self.angular_velocity = self.rotation_rate * self.rotation_axis
            self.current_angle = 0.0

            self.sunlight = self.position / np.linalg.norm(self.position)
            self.irradiance = np.ndarray(shape=len(self.surface.vertices), dtype=np.float64)

        except Exception as err:
            raise Exception(f"Error in the constructor of `Planet`:\n{err}")


    def update_sunlight(self, delta_t: float, total_irradiance: float):
        self.current_angle = (self.current_angle + self.rotation_rate * delta_t) % (2*np.pi)
        try:
            absolute_sunlight_unit_vector = self.position / np.linalg.norm(self.position)
        except RuntimeWarning:
            raise ZeroDivisionError("Division by zero in `Planet.update_sunlight`; attribute `Planet.position` has "
                                    "a magnitude of zero, and an attempt was made to normalize it.")
        absolute_sunlight_vector = total_irradiance * absolute_sunlight_unit_vector

        # Convert to the planet's rotating frame of reference
        self.sunlight = rotate_vector(absolute_sunlight_vector, self.rotation_axis, -self.current_angle)
        self.sunlight = np.dot(self.sunlight, self.axial_tilt_matrix)
        # print(f"{self.name}, {24*self.current_angle/(2*np.pi)} h: sunlight = {self.sunlight}")

        try:
            self.irradiance = -np.einsum('j, ij -> i', self.sunlight, self.surface.normals)
            self.irradiance = np.fmax(self.irradiance, 0.0)
        except Exception as err:
            raise Exception(f"Error calculating dot products in `Planet.update_sunlight`:\n{err}")

        # print(f"{self.name}, {24*self.current_angle/(2*np.pi)} h: {np.mean(self.irradiance)=}")
