import numpy as np
from vector_utils import deg2rad, rotate_vector, rotation_mat_x, rotation_mat_y, rotation_mat_z
from celestial_body import CelestialBody

class Planet(CelestialBody):
    def __init__(self, name: str, body_type: str, position: np.ndarray, velocity: np.ndarray, mass: float,
                 sidereal_day: float, axial_tilt_deg: float, longitude_ascending_node_deg: float, color: str,
                 orbital_period: float = None, eccentricity: float = 0.0, year_percentage: float = 0.0,
                 argument_of_perihelion_deg: float = 0.0):

        super().__init__(name=name, body_type=body_type, position=position, velocity=velocity, mass=mass, color=color,
                 orbital_period=orbital_period, eccentricity=eccentricity, year_percentage=year_percentage,
                 argument_of_perihelion_deg=argument_of_perihelion_deg)

        self.sidereal_day = sidereal_day
        self.axial_tilt = deg2rad(axial_tilt_deg)
        self.longitude_ascending_node = deg2rad(longitude_ascending_node_deg)

        self.rotation_rate = 2*np.pi / self.sidereal_day
        self.axial_tilt_matrix = np.dot(rotation_mat_z(self.longitude_ascending_node), rotation_mat_x(self.axial_tilt))
        self.rotation_axis = np.dot(self.axial_tilt_matrix, np.array([0, 0, 1], dtype = np.float64))
        self.rotation_axis /= np.linalg.norm(self.rotation_axis)  # Just to be sure
        # print(f"{self.name}'s rotation axis: {self.rotation_axis}")


        self.angular_velocity = self.rotation_rate * self.rotation_axis

        self.current_angle = 0.0
        self.sunlight = self.position / np.linalg.norm(self.position)


    def update_sunlight(self, delta_t: float, irradiance: float):
        self.current_angle = (self.current_angle + self.rotation_rate * delta_t) % (2*np.pi)
        absolute_sunlight_unit_vector = self.position / np.linalg.norm(self.position)
        absolute_sunlight_vector = irradiance * absolute_sunlight_unit_vector
        self.sunlight = rotate_vector(absolute_sunlight_vector, self.rotation_axis, -self.current_angle)
        self.sunlight = np.dot(self.sunlight, self.axial_tilt_matrix)
        # print(f"{self.name}, {24*self.current_angle/(2*np.pi)} h: irradiance = {irradiance}")
