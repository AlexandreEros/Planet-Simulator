import numpy as np
from vector_utils import rotate_vector, deg2rad
from scipy.spatial.transform import Rotation

class CelestialBody:
    def __init__(self, name: str, body_type: str, position: np.ndarray, velocity: np.ndarray, mass: float, color: str,
                 orbital_period: float = None, eccentricity: float = 0.0, year_percentage: float = 0.0,
                 argument_of_perihelion_deg: float = 0.0, inclination_deg: float = 0.0, lon_ascending_node_deg: float = 0.0):
        self.name = name
        self.body_type = body_type
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.color = color
        self.orbital_period = orbital_period
        self.eccentricity = eccentricity
        self.initial_year_percentage = year_percentage
        self.argument_of_perihelion = deg2rad(argument_of_perihelion_deg)
        self.inclination = deg2rad(inclination_deg)
        self.lon_ascending_node = deg2rad(lon_ascending_node_deg)

        ascending_node_vec = np.array([np.cos(self.lon_ascending_node), np.sin(self.lon_ascending_node), 0.0])
        self.inclination_matrix = Rotation.from_rotvec(ascending_node_vec * self.inclination).as_matrix()

        self.net_force = np.zeros((3,), dtype = np.float64)


    def apply_force(self, force):
        self.net_force = force

    def accelerate(self, delta_t):
        self.velocity += self.net_force * delta_t / self.mass

    def move(self, delta_t) -> None:
        self.position += self.velocity * delta_t

    @property
    def current_angular_momentum(self):
        return np.cross(self.position, self.mass * self.velocity)
