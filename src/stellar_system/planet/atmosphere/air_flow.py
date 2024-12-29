import numpy as np

from src.math_utils.vector_utils import polar_to_cartesian_velocity, cartesian_to_polar_velocity
from .air_data import AirData
from .adjacency_manager import AdjacencyManager
from src.stellar_system.planet.surface import Surface



class AirFlow:
    def __init__(self, air_data: AirData, surface: Surface, adjacency: AdjacencyManager, angular_velocity_vector: np.ndarray):
        """
        Initializes the calculator with the given physical parameters.

        :param air_data: Instance of the AirData class containing information about the atmosphere and its properties.
        :param surface: Instance of the Surface class representing the planetary surface and relevant surface properties.
        :param adjacency: Instance of the AdjacencyManager class representing the geometry of the atmosphere.
        :param angular_velocity_vector: numpy array representing the planetary angular velocity vector [rad/s]
        """
        self.air_data = air_data
        self.surface = surface
        self.adjacency = adjacency
        self.omega = np.array(angular_velocity_vector)

        self.n_layers = self.air_data.n_layers
        self.n_columns = self.surface.n_vertices
        self.shape = (self.n_layers, self.n_columns)
        self.R_specific = self.air_data.R_specific
        self.is_underground = self.air_data.is_underground

        self.zonal_derivative, self.meridional_derivative, self.vertical_derivative = self.adjacency.vector_operators.partial_derivative_operators
        self.calculate_gradient = self.adjacency.vector_operators.calculate_gradient
        self.calculate_divergence = self.adjacency.vector_operators.calculate_divergence
        self.calculate_curl = self.adjacency.vector_operators.calculate_curl
        self.calculate_laplacian = self.adjacency.vector_operators.calculate_laplacian

        self.pressure_gradient = np.zeros((self.n_layers, self.n_columns, 3))
        self.temperature_gradient = np.zeros((self.n_layers, self.n_columns, 3))
        self.net_force = np.zeros((self.n_layers, self.n_columns, 3))

        self.velocity = np.zeros((self.n_layers, self.n_columns, 3))
        self.temperature_rate = np.zeros_like(self.air_data.temperature)
        self.density_rate = np.zeros_like(self.air_data.density)
        self.velocity_divergence = np.zeros((self.n_layers, self.n_columns))


    def update_gradients(self):
        self.pressure_gradient = self.calculate_gradient(self.air_data.pressure.ravel()).reshape(self.shape + (3,))
        self.temperature_gradient = self.calculate_gradient(self.air_data.temperature.ravel()).reshape(self.shape + (3,))
        self.velocity_divergence = self.calculate_divergence(self.velocity.reshape((-1,3))).reshape(self.shape)

    def apply_forces(self):
        self.update_gradients()
        gradient_force = -self.pressure_gradient

        down = np.array([0.0, 0.0, -1.0])
        weight = self.air_data.density[...,None] * self.air_data.g[:,None,None] * down

        velocity_cartesian = polar_to_cartesian_velocity(
            self.velocity[:,:,0].flatten(),
            self.velocity[:,:,1].flatten(),
            self.velocity[:,:,2].flatten(),
            self.adjacency.cartesian
        )
        coriolis_force_cartesian = -2 * self.air_data.density.reshape((-1,1)) * np.cross(self.omega, velocity_cartesian)
        coriolis_force = np.stack(cartesian_to_polar_velocity(coriolis_force_cartesian, self.adjacency.cartesian), axis=-1).reshape(self.shape + (3,))

        self.net_force = gradient_force + coriolis_force + weight
        self.net_force[self.is_underground] = 0.0

    def accelerate(self, delta_t: float):
        self.apply_forces()
        is_air = (self.air_data.density > 1e-9) & ~self.is_underground
        self.velocity[~is_air] = 0.0
        self.velocity[is_air] += delta_t * self.net_force[is_air] / self.air_data.density[is_air][...,None]

        self.temperature_rate[is_air] = -np.einsum('...i,...i->...', self.velocity[is_air], self.temperature_gradient[is_air])
        self.density_rate[is_air] = -self.air_data.density[is_air] * self.velocity_divergence[is_air]
        self.temperature_rate[~is_air] = 0.0
        self.density_rate[~is_air] = 0.0

    def change_air_data(self, delta_t: float):
        self.air_data.temperature += delta_t * self.temperature_rate
        self.air_data.density += delta_t * self.density_rate
        self.air_data.pressure[...] = self.air_data.density * self.air_data.R_specific * self.air_data.temperature

        self.air_data.temperature = np.fmax(self.air_data.temperature, 0.0)
        self.air_data.density = np.fmax(self.air_data.density, 0.0)
        self.air_data.pressure = np.fmax(self.air_data.pressure, 0.0)
