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
        self.dynamic_viscosity = self.air_data.material['dynamic_viscosity']

        self.omega_spherical = np.array(polar_to_cartesian_velocity(*angular_velocity_vector,
                                                                    self.adjacency.cartesian)).reshape((self.n_layers, self.n_columns, 3,))
        self.cos_lat = np.cos(np.deg2rad(self.air_data.coordinates[...,1])).reshape((self.n_layers, self.n_columns))

        self.zonal_derivative, self.meridional_derivative, self.vertical_derivative = self.adjacency.vector_operators.partial_derivative_operators
        self.calculate_gradient = self.adjacency.vector_operators.calculate_gradient
        self.calculate_divergence = self.adjacency.vector_operators.calculate_divergence
        self.calculate_curl = self.adjacency.vector_operators.calculate_curl
        self.calculate_laplacian = self.adjacency.vector_operators.calculate_laplacian
        self.calculate_gradient_tensor = self.adjacency.vector_operators.calculate_vector_gradient

        self.pressure_gradient = np.zeros((self.n_layers, self.n_columns, 3))
        self.temperature_gradient = np.zeros((self.n_layers, self.n_columns, 3))
        self.net_force = np.zeros((self.n_layers, self.n_columns, 3))

        self.velocity = np.zeros((self.n_layers, self.n_columns, 3))
        self.temperature_rate = np.zeros_like(self.air_data.temperature)
        self.density_rate = np.zeros_like(self.air_data.density)
        self.velocity_divergence = np.zeros((self.n_layers, self.n_columns))
        self.velocity_laplacian = np.zeros((self.n_layers, self.n_columns, 3))
        self.velocity_gradient_tensor = np.zeros((self.n_layers, self.n_columns, 3, 3))


    def update_gradients(self):
        self.pressure_gradient = self.calculate_gradient(self.air_data.pressure.ravel()).reshape(self.shape + (3,))
        self.temperature_gradient = self.calculate_gradient(self.air_data.temperature.ravel()).reshape(self.shape + (3,))
        self.velocity_divergence = self.calculate_divergence(self.velocity.reshape((-1,3))).reshape(self.shape)
        self.velocity_laplacian = self.calculate_laplacian(self.velocity)
        self.velocity_gradient_tensor = self.calculate_gradient_tensor(self.velocity)

    def apply_forces(self):
        # ρ(u · ∇)u
        advection = 0.0 #self.air_data.density[...,None] * np.einsum('ijk, ijkl -> ijl',
                        #                                            self.velocity,
                        #                                                    self.velocity_gradient_tensor
                        #                                                )

        self.update_gradients()
        gradient_force = -self.pressure_gradient

        weight = self.air_data.density[...,None] * self.air_data.g[...,None] * np.array([0.0, 0.0, -1.0])

        coriolis_force = -2 * self.air_data.density[...,None] * np.cross(self.omega_spherical, self.velocity)

        viscous_force = self.dynamic_viscosity * self.velocity_laplacian

        self.net_force = advection + gradient_force + weight + coriolis_force + viscous_force


    def accelerate(self, delta_t: float):
        self.apply_forces()
        is_air = (self.air_data.density > 1e-9)
        self.velocity[~is_air] = 0.0
        self.velocity[is_air] += delta_t * self.net_force[is_air] / self.air_data.density[is_air][...,None]
        self.velocity[0] = 0.0

        # # Bottom layer
        # vel0 = polar_to_cartesian_velocity(self.velocity[0,:,0], self.velocity[0,:,1], self.velocity[0,:,2], self.surface.vertices)
        # normal_mag = np.einsum('...i,...i -> ...', vel0, self.surface.normals)
        # normal_component = normal_mag[...,None] * self.surface.normals
        # self.velocity[0] -= np.where(normal_mag[...,None]>0, normal_component, 0.0)

        self.temperature_rate[...] = -np.einsum('...i,...i->...', self.velocity, self.temperature_gradient)
        self.density_rate[...] = -self.air_data.density * self.velocity_divergence


    def change_air_data(self, delta_t: float):
        self.air_data.temperature += delta_t * self.temperature_rate
        initial_mass = np.sum(self.air_data.density)
        self.air_data.density += delta_t * self.density_rate
        later_mass = np.sum(self.air_data.density)
        self.air_data.density *= initial_mass / later_mass
        self.air_data.pressure[...] = self.air_data.density * self.air_data.R_specific * self.air_data.temperature

        # wrong_idx = np.argwhere((self.air_data.temperature.ravel() > 1e3) | (self.air_data.temperature.ravel() < 0.0))
        # if wrong_idx.size > 0:
        #     raise ValueError(f'impossible temperature at {wrong_idx=}:'
        #                      f'\n{self.air_data.coordinates.reshape((-1,3))[wrong_idx]=};'
        #                      f'\n{self.air_data.temperature.ravel()[wrong_idx]=};'
        #                      f'\n{self.temperature_rate.ravel()[wrong_idx]=};'
        #                      f'\n{self.velocity.reshape((-1,3))[wrong_idx]=};'
        #                      f'\n{self.temperature_gradient.reshape((-1,3))[wrong_idx]=};')

        self.air_data.temperature[...] = np.fmax(self.air_data.temperature, 0.0)
        self.air_data.density[...] = np.fmax(self.air_data.density, 0.0)
        self.air_data.pressure[...] = np.fmax(self.air_data.pressure, 0.0)
