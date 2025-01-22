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
        self.update_gradients()
        gradient_force = -self.pressure_gradient

        weight = self.air_data.density[...,None] * self.air_data.g[...,None] * np.array([0.0, 0.0, -1.0])

        coriolis_force = -2 * self.air_data.density[...,None] * np.cross(self.omega_spherical, self.velocity)

        viscous_force = self.dynamic_viscosity * self.velocity_laplacian

        self.net_force = gradient_force + weight + coriolis_force + viscous_force


    def rk4_step(self, delta_t):
        """
        Perform one 4th-order Runge-Kutta step to update velocity, temperature, and density.
        """

        # 1) Save copies of your current state
        v0 = self.velocity.copy()  # shape: (n_layers, n_columns, 3)
        T0 = self.air_data.temperature.copy()  # shape: (n_layers, n_columns)
        rho0 = self.air_data.density.copy()  # shape: (n_layers, n_columns)

        # ---------------------------------------------------------------------
        # K1
        dv1, dT1, dRho1 = self.get_tendencies()  # Evaluate derivatives at (v0, T0, rho0)

        # 2) Make a "predicted" state for the midpoint (dt/2) step
        self.velocity = v0 + 0.5 * delta_t * dv1
        self.air_data.temperature = T0 + 0.5 * delta_t * dT1
        self.air_data.density = rho0 + 0.5 * delta_t * dRho1
        # K2
        dv2, dT2, dRho2 = self.get_tendencies()

        # 3) Another midpoint
        self.velocity = v0 + 0.5 * delta_t * dv2
        self.air_data.temperature = T0 + 0.5 * delta_t * dT2
        self.air_data.density = rho0 + 0.5 * delta_t * dRho2
        # K3
        dv3, dT3, dRho3 = self.get_tendencies()

        # 4) Full step
        self.velocity = v0 + delta_t * dv3
        self.air_data.temperature = T0 + delta_t * dT3
        self.air_data.density = rho0 + delta_t * dRho3
        # K4
        dv4, dT4, dRho4 = self.get_tendencies()

        # ---------------------------------------------------------------------
        # Combine them
        self.velocity = v0 + (delta_t / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        self.air_data.temperature = T0 + (delta_t / 6.0) * (dT1 + 2.0 * dT2 + 2.0 * dT3 + dT4)
        self.air_data.density = rho0 + (delta_t / 6.0) * (dRho1 + 2.0 * dRho2 + 2.0 * dRho3 + dRho4)

        # ---------------------------------------------------------------------
        # Enforce constraints such as:
        #  - No-slip boundary conditions
        self.velocity[0] = 0.0  # Example no-slip at the bottom layer
        self.velocity[...,2] = 0.0
        #  - Non-negativity of density & temperature
        self.air_data.temperature = np.fmax(self.air_data.temperature, 0.0)
        self.air_data.density = np.fmax(self.air_data.density, 0.0)
        #  - Mass conservation
        total_mass_before = np.sum(rho0)
        total_mass_after = np.sum(self.air_data.density)
        self.air_data.density *= (total_mass_before / total_mass_after)

        # Update pressure
        self.air_data.pressure = (self.air_data.density
                                  * self.air_data.R_specific
                                  * self.air_data.temperature)

    def get_tendencies(self):
        """
        Returns:
          dvdt: (n_layers, n_columns, 3)
          dTdt: (n_layers, n_columns)
          dRhodt: (n_layers, n_columns)
        """

        # 1) For the current self.velocity, self.air_data.temperature, etc.,
        #    compute the gradients, net_force, etc. But do not modify self.* permanently!
        self.update_gradients()
        self.apply_forces()   # This sets self.net_force, etc., based on the CURRENT velocity/pressure

        # 2) The acceleration step: dv/dt = F / density
        is_air = (self.air_data.density > 1e-9)
        dvdt = np.zeros_like(self.velocity)
        dvdt[is_air] = self.net_force[is_air] / self.air_data.density[is_air][...,None]

        # 3) The thermodynamic tendencies
        #    temperature_rate = - velocity dot grad(T)
        #    density_rate = - density * div(velocity)
        dTdt = -np.einsum('...i,...i->...', self.velocity, self.temperature_gradient)
        dRhodt = -(self.air_data.density * self.velocity_divergence)

        return dvdt, dTdt, dRhodt
