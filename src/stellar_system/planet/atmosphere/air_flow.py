import numpy as np

from src.math_utils.vector_utils import cartesian_to_polar_velocity
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

        self.omega_spherical = np.stack(np.array(cartesian_to_polar_velocity(angular_velocity_vector,
                                                                            self.adjacency.cartesian)),
                                        axis=-1).reshape((self.n_layers, self.n_columns, 3))

        self.zonal_derivative, self.meridional_derivative, self.vertical_derivative = self.adjacency.vector_operators.partial_derivative_operators
        self.calculate_gradient = self.adjacency.vector_operators.calculate_gradient
        self.calculate_divergence = self.adjacency.vector_operators.calculate_divergence
        self.calculate_curl = self.adjacency.vector_operators.calculate_curl
        self.calculate_laplacian = self.adjacency.vector_operators.calculate_laplacian
        self.calculate_gradient_tensor = self.adjacency.vector_operators.calculate_vector_gradient

        self.set_hydrostatic_equilibrium()

        self.pressure_gradient = np.zeros((self.n_layers, self.n_columns, 3))
        self.temperature_gradient = np.zeros((self.n_layers, self.n_columns, 3))
        self.net_force = np.zeros((self.n_layers, self.n_columns, 3))

        self.velocity = np.zeros((self.n_layers, self.n_columns, 3))
        self.temperature_rate = np.zeros_like(self.air_data.temperature)
        self.density_rate = np.zeros_like(self.air_data.density)
        self.velocity_divergence = np.zeros((self.n_layers, self.n_columns))
        self.velocity_laplacian = np.zeros((self.n_layers, self.n_columns, 3))
        self.velocity_gradient_tensor = np.zeros((self.n_layers, self.n_columns, 3, 3))

        # Damping coefficients
        self.nu_div = 1e-2         # Divergence damping coefficient
        self.alpha_rayleigh = 1e-1 * (self.air_data.altitudes - self.air_data.altitudes[0]) / (
            self.air_data.altitudes[-1] - self.air_data.altitudes[0]
        ) # Rayleigh friction coefficient


    def set_hydrostatic_equilibrium(self):
        dpdh = self.vertical_derivative.dot(self.air_data.pressure.ravel()).reshape(self.air_data.pressure.shape)
        self.air_data.density = -dpdh / self.air_data.g

        self.air_data.temperature = self.air_data.pressure / (self.air_data.density * self.R_specific)


    def update_gradients(self):
        self.pressure_gradient = self.calculate_gradient(self.air_data.pressure.ravel()).reshape(self.shape + (3,))
        self.temperature_gradient = self.calculate_gradient(self.air_data.temperature.ravel()).reshape(self.shape + (3,))
        self.velocity_divergence = self.calculate_divergence(self.velocity.reshape((-1,3))).reshape(self.shape)
        self.velocity_laplacian = self.calculate_laplacian(self.velocity)
        # self.velocity_gradient_tensor = self.calculate_gradient_tensor(self.velocity)


    def rk4_step(self, delta_t):
        """
        Perform one 4th-order Runge-Kutta step to update velocity, temperature, and density.
        """
        # Save the current state
        v0 = self.velocity.copy()  # shape: (n_layers, n_columns, 3)
        T0 = self.air_data.temperature.copy()  # shape: (n_layers, n_columns)
        rho0 = self.air_data.density.copy()  # shape: (n_layers, n_columns)

        # Compute RK4 coefficients (k1, k2, k3, k4)
        # --------------------------------------------
        # Stage 1: k1
        dv1, dT1, dRho1 = self.get_tendencies(v0, T0, rho0)

        # Stage 2: k2
        v_temp = v0 + 0.5 * delta_t * dv1
        T_temp = T0 + 0.5 * delta_t * dT1
        rho_temp = rho0 + 0.5 * delta_t * dRho1
        dv2, dT2, dRho2 = self.get_tendencies(v_temp, T_temp, rho_temp)

        # Stage 3: k3
        v_temp = v0 + 0.5 * delta_t * dv2
        T_temp = T0 + 0.5 * delta_t * dT2
        rho_temp = rho0 + 0.5 * delta_t * dRho2
        dv3, dT3, dRho3 = self.get_tendencies(v_temp, T_temp, rho_temp)

        # Stage 4: k4
        v_temp = v0 + delta_t * dv3
        T_temp = T0 + delta_t * dT3
        rho_temp = rho0 + delta_t * dRho3
        dv4, dT4, dRho4 = self.get_tendencies(v_temp, T_temp, rho_temp)

        # Combine the coefficients to update the state
        # ----------------------------------------------
        self.velocity = v0 + (delta_t / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
        self.air_data.temperature = T0 + (delta_t / 6.0) * (dT1 + 2.0 * dT2 + 2.0 * dT3 + dT4)
        self.air_data.density = rho0 + (delta_t / 6.0) * (dRho1 + 2.0 * dRho2 + 2.0 * dRho3 + dRho4)

        # Mass conservation
        total_mass_before = np.sum(rho0)
        total_mass_after = np.sum(self.air_data.density)
        self.air_data.density *= (total_mass_before / total_mass_after)

        # Update pressure using the ideal gas law
        old_pressure = self.air_data.pressure.copy()
        self.air_data.pressure = self.air_data.density * self.R_specific * self.air_data.temperature

        # Omega equation for vertical flow
        omega = (self.air_data.pressure - old_pressure) / delta_t
        self.velocity[..., 2] = omega / (self.air_data.density * self.air_data.g)
        self.velocity[..., 2] = np.fmin(self.velocity[..., 2], 1.0)
        self.velocity[..., 2] = np.fmax(self.velocity[..., 2], -1.0)

        # No-slip boundary conditions
        self.velocity[0] = 0.0  # No-slip at the surface



    def get_tendencies(self, velocity, temperature, density):
        """
        Compute tendencies for velocity, temperature, and density.

        Args:
            velocity: Current velocity field (n_layers, n_columns, 3)
            temperature: Current temperature field (n_layers, n_columns)
            density: Current density field (n_layers, n_columns)

        Returns:
            dvdt: Velocity tendency (n_layers, n_columns, 3)
            dTdt: Temperature tendency (n_layers, n_columns)
            dRhodt: Density tendency (n_layers, n_columns)
        """
        # Compute gradients and forces based on the input state
        pressure = density * self.R_specific * temperature
        pressure_gradient = self.calculate_gradient(pressure.ravel()).reshape(self.shape + (3,))
        temperature_gradient = self.calculate_gradient(temperature.ravel()).reshape(self.shape + (3,))
        velocity_divergence = self.calculate_divergence(velocity.reshape((-1, 3))).reshape(self.shape)
        velocity_laplacian = self.calculate_laplacian(velocity)

        # Compute net forces
        gradient_force = -pressure_gradient
        weight = density[..., None] * self.air_data.g[..., None] * np.array([0.0, 0.0, -1.0])
        vel_without_z = np.zeros(velocity.shape)
        vel_without_z[..., :2] = velocity[..., :2]
        coriolis_force = -2 * density[..., None] * np.cross(self.omega_spherical, vel_without_z)
        viscous_force = self.dynamic_viscosity * velocity_laplacian
        net_force = gradient_force + weight + coriolis_force + viscous_force

        self.velocity_gradient_tensor = self.calculate_gradient_tensor(self.velocity)
        convective_acceleration = np.einsum('...j,...ij->...i', self.velocity, self.velocity_gradient_tensor)

        # Compute tendencies
        is_air = (density > 1e-9)
        dvdt = np.zeros_like(velocity)
        dvdt[is_air] = net_force[is_air] / density[is_air][..., None] - convective_acceleration[is_air]

        # Divergence damping
        grad_divergence = self.calculate_gradient(velocity_divergence.ravel()).reshape(self.shape + (3,))
        dvdt[is_air] += - self.nu_div * grad_divergence[is_air]

        # Rayleigh friction
        dvdt[is_air] += - self.alpha_rayleigh[is_air][...,None] * velocity[is_air]

        dTdt = -np.einsum('...i,...i->...', velocity, temperature_gradient)
        dRhodt = -density * velocity_divergence

        return dvdt, dTdt, dRhodt

