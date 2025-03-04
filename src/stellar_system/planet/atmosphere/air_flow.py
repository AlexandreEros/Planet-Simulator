import numpy as np

from src.math_utils.vector_utils import cartesian_to_polar_velocity
from .reference_atmosphere import ReferenceAtmosphere
from .adjacency_manager import AdjacencyManager
from src.stellar_system.planet.surface import Surface

class AirFlow:
    def __init__(self, ref: ReferenceAtmosphere, surface: Surface, adjacency: AdjacencyManager, angular_velocity_vector: np.ndarray):
        """
        Initializes the calculator with the given physical parameters.

        :param ref: Instance of the ReferenceAtmosphere class containing information about the atmosphere and its properties.
        :param surface: Instance of the Surface class representing the planetary surface and relevant surface properties.
        :param adjacency: Instance of the AdjacencyManager class representing the geometry of the atmosphere.
        :param angular_velocity_vector: numpy array representing the planetary angular velocity vector [rad/s]
        """
        self.ref = ref
        self.surface = surface
        self.adjacency = adjacency
        self.omega = np.array(angular_velocity_vector)

        self.n_layers = self.ref.n_layers
        self.n_columns = self.surface.n_vertices
        self.shape = (self.n_layers, self.n_columns)
        self.R_specific = self.ref.R_specific
        self.dynamic_viscosity = self.ref.material['dynamic_viscosity']

        self.omega_spherical = np.stack(np.array(cartesian_to_polar_velocity(angular_velocity_vector,
                                                                            self.adjacency.cartesian)),
                                        axis=-1).reshape((self.n_layers, self.n_columns, 3))

        self.zonal_derivative, self.meridional_derivative, self.vertical_derivative = self.adjacency.vector_operators.partial_derivative_operators
        self.calculate_gradient = lambda x: self.adjacency.vector_operators.calculate_gradient(x.ravel()).reshape(x.shape+(3,))
        self.calculate_divergence = lambda x: self.adjacency.vector_operators.calculate_divergence(x.reshape((-1,3))).reshape(x.shape[:-1])
        self.calculate_curl = lambda x: self.adjacency.vector_operators.calculate_curl(x.reshape((-1,3))).reshape(x.shape)
        self.calculate_laplacian = lambda x: self.adjacency.vector_operators.calculate_laplacian(x.reshape((-1,3))).reshape(x.shape)
        self.calculate_gradient_tensor = lambda x: self.adjacency.vector_operators.calculate_vector_gradient(x.reshape((-1,3))).reshape(x.shape+(3,))

        self.set_hydrostatic_equilibrium()

        self.velocity = np.zeros((self.n_layers, self.n_columns, 3))

        self.temperature_prt = np.zeros_like(self.ref.temperature)
        self.pressure_prt = np.zeros_like(self.ref.pressure)
        self.density_prt = np.zeros_like(self.ref.density)

        # Damping coefficients
        self.nu_div = 1e-1         # Divergence damping coefficient
        self.alpha_rayleigh = 1e-1 * (self.ref.altitude - self.ref.altitude[0]) / (
                self.ref.altitude[-1] - self.ref.altitude[0]
        ) # Rayleigh friction coefficient


    @property
    def temperature(self):
        return self.ref.temperature + self.temperature_prt
    
    @property
    def pressure(self):
        return self.ref.pressure + self.pressure_prt
    
    @property
    def density(self):
        return self.ref.density + self.density_prt
    

    def set_hydrostatic_equilibrium(self):
        dpdh = self.vertical_derivative.dot(self.ref.pressure.ravel()).reshape(self.ref.pressure.shape)
        self.ref.density = -dpdh / self.ref.g

        self.ref.temperature = self.ref.pressure / (self.ref.density * self.R_specific)


    def update(self, delta_t):
        self.rk4_step(delta_t)

        # Update pressure using the ideal gas law
        self.pressure_prt = self.density * self.R_specific * self.temperature - self.ref.pressure

        # Filtering vertical component
        self.velocity[..., 2] = self.vertical_spectral_filter(self.velocity[..., 2], 0.3)

        # No-slip boundary conditions
        self.velocity[0] = 0.0  # No-slip at the surface


    def rk4_step(self, delta_t):
        """
        Perform one 4th-order Runge-Kutta step to update velocity, temperature, and density.
        """
        # Save the current state
        v0 = self.velocity.copy()  # shape: (n_layers, n_columns, 3)
        T0 = self.temperature  # shape: (n_layers, n_columns)
        rho0 = self.density  # shape: (n_layers, n_columns)

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
        self.temperature_prt = T0 + (delta_t / 6.0) * (dT1 + 2.0 * dT2 + 2.0 * dT3 + dT4) - self.ref.temperature
        self.density_prt = rho0 + (delta_t / 6.0) * (dRho1 + 2.0 * dRho2 + 2.0 * dRho3 + dRho4) - self.ref.density


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
        pressure_gradient = self.calculate_gradient(pressure)
        geopotential_gradient = self.calculate_gradient(self.ref.geopotential)
        temperature_gradient = self.calculate_gradient(temperature)
        velocity_divergence = self.calculate_divergence(velocity)
        velocity_laplacian = self.calculate_laplacian(velocity)

        # Compute net forces
        gradient_force = -pressure_gradient
        weight = -density[..., None] * geopotential_gradient
        vel_without_z = np.zeros(velocity.shape)
        vel_without_z[..., :2] = velocity[..., :2]
        coriolis_force = -2 * density[..., None] * np.cross(self.omega_spherical, vel_without_z)
        viscous_force = self.dynamic_viscosity * velocity_laplacian
        net_force = gradient_force + weight + coriolis_force + viscous_force

        velocity_gradient_tensor = self.calculate_gradient_tensor(velocity)
        convective_acceleration = np.einsum('...j,...ij->...i', velocity, velocity_gradient_tensor)

        # Compute tendencies
        is_air = (density > 1e-9)
        dvdt = np.zeros_like(velocity)
        dvdt[is_air] = net_force[is_air] / density[is_air][..., None] - convective_acceleration[is_air]

        # Divergence damping
        grad_divergence = self.calculate_gradient(velocity_divergence.ravel()).reshape(self.shape + (3,))
        dvdt[is_air] += - self.nu_div * grad_divergence[is_air]
        dvdt[..., 2] += - self.nu_div * grad_divergence[..., 2]  # Stronger damping in vertical

        # Rayleigh friction
        dvdt[is_air] += - self.alpha_rayleigh[is_air][...,None] * velocity[is_air]

        dTdt = -np.einsum('...i,...i->...', velocity, temperature_gradient)
        dRhodt = -density * velocity_divergence

        return dvdt, dTdt, dRhodt


    @staticmethod
    def vertical_spectral_filter(field, cutoff=0.2):
        """
        Applies a spectral filter to remove high-frequency vertical oscillations.

        :param field: 3D numpy array (n_layers, n_columns)
        :param cutoff: Fraction of max wavenumber to retain
        :return: Filtered field
        """
        # Take 1D FFT in the vertical direction
        field_fft = np.fft.fft(field, axis=0)

        # Create vertical wavenumber grid
        nz = field.shape[0]
        kz = np.fft.fftfreq(nz) * 2 * np.pi  # Vertical wavenumbers

        # Apply filter (zero out high wavenumbers)
        filter_mask = np.abs(kz) < cutoff * np.max(kz)
        field_fft_filtered = field_fft * filter_mask[:, None]

        # Inverse FFT to get back to physical space
        field_filtered = np.fft.ifft(field_fft_filtered, axis=0).real
        return field_filtered
