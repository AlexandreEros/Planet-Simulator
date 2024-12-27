import numpy as np
from scipy import sparse

from .air_data import AirData
from .adjacency_manager import AdjacencyManager
from src.stellar_system.planet.surface import Surface
from src.math_utils.vector_utils import cartesian_to_spherical

class Thermodynamics:
    def __init__(self, air_data: AirData, adjacency_manager: AdjacencyManager, surface: Surface, material: dict):
        self.air_data = air_data
        self.adjacency = adjacency_manager
        self.surface = surface
        self.material = material

    def exchange_heat_with_surface(self, delta_t: float):
        """
        Exchange heat between the surface and the lowest atmospheric layer.

        :param delta_t: Time step in seconds.
        """
        # Constants and parameters
        k_surface_atmosphere = self.material['convective_heat_transfer_coefficient']  # W/m²·K

        specific_heat_air = self.material['isobaric_mass_heat_capacity']  # J/kg·K

        surface_temperature = self.surface.temperature

        layer_indices = self.air_data.lowest_layer_above_surface
        vertex_indices = np.arange(self.surface.n_vertices)

        # Lowest atmospheric layer properties
        atmospheric_temperature = self.air_data.temperature[layer_indices, vertex_indices]

        # Heat flux between surface and atmosphere
        area = self.surface.vertex_area
        heat_flux_up = k_surface_atmosphere * area * (surface_temperature - atmospheric_temperature)

        # Update surface temperature
        ground_layer_thickness = self.surface.layer_depths[1] - self.surface.layer_depths[0]
        self.surface.subsurface_temperature[:, 0] -= (
            heat_flux_up / (self.surface.density * self.surface.specific_heat_capacity * ground_layer_thickness * area)
        ) * delta_t

        # Update atmospheric temperature
        atmospheric_density = self.air_data.density[layer_indices, vertex_indices]
        atmospheric_layer_thickness = self.air_data.altitudes[layer_indices] - self.air_data.altitudes[layer_indices - 1]
        increment = (heat_flux_up /
                     (atmospheric_density * specific_heat_air * atmospheric_layer_thickness * area)
                     ) * delta_t
        np.add.at(self.air_data.temperature, (layer_indices, vertex_indices), increment)

        # Ensure temperatures remain physical (e.g., above 0 K)
        self.air_data.temperature = np.fmax(self.air_data.temperature, 0.0)
        self.surface.subsurface_temperature[:, 0] = np.fmax(self.surface.subsurface_temperature[:, 0], 0.0)


    def conduct_heat(self, delta_t: float):
        """
        Perform a conduction step using the sparse Laplacian matrix:
        T_new = T + Δt * D * L * T
        """
        k_air = self.material['thermal_conductivity']
        cp_air = self.material['isobaric_mass_heat_capacity']

        # D = diag(k/(ρcp)), where ρ varies by cell.
        density_flat = np.where(self.air_data.density>0, self.air_data.density, 1.0).ravel()
        D = sparse.diags(k_air / (density_flat * cp_air))

        T_flat = self.air_data.temperature.ravel()
        dT = delta_t * D.dot(-self.adjacency.laplacian_matrix.dot(T_flat))
        T_new = np.fmax(T_flat + dT, 0.0)
        self.air_data.temperature = T_new.reshape(self.air_data.n_layers, self.surface.n_vertices)




    def calculate_pressure_gradient(self, layer_id = -1):
        p = self.air_data.pressure[layer_id]
        adjacency_matrix = self.surface.adjacency_matrix
        dx = self.surface.dx
        dy = self.surface.dy
        dz = self.surface.dz

        # Ensure p is a column vector
        p = p.reshape(-1, 1)  # Shape: (N_vert, 1)

        # Element-wise multiply difference matrices with adjacency_matrix
        dx_adj = dx.multiply(adjacency_matrix)  # (N_vert, N_vert)
        dy_adj = dy.multiply(adjacency_matrix)  # (N_vert, N_vert)
        dz_adj = dz.multiply(adjacency_matrix)  # (N_vert, N_vert)

        # Compute (dx_adj * p), (dy_adj * p), (dz_adj * p)
        grad_p_x_contrib = dx_adj.dot(p)  # Shape: (N_vert, 1)
        grad_p_y_contrib = dy_adj.dot(p)  # Shape: (N_vert, 1)
        grad_p_z_contrib = dz_adj.dot(p)  # Shape: (N_vert, 1)

        # Compute sums for each row: dx_adj.sum(axis=1), etc.
        dx_adj_sum = dx_adj.sum(axis=1)  # Shape: (N_vert, 1) as a matrix
        dy_adj_sum = dy_adj.sum(axis=1)
        dz_adj_sum = dz_adj.sum(axis=1)

        # Convert sums to flat arrays for element-wise multiplication
        dx_adj_sum = np.array(dx_adj_sum).flatten().reshape(-1, 1)  # (N_vert, 1)
        dy_adj_sum = np.array(dy_adj_sum).flatten().reshape(-1, 1)
        dz_adj_sum = np.array(dz_adj_sum).flatten().reshape(-1, 1)

        # Compute p * (dx_adj_sum), etc.
        grad_p_x_self = p * dx_adj_sum  # (N_vert, 1)
        grad_p_y_self = p * dy_adj_sum
        grad_p_z_self = p * dz_adj_sum

        # Final gradient components: contributions minus self-contributions
        grad_p_x = grad_p_x_contrib - grad_p_x_self  # (N_vert, 1)
        grad_p_y = grad_p_y_contrib - grad_p_y_self
        grad_p_z = grad_p_z_contrib - grad_p_z_self

        # Flatten the results and stack into a single (N_vert, 3) array
        grad_p_x = grad_p_x.flatten()
        grad_p_y = grad_p_y.flatten()
        grad_p_z = grad_p_z.flatten()

        pressure_gradient = np.vstack((grad_p_x, grad_p_y, grad_p_z)).T  # Shape: (N_vert, 3)

        return pressure_gradient

    def cartesian_gradient_to_spherical(self, pressure_gradient):
        """
        Convert Cartesian pressure gradient to spherical coordinates.

        Parameters:
        - pressure_gradient: (N_vert, 3) array of Cartesian pressure gradients.
        - vertices: (N_vert, 3) array of Cartesian coordinates.

        Returns:
        - grad_p_r: (N_vert,) array of radial pressure gradients.
        - grad_p_theta: (N_vert,) array of polar pressure gradients.
        - grad_p_phi: (N_vert,) array of azimuthal pressure gradients.
        """
        vertices = self.surface.vertices

        # Convert Cartesian coordinates to spherical coordinates
        coordinates = np.apply_along_axis(cartesian_to_spherical, -1, vertices)
        phi, theta, r = coordinates[:,0], coordinates[:,1], coordinates[:,2]

        # Extract Cartesian gradient components
        dp_dx = pressure_gradient[:, 0]
        dp_dy = pressure_gradient[:, 1]
        dp_dz = pressure_gradient[:, 2]

        # Compute trigonometric functions
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        # Avoid division by zero at poles for phi components
        sin_theta_nonzero = sin_theta.copy()
        sin_theta_nonzero[sin_theta_nonzero == 0] = 1e-10  # small number to prevent division by zero

        # Compute spherical gradient components
        grad_p_r = sin_theta * cos_phi * dp_dx + sin_theta * sin_phi * dp_dy + cos_theta * dp_dz
        grad_p_theta = cos_theta * cos_phi * dp_dx + cos_theta * sin_phi * dp_dy - sin_theta * dp_dz
        grad_p_phi = -sin_phi * dp_dx + cos_phi * dp_dy  # Note: independent of theta

        return grad_p_r, grad_p_theta, grad_p_phi
