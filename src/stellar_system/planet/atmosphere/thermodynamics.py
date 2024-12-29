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
        atmospheric_layer_thickness = self.air_data.altitudes[layer_indices] - np.where(layer_indices==0, 0.0, self.air_data.altitudes[layer_indices - 1])
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
