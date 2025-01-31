import numpy as np
from scipy import sparse

from .air_data import AirData
from .adjacency_manager import AdjacencyManager
from src.stellar_system.planet.surface import Surface


class Thermodynamics:
    def __init__(self, air_data: AirData, adjacency_manager: AdjacencyManager, surface: Surface, material: dict):
        self.air_data = air_data
        self.adjacency = adjacency_manager
        self.surface = surface
        self.material = material

        self.heat_flux_from_surface = np.zeros_like(self.air_data.temperature)
        self.temperature_rate = np.zeros(shape=(self.air_data.temperature.size,))


    def set_heat_from_surface(self):
        """
        Exchange heat between the surface and the lowest atmospheric layer.
        """
        # Constants and parameters
        k_surface_atmosphere = self.material['convective_heat_transfer_coefficient']  # W/m²·K

        # Heat flux between surface and atmosphere
        area = self.surface.vertex_area
        self.heat_flux_from_surface = k_surface_atmosphere * area * (self.surface.temperature - self.air_data.temperature[0])

    def exchange_heat_with_surface(self, delta_t: float):
        # Update surface temperature

        specific_heat_air = self.material['isobaric_mass_heat_capacity']  # J/kg·K
        area = self.surface.vertex_area

        ground_layer_thickness = self.surface.layer_depths[1] - self.surface.layer_depths[0]
        self.surface.subsurface_temperature[0] -= self.heat_flux_from_surface / (
            self.surface.density * self.surface.specific_heat_capacity * ground_layer_thickness * area
        ) * delta_t

        # Update atmospheric temperature
        atmospheric_layer_thickness = self.air_data.altitudes[1] - self.air_data.altitudes[0]
        self.air_data.temperature[0] += self.heat_flux_from_surface / (
                        self.air_data.density[0] * specific_heat_air * atmospheric_layer_thickness * area
        ) * delta_t

        # Ensure temperatures remain physical (e.g., above 0 K)
        self.air_data.temperature = np.fmax(self.air_data.temperature, 0.0)
        self.surface.subsurface_temperature[0] = np.fmax(self.surface.subsurface_temperature[0], 0.0)


    def set_temperature_rate(self):
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
        self.temperature_rate = D.dot(self.adjacency.laplacian_matrix.dot(T_flat))

    def conduct_heat(self, delta_t: float):
        T_flat = self.air_data.temperature.ravel()
        T_new = np.fmax(T_flat + delta_t * self.temperature_rate, 0.0)
        self.air_data.temperature = np.fmax(T_new.reshape(self.air_data.n_layers, self.surface.n_vertices), 0.0)
