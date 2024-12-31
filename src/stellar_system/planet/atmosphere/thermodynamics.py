import cupy as cp
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

        self.heat_flux_from_surface = cp.zeros_like(self.air_data.temperature)
        self.temperature_rate = cp.zeros(shape=(self.air_data.temperature.size,))


    def set_heat_from_surface(self):
        """
        Exchange heat between the surface and the lowest atmospheric layer.
        """
        # Constants and parameters
        k_surface_atmosphere = self.material['convective_heat_transfer_coefficient']  # W/m²·K

        surface_temperature = self.surface.temperature

        layer_indices = cp.array([0,])  # self.air_data.lowest_layer_above_surface
        vertex_indices = cp.arange(self.surface.n_vertices)

        # Lowest atmospheric layer properties
        atmospheric_temperature = self.air_data.temperature[layer_indices, vertex_indices]

        # Heat flux between surface and atmosphere
        area = self.surface.vertex_area
        self.heat_flux_from_surface = k_surface_atmosphere * area * (surface_temperature - atmospheric_temperature)

    def exchange_heat_with_surface(self, delta_t: float):
        # Update surface temperature

        specific_heat_air = self.material['isobaric_mass_heat_capacity']  # J/kg·K
        area = self.surface.vertex_area

        layer_indices = cp.array([0,])  # self.air_data.lowest_layer_above_surface
        vertex_indices = cp.arange(self.surface.n_vertices)

        ground_layer_thickness = self.surface.layer_depths[1] - self.surface.layer_depths[0]
        self.surface.subsurface_temperature[:, 0] -= (
            self.heat_flux_from_surface / (self.surface.density * self.surface.specific_heat_capacity * ground_layer_thickness * area)
        ) * delta_t

        # Update atmospheric temperature
        atmospheric_density = self.air_data.density[layer_indices, vertex_indices]
        atmospheric_layer_thickness = self.air_data.altitudes[1]
        increment = self.heat_flux_from_surface / (
                        atmospheric_density * specific_heat_air * atmospheric_layer_thickness * area
                    ) * delta_t
        cp.add.at(self.air_data.temperature, (layer_indices, vertex_indices), increment)

        # Ensure temperatures remain physical (e.g., above 0 K)
        self.air_data.temperature = cp.fmax(self.air_data.temperature, 0.0)
        self.surface.subsurface_temperature[:, 0] = cp.fmax(self.surface.subsurface_temperature[:, 0], 0.0)


    def set_temperature_rate(self):
        """
        Perform a conduction step using the sparse Laplacian matrix:
        T_new = T + Δt * D * L * T
        """
        k_air = self.material['thermal_conductivity']
        cp_air = self.material['isobaric_mass_heat_capacity']

        # D = diag(k/(ρcp)), where ρ varies by cell.
        density_flat = cp.where(self.air_data.density>0, self.air_data.density, 1.0).ravel()
        D = sparse.diags(k_air / (density_flat * cp_air))

        T_flat = self.air_data.temperature.ravel()
        self.temperature_rate = D.dot(-self.adjacency.laplacian_matrix.dot(T_flat))

    def conduct_heat(self, delta_t: float):
        T_flat = self.air_data.temperature.ravel()
        T_new = cp.fmax(T_flat + delta_t * self.temperature_rate, 0.0)
        self.air_data.temperature = cp.fmax(T_new.reshape(self.air_data.n_layers, self.surface.n_vertices), 0.0)
