import numpy as np
from scipy import sparse

from air_data import AirData
from adjacency_manager import AdjacencyManager
from surface import Surface

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
        atmospheric_density = self.air_data.density[layer_indices, vertex_indices]
        atmospheric_layer_thickness = self.air_data.altitudes[layer_indices] - self.air_data.altitudes[layer_indices - 1]

        # Heat flux between surface and atmosphere
        heat_flux = k_surface_atmosphere * (surface_temperature - atmospheric_temperature)

        # Update surface temperature
        surface_heat_loss = heat_flux  # Total energy lost by the surface
        self.surface.subsurface_temperature[:, 0] -= (
            surface_heat_loss / (self.surface.density * self.surface.specific_heat_capacity)
        ) * delta_t

        # Update atmospheric temperature
        is_vac = atmospheric_density < 1e-9
        self.air_data.temperature[layer_indices, vertex_indices][~is_vac] += heat_flux[~is_vac] / (
            specific_heat_air * (atmospheric_density * atmospheric_layer_thickness)[~is_vac]
        ) * delta_t

        # Ensure temperatures remain physical (e.g., above 0 K)
        self.air_data.temperature[~np.isnan(self.air_data.temperature)] = np.fmax(self.air_data.temperature[~np.isnan(self.air_data.temperature)], 0.0)
        self.surface.subsurface_temperature[:, 0] = np.fmax(self.surface.subsurface_temperature[:, 0], 0.0)


    def conduct_heat(self, delta_t: float):
        """
        Perform a conduction step using the sparse Laplacian matrix:
        T_new = T + Δt * D * L * T
        """
        k_air = self.material['thermal_conductivity']
        cp_air = self.material['isobaric_mass_heat_capacity']

        # D = diag(k/(ρcp)), where ρ varies by cell.
        D = sparse.diags(k_air / (self.air_data.density.ravel() * cp_air))

        T_flat = self.air_data.temperature.ravel()
        dT = delta_t * D.dot(self.adjacency.laplacian_matrix.dot(T_flat))
        T_new = np.fmax(T_flat + dT, 0.0)
        self.air_data.temperature = T_new.reshape(self.air_data.n_layers, self.surface.n_vertices)

