import numpy as np
from scipy import sparse

from .reference_atmosphere import ReferenceAtmosphere
from .air_flow import AirFlow
from .adjacency_manager import AdjacencyManager
from src.stellar_system.planet.surface import Surface


class Thermodynamics:
    def __init__(self, air_flow: AirFlow, ref: ReferenceAtmosphere, adjacency_manager: AdjacencyManager, surface: Surface, material: dict):
        self.air_flow = air_flow
        self.ref = ref
        self.adjacency = adjacency_manager
        self.surface = surface
        self.material = material


    def exchange_heat_with_surface(self, delta_t: float):
        """
        Exchange heat between the surface and the lowest atmospheric layer.
        """
        # Constants and parameters
        k_surface_atmosphere = self.material['convective_heat_transfer_coefficient']  # W/m²·K

        # Heat flux between surface and atmosphere
        area = self.surface.vertex_area
        heat_flux_from_surface = k_surface_atmosphere * area * (self.surface.temperature - self.ref.temperature[0])

        # Update surface temperature
        specific_heat_air = self.material['isobaric_mass_heat_capacity']  # J/kg·K

        ground_layer_thickness = self.surface.layer_depths[1] - self.surface.layer_depths[0]
        self.surface.subsurface_temperature[0] -= heat_flux_from_surface / (
            self.surface.density * self.surface.specific_heat_capacity * ground_layer_thickness * area
        ) * delta_t

        # Update atmospheric temperature
        atmospheric_layer_thickness = self.ref.altitude[1] - self.ref.altitude[0]
        self.air_flow.temperature_prt[0] += heat_flux_from_surface / (
                        self.ref.density[0] * specific_heat_air * atmospheric_layer_thickness * area
        ) * delta_t


    def conduct_heat(self, delta_t: float):
        """
        Perform a conduction step using the sparse Laplacian matrix:
        T_new = T + Δt * D * L * T
        """
        k_air = self.material['thermal_conductivity']
        cp_air = self.material['isobaric_mass_heat_capacity']

        # D = diag(k/(ρcp)), where ρ varies by cell.
        density_flat = np.where(self.ref.density>0, self.ref.density, 1.0).ravel()
        D = sparse.diags(k_air / (density_flat * cp_air))

        T_flat = self.ref.temperature.ravel()
        temperature_rate = D.dot(self.adjacency.laplacian_matrix.dot(T_flat))

        self.air_flow.temperature_prt += delta_t * temperature_rate.reshape(self.ref.temperature.shape)
