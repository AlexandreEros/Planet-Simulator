import cupy as cp
import numpy as np
from scipy import constants

from src.stellar_system.planet.surface import Surface

class AirData:
    def __init__(self, surface: Surface, planet_mass: float, material: dict, atmosphere_data: dict):
        # Initialize layer properties
        self.surface = surface
        self.planet_mass = planet_mass

        self.n_layers = atmosphere_data.get('n_layers', 16)
        self.n_columns = self.surface.n_vertices
        
        self.material = material
        self.molar_mass = 0.02896 if 'molar_mass' not in self.material else self.material['molar_mass']
        self.cp = self.material['isobaric_mass_heat_capacity']

        self.surface_pressure = 0.0 if 'surface_pressure' not in atmosphere_data else atmosphere_data['surface_pressure']
        self.k_GH = 1e-5 if 'greenhouse_efficiency' not in atmosphere_data else atmosphere_data['greenhouse_efficiency']
        self.chi_CO2 = 1.0 if 'chi_CO2' not in atmosphere_data else atmosphere_data['chi_CO2']  # CO2 dominance
        self.f_GH = self.k_GH * self.surface_pressure * (self.chi_CO2 / self.molar_mass)  # Greenhouse factor

        self.R_specific = constants.R / self.material['molar_mass']  # Specific gas constant (J/kg·K)
        self.g0 = self.grav(0)
        self.lapse_rate = -self.g0 / self.cp  # Adiabatic dry lapse rate as a function of geopotential altitude
        self.T0 = cp.mean(self.surface.temperature) # Blackbody temperature of the planet, and initial temperature of the surface

        scale_height_0 = self.R_specific * self.T0 / self.g0
        self.top_geopotential = -np.log(1e-2) * scale_height_0  # Height where pressure drops to 1% of its value at the surface
        self.bottom = cp.amin(self.surface.elevation)
        self.altitudes = self.surface.elevation + (self.top_geopotential-surface.elevation) * cp.linspace(
            cp.zeros(self.surface.elevation.shape),
            cp.ones(self.surface.elevation.shape),
            num=self.n_layers
        ) ** 2
        self.g = self.grav(self.altitudes)  # Gravity at all altitudes

        self.coordinates = cp.zeros((self.n_layers, self.n_columns, 3))
        for layer_idx in range(self.n_layers):
            self.coordinates[layer_idx] = self.surface.coordinates  # longitude and latitude
            self.coordinates[layer_idx, :, 2] = self.altitudes[layer_idx]  # altitude
        
        self.temperature = self.T0 + self.lapse_rate * self.altitudes
        self.pressure = self.get_pressure(self.temperature)
        self.density = self.get_density(self.temperature, self.pressure)


    def get_pressure(self, temperature) -> cp.ndarray:
        shp = (self.n_layers, self.n_columns)
        pressure = cp.zeros(shp)
        
        pressure[0, :] = self.surface_pressure

        scale_height = self.R_specific * temperature / self.g
        
        # Calculate pressure for each layer using hydrostatic equilibrium
        for layer_idx in range(1, self.n_layers):
            delta_h = self.altitudes[layer_idx] - self.altitudes[layer_idx - 1]
            is_vac = scale_height[layer_idx] < 1e-15
            pressure[layer_idx][is_vac] = pressure[layer_idx - 1, is_vac]
            pressure[layer_idx][~is_vac] = pressure[layer_idx - 1][~is_vac] * cp.exp(
                -delta_h[~is_vac] / scale_height[layer_idx][~is_vac]
            )
        
        return pressure
    
    def get_density(self, temperature: cp.ndarray, pressure: cp.ndarray) -> cp.ndarray:
        # Compute density using the ideal gas law
        density = cp.zeros_like(pressure)
        is_vac = temperature == 0.0
        density[is_vac] = 0.0
        density[~is_vac] = pressure[~is_vac] / (self.R_specific * temperature[~is_vac])
        return density


    def grav(self, h) -> cp.ndarray:
        """
        Calculate gravity at altitude h.
        :param h: Altitude in meters; distance from center of planet minus radius. Can be scalar or array.
        :return: Magnitude of gravity in m/s²
        """
        return constants.G * self.planet_mass / (h + self.surface.radius) ** 2


    def update(self):
        self.pressure = self.get_pressure(self.temperature)
        self.density = self.get_density(self.temperature, self.pressure)