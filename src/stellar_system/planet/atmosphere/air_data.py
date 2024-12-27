import numpy as np
from scipy import constants

from src.stellar_system.planet.surface import Surface

class AirData:
    def __init__(self, surface: Surface, planet_mass: float, material: dict, atmosphere_data: dict):
        # Initialize layer properties
        self.surface = surface
        self.planet_mass = planet_mass

        self.n_layers = atmosphere_data.get('n_layers', 16)
        self.n_vertices = self.surface.n_vertices
        
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
        self.T0 = np.mean(self.surface.temperature) # Blackbody temperature of the planet, and initial temperature of the surface

        # pressure(h) = exp(-h / scale_height_0) * self.surface_pressure
        scale_height_0 = self.R_specific * self.T0 / self.g0
        self.top_geopotential = -np.log(1e-2) * scale_height_0  # Height where pressure drops to 1% of its value at the surface
        self.bottom = np.amin(self.surface.elevation)
        self.altitudes = self.bottom + (self.top_geopotential-self.bottom) * np.linspace(0,1, num=self.n_layers+1)[1:] ** 2
        
        layer_temperatures = self.T0 + self.lapse_rate * self.altitudes
        self.temperature = np.full((self.n_layers, self.n_vertices), layer_temperatures[:,None])
        self.pressure = self.get_pressure(self.temperature)
        self.density = self.get_density(self.temperature, self.pressure)

        self.pressure_gradient = np.zeros((self.n_layers, self.n_vertices, 2))

        self.is_underground = self.altitudes[:, None] <= self.surface.elevation
        self.lowest_layer_above_surface = np.argmin(self.is_underground, axis=0)



    def get_pressure(self, temperature) -> np.ndarray:
        shp = (self.n_layers, self.n_vertices)
        pressure = np.zeros(shp)
        
        pressure[0, :] = self.surface_pressure
        
        g = self.grav(self.altitudes)  # Gravity at all altitudes
        scale_height = self.R_specific * temperature / g[:, None]
        
        # Calculate pressure for each layer using hydrostatic equilibrium
        for layer_idx in range(1, self.n_layers):
            delta_h = self.altitudes[layer_idx] - self.altitudes[layer_idx - 1]
            is_vac = scale_height[layer_idx] < 1e-15
        
            pressure[layer_idx][is_vac] = 0.0
            pressure[layer_idx][~is_vac] = pressure[layer_idx - 1][~is_vac] * np.exp(
                -delta_h / scale_height[layer_idx][~is_vac]
            )
        
        return pressure
    
    def get_density(self, temperature, pressure) -> np.ndarray:
        # Compute density using the ideal gas law
        density = np.zeros_like(pressure)
        is_vac = temperature == 0.0
        density[is_vac] = 0.0
        density[~is_vac] = pressure[~is_vac] / (self.R_specific * temperature[~is_vac])
        return density


    def grav(self, h) -> np.ndarray:
        """
        Calculate gravity at altitude h.
        :param h: Altitude in meters; distance from center of planet minus radius. Can be scalar or array.
        :return: Gravity in m/s²
        """
        return constants.G * self.planet_mass / (h + self.surface.radius) ** 2


    def update(self):
        self.temperature[self.is_underground] = self.surface.blackbody_temperature

        self.pressure = self.get_pressure(self.temperature)
        self.density = self.get_density(self.temperature, self.pressure)

        self.pressure[self.is_underground] = self.surface_pressure
        self.density[self.is_underground] = 1.0