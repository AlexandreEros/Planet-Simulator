import numpy as np
from scipy import constants
from scipy.interpolate import interp1d

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
        self.T0 = np.mean(self.surface.temperature) # Blackbody temperature of the planet, and initial temperature of the surface

        scale_height_0 = self.R_specific * self.T0 / self.g0
        self.top_geopotential = -np.log(1e-2) * scale_height_0  # Height where pressure drops to 1% of its value at the surface
        self.bottom = np.amin(self.surface.elevation)
        self.altitudes = self.surface.elevation + (self.top_geopotential-surface.elevation) * np.linspace(
            np.zeros(self.surface.elevation.shape),
            np.ones(self.surface.elevation.shape),
            num=self.n_layers
        ) ** 2
        self.g = self.grav(self.altitudes)  # Gravity at all altitudes
        self.geopotential = constants.G * planet_mass * (-1 / (self.altitudes + self.surface.radius) + 1 / self.surface.radius)
        self.all_altitudes = np.linspace(self.bottom, self.top_geopotential, num=50).reshape((-1,1))

        self.coordinates = np.zeros((self.n_layers, self.n_columns, 3))
        for layer_idx in range(self.n_layers):
            self.coordinates[layer_idx] = self.surface.coordinates  # longitude and latitude
            self.coordinates[layer_idx, :, 2] = self.altitudes[layer_idx]  # altitude

        self.all_temperatures = self.T0 + self.lapse_rate * self.all_altitudes
        self.all_pressures = self.get_all_pressures(self.all_altitudes)
        self.all_densities = self.get_density(self.all_temperatures, self.all_pressures)
        
        self.temperature = self.T0 + self.lapse_rate * self.altitudes
        # self.pressure = self.get_pressure(self.temperature)
        # Interpolate pressures from self.all_pressures to self.altitudes
        interpolate_pressure = interp1d(
            self.all_altitudes.flatten(),  # Flatten to ensure 1D array for interpolation
            self.all_pressures.flatten(),
            kind='linear',
            fill_value="extrapolate"
        )
        self.pressure = np.array(interpolate_pressure(self.altitudes))
        self.density = self.get_density(self.temperature, self.pressure)


    def get_all_pressures(self, all_altitudes) -> np.ndarray:
        # Initialize pressure array
        all_pressures = np.zeros_like(all_altitudes)
        # Pressure at the lowest point is 1.0
        all_pressures[0] = 1.0
        # Calculate scale height for each altitude
        g = self.grav(all_altitudes)
        scale_height = self.R_specific * self.all_temperatures / g

        # Calculate pressure for each altitude using hydrostatic equilibrium
        for i in range(1, len(all_altitudes)):
            delta_h = all_altitudes[i] - all_altitudes[i - 1]
            is_vac = scale_height[i] < 1e-15
            if is_vac:
                all_pressures[i] = 0.0
            else:
                all_pressures[i] = all_pressures[i - 1] * np.exp(-delta_h / scale_height[i])

        # Interpolate to find the relative pressure at altitude = 0
        interp_relative_pressure = interp1d(
            all_altitudes.flatten(),  # Flatten to ensure 1D array for interpolation
            all_pressures.flatten(),
            kind='linear',
            fill_value="extrapolate"
        )
        relative_pressure_at_0 = interp_relative_pressure(0)
        # Scale the pressures to match actual surface pressure
        all_pressures *= self.surface_pressure / relative_pressure_at_0

        return all_pressures
    
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
