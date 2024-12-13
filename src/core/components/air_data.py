import numpy as np
from scipy import constants

from .surface import Surface

class AirData:
    def __init__(self, surface: Surface, planet_mass, material: dict, atmosphere_data: dict):
        # Initialize layer properties
        self.surface = surface
        self.planet_mass = planet_mass
        
        self.material = material
        self.molar_mass = 0.02896 if 'molar_mass' not in self.material else self.material['molar_mass']

        self.n_layers = atmosphere_data.get('n_layers', 16)
        self.n_vertices = self.surface.n_vertices


        self.surface_pressure = 0.0 if 'surface_pressure' not in atmosphere_data else atmosphere_data['surface_pressure']
        self.lapse_rate = 0.0 if 'lapse_rate' not in atmosphere_data else atmosphere_data['lapse_rate']
        self.k_GH = 1e-5 if 'greenhouse_efficiency' not in atmosphere_data else atmosphere_data['greenhouse_efficiency']
        self.chi_CO2 = 1.0 if 'chi_CO2' not in atmosphere_data else atmosphere_data['chi_CO2']  # CO2 dominance
        self.f_GH = self.k_GH * self.surface_pressure * (self.chi_CO2 / self.molar_mass)  # Greenhouse factor

        top_of_atmosphere = self.surface.radius / 64
        initial_altitudes = np.linspace(
            np.amin(self.surface.elevation),
            np.amin(self.surface.elevation) + top_of_atmosphere,
            num=self.n_layers
        )
        self.altitudes, self.pressure, self.density, self.temperature = self.initialize_layers(initial_altitudes)

        self.is_underground = self.altitudes[:, None] <= self.surface.elevation
        self.lowest_layer_above_surface = np.argmin(self.is_underground, axis=0)

    def initialize_layers(self, altitudes, max_iterations=100, tol=1e-6) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R_specific = constants.R / self.material['molar_mass']  # Specific gas constant (J/kg·K)
        delta_h = np.diff(altitudes, prepend=0)  # Height differences between layers

        # Initialize arrays
        shp = (self.n_layers, self.n_vertices)
        pressure = np.full(shp, self.surface_pressure)  # Start with surface pressure
        temperature = np.full(shp, self.surface.temperature)  # Initialize temperature
        scale_height = np.full(shp, R_specific * temperature / self.grav(altitudes[:, None]))
        density = np.full(shp, self.surface_pressure / (self.grav(0) * scale_height))  # Placeholder

        cp = self.material['isobaric_mass_heat_capacity']

        for iteration in range(max_iterations):
            prev_temperature = temperature.copy()
            prev_pressure = pressure.copy()

            for layer_idx in range(1, self.n_layers):
                altitude = altitudes[layer_idx]
                g = self.grav(altitude)

                # Calculate adiabatic lapse rate Γ = g / cp
                gamma = g / cp  # K/m

                # Update temperature based on adiabatic lapse rate
                temperature[layer_idx] = temperature[layer_idx - 1] - gamma * delta_h[layer_idx]
                temperature[layer_idx] = np.fmax(temperature[layer_idx], 1e-6)

                # Update scale height based on new temperature
                scale_height[layer_idx] = R_specific * temperature[layer_idx] / g
                is_vac = scale_height[layer_idx] < 1e-15

                # Update pressure based on hydrostatic equilibrium
                pressure[layer_idx][is_vac] = 0.0
                pressure[layer_idx][~is_vac] = pressure[layer_idx-1][~is_vac] * np.exp(
                                                            -delta_h[layer_idx] / scale_height[layer_idx][~is_vac]
                )

                # Update density using the ideal gas law
                density[layer_idx][is_vac] = 0.0
                density[layer_idx][~is_vac] = pressure[layer_idx][~is_vac] / (R_specific * temperature[layer_idx][~is_vac])

                if np.any(pressure[layer_idx]<1e-15) or np.any(density[layer_idx]<1e-15):
                # Upper limit of the atmosphere found
                    top_of_atmosphere = altitudes[layer_idx-1]
                    altitudes = np.linspace(
                        np.amin(self.surface.elevation),
                        top_of_atmosphere,
                        num=self.n_layers
                    )
                    return self.initialize_layers(altitudes)

            # Check for convergence based on temperature and pressure
            temp_err = np.linalg.norm(temperature - prev_temperature)
            press_err = np.linalg.norm(pressure - prev_pressure)
            err = temp_err + press_err

            if err < tol:
                break
        else:
            raise ValueError("Atmosphere initialization did not converge within the maximum number of iterations.")

        return altitudes, pressure, density, temperature


    def grav(self, h) -> np.ndarray:
        """
        Calculate gravity at altitude h.
        :param h: Altitude in meters; distance from center of planet minus radius. Can be scalar or array.
        :return: Gravity in m/s²
        """
        return constants.G * self.planet_mass / (h + self.surface.radius) ** 2
