import numpy as np
from scipy import constants
import json

import matplotlib.pyplot as plt

from vector_utils import normalize
from surface import  Surface

class Atmosphere:
    def __init__(self, surface: Surface, planet_mass: float, **kwargs):
        """
        Initialize the atmospheric model.
        :param surface: The surface object (provides vertices and neighbors).
        :param planet_mass: Mass of the planet (used to calculate gravity).
        """

        self.surface = surface
        self.planet_mass = planet_mass

        self.n_layers = 16 if 'n_layers' not in kwargs else kwargs['n_layers']
        self.altitudes = np.logspace(0, np.log10(self.surface.radius/64), num=self.n_layers) - 1.0 + np.amin(self.surface.elevation)

        normalized_vertices = normalize(self.surface.vertices)
        radii = self.altitudes + self.surface.radius

        # Layer structure: axis 0 = layer, axis 1 = vertices, axis 2 = vector components (for position and velocity)
        self.vertices = np.stack([normalized_vertices * radius for radius in radii], axis=0)
        self.velocity = np.zeros(self.vertices.shape)         # (m/s)

        self.surface_pressure = 0.0 if 'surface_pressure' not in kwargs else kwargs['surface_pressure']
        self.lapse_rate = 0.0 if 'lapse_rate' not in kwargs else kwargs['lapse_rate']
        self.material = self.load_material(kwargs['material_name'])
        self.molar_mass = 0.02896 if 'molar_mass' not in self.material else self.material['molar_mass']

        self.pressure, self.density, self.temperature = self.initialize_atmosphere()

        self.is_underground = self.altitudes[:,None] <= self.surface.elevation
        # self.pressure[self.is_underground] = np.nan
        # self.density[self.is_underground] = np.nan
        # self.temperature[self.is_underground] = np.nan
        self.touches_ground = ~self.is_underground & np.roll(self.is_underground, 1, axis=0)
        self.layer_where_touches_ground = np.argmax(self.touches_ground, axis=0)

        self.neighbors = self.build_neighbors()


    @staticmethod
    def load_material(material_name) -> dict:
        with open('materials.json', 'r') as f:
            materials = json.load(f)['materials']
            material = next((m for m in materials if m['name'] == material_name), None)
            if not material:
                raise ValueError(f"Material '{material_name}' not found in library.")
        return material


    def initialize_atmosphere(self, max_iterations=100, tol=1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        R_specific = constants.R / self.molar_mass  # Specific gas constant (J/kg·K)
        delta_h = np.diff(self.altitudes, prepend=0)  # Height differences between layers

        # Initialize arrays
        shp = (self.n_layers, self.surface.n_vertices)
        pressure = np.full(shp, self.surface_pressure)  # Start with surface pressure
        temperature = np.full(shp, self.surface.temperature)  # Initialize temperature
        scale_height = np.full(shp, R_specific * temperature / self.grav(self.altitudes[:, None]))
        density = np.full(shp, self.surface_pressure / (self.grav(0) * scale_height))  # Placeholder

        # Compute specific heat capacity (cp) assuming diatomic gas (CO2)
        cp = self.material['isobaric_mass_heat_capacity']

        for iteration in range(max_iterations):
            prev_temperature = temperature.copy()
            prev_pressure = pressure.copy()

            for layer_idx in range(1, self.n_layers):
                altitude = self.altitudes[layer_idx]
                g = self.grav(altitude)

                # Calculate adiabatic lapse rate Γ = g / cp
                gamma = g / cp  # K/m

                # Update temperature based on adiabatic lapse rate
                temperature[layer_idx] = temperature[layer_idx - 1] - gamma * delta_h[layer_idx]

                # Ensure temperature doesn't drop below a minimum threshold (e.g., 0 K)
                temperature[layer_idx] = np.fmax(temperature[layer_idx], 0.0)

                # Update scale height based on new temperature
                scale_height[layer_idx] = R_specific * temperature[layer_idx] / g

                # Update pressure based on hydrostatic equilibrium
                pressure[layer_idx] = pressure[layer_idx - 1] * np.exp(-delta_h[layer_idx] / scale_height[layer_idx])

                # Update density using the ideal gas law
                density[layer_idx] = pressure[layer_idx] / (R_specific * temperature[layer_idx])

            # Check for convergence based on temperature and pressure
            temp_err = np.linalg.norm(temperature - prev_temperature)
            press_err = np.linalg.norm(pressure - prev_pressure)
            err = temp_err + press_err

            if err < tol:
                break
        else:
            raise ValueError("Atmosphere initialization did not converge within the maximum number of iterations.")

        return pressure, density, temperature

    def grav(self, h) -> np.ndarray:
        """
        Calculate gravity at altitude h.
        :param h: Altitude in meters; distance from center of planet minus radius. Can be scalar or array.
        :return: Gravity in m/s²
        """
        return constants.G * self.planet_mass / (h + self.surface.radius) ** 2


    def build_neighbors(self) -> dict[tuple[int, int], set[tuple[int, int]]]:
        """
        Extend the surface neighbors dictionary to include vertical neighbors.
        """
        neighbors = {}

        for layer_idx in range(self.n_layers):
            for vertex_idx in range(self.surface.n_vertices):
                # Copy horizontal neighbors from the surface
                neighbors[(layer_idx, vertex_idx)] = set([(layer_idx, neighbor) for neighbor in self.surface.neighbors[vertex_idx]])

                # Add vertical neighbors (above and below)
                if layer_idx > 0:
                    neighbors[(layer_idx, vertex_idx)].add((layer_idx - 1, vertex_idx))
                if layer_idx < self.n_layers - 1:
                    neighbors[(layer_idx, vertex_idx)].add((layer_idx + 1, vertex_idx))

        return neighbors


    def exchange_heat_with_surface(self, delta_t: float):
        """
        Exchange heat between the surface and the lowest atmospheric layer.
        :param delta_t: Time step in seconds.
        """
        # Constants and parameters
        k_surface_atmosphere = 10.0  # W/m²·K, approximate heat transfer coefficient
        specific_heat_air = self.material['isobaric_mass_heat_capacity']  # J/kg·K for air at constant pressure

        # Surface properties
        surface_temperature = self.surface.temperature
        # surface_area = self.surface.vertex_area

        layer_indices = self.layer_where_touches_ground
        vertex_indices = np.arange(self.surface.n_vertices)

        # Lowest atmospheric layer properties
        atmospheric_temperature = self.temperature[layer_indices, vertex_indices]
        atmospheric_density = self.density[layer_indices, vertex_indices]
        # atmospheric_temperature = self.temperature[self.touches_ground]
        # atmospheric_density = self.density[self.touches_ground]
        atmospheric_layer_thickness = self.altitudes[layer_indices] - self.altitudes[layer_indices - 1]
        # atmospheric_layer_thickness = self.altitudes[self.layer_where_touches_ground] - self.altitudes[self.layer_where_touches_ground-1]

        # Heat flux between surface and atmosphere
        heat_flux = k_surface_atmosphere * (surface_temperature - atmospheric_temperature)

        # Update surface temperature
        surface_heat_loss = heat_flux  # Total energy lost by the surface
        self.surface.subsurface_temperature[:, 0] -= (
            surface_heat_loss / (self.surface.density * self.surface.specific_heat_capacity)
        ) * delta_t

        # Update atmospheric temperature
        vacuum = atmospheric_density == 0
        self.temperature[layer_indices, vertex_indices][~vacuum] += heat_flux[~vacuum] / (
            specific_heat_air * (atmospheric_density * atmospheric_layer_thickness)[~vacuum]
        ) * delta_t

        # Ensure temperatures remain physical (e.g., above 0 K)
        self.temperature[~np.isnan(self.temperature)] = np.fmax(self.temperature[~np.isnan(self.temperature)], 0.0)
        self.surface.subsurface_temperature[:, 0] = np.fmax(self.surface.subsurface_temperature[:, 0], 0.0)
