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
        self.altitudes = np.logspace(0, np.log10(self.surface.radius/100), num=self.n_layers)-1 + np.amin(self.surface.elevation)

        normalized_vertices = normalize(self.surface.vertices)
        radii = self.altitudes + self.surface.radius

        # Layer structure: axis 0 = layer, axis 1 = vertices, axis 2 = vector components (for position and velocity)
        self.vertices = np.stack([normalized_vertices * radius for radius in radii], axis=0)
        self.velocity = np.zeros(self.vertices.shape)         # (m/s)

        self.is_underground = self.altitudes[:,None] <= self.surface.elevation
        self.touches_ground = ~self.is_underground & np.roll(self.is_underground, 1, axis=0)

        self.surface_pressure = 0.0 if 'surface_pressure' not in kwargs else kwargs['surface_pressure']
        self.lapse_rate = 0.0 if 'lapse_rate' not in kwargs else kwargs['lapse_rate']
        self.material = self.load_material(kwargs['material_name'])
        self.molar_mass = 0.02896 if 'molar_mass' not in self.material else self.material['molar_mass']

        self.pressure, self.density, self.temperature = self.initialize_atmosphere()

        self.neighbors = self.build_neighbors()

        self.view()


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


    def view(self):
        plt.plot(self.temperature[:,0]-273.15, self.altitudes / 1000)
        plt.title("Temperature vs Altitude")
        plt.xlabel("Temperature (ºC)")
        plt.ylabel("Altitude (km)")
        plt.tight_layout()
        plt.show()

        plt.plot(self.pressure[:,0], self.altitudes / 1000)
        plt.title("Pressure vs Altitude")
        plt.xlabel("Pressure (Pa)")
        plt.ylabel("Altitude (km)")
        plt.tight_layout()
        plt.show()

        plt.plot(self.density[:,0], self.altitudes / 1000)
        plt.title("Density vs Altitude")
        plt.xlabel("Density (kg/m³)")
        plt.ylabel("Altitude (km)")
        plt.tight_layout()
        plt.show()