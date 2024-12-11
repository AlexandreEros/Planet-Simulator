import numpy as np
from scipy import constants, sparse
import json

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
        # self.altitudes = np.logspace(0, np.log10(self.surface.radius/64), num=self.n_layers) - 1.0 + np.amin(self.surface.elevation)
        temp_top_of_atmosphere = self.surface.radius / 64
        temp_altitudes = np.linspace(
            np.amin(self.surface.elevation),
            np.amin(self.surface.elevation) + temp_top_of_atmosphere,
            num=self.n_layers
        )

        self.surface_pressure = 0.0 if 'surface_pressure' not in kwargs else kwargs['surface_pressure']
        self.lapse_rate = 0.0 if 'lapse_rate' not in kwargs else kwargs['lapse_rate']
        self.material = self.load_material(kwargs['material_name'])
        self.molar_mass = 0.02896 if 'molar_mass' not in self.material else self.material['molar_mass']
        self.k_GH = 1e-5 if 'greenhouse_efficiency' not in kwargs else kwargs['greenhouse_efficiency']
        self.chi_CO2 = 1.0 if 'chi_CO2' not in kwargs else kwargs['chi_CO2']  # CO2 dominance
        self.f_GH = self.k_GH * self.surface_pressure * (self.chi_CO2 / self.molar_mass)  # Greenhouse factor

        # Layer structure: axis 0 = layer, axis 1 = vertices, axis 2 = vector components (for position and velocity)
        self.altitudes, self.pressure, self.density, self.temperature = self.initialize_atmosphere(temp_altitudes)
        self.top_of_atmosphere = self.altitudes[-1]
        normalized_vertices = normalize(self.surface.vertices)
        radii = self.altitudes + self.surface.radius
        self.vertices = np.stack([normalized_vertices * radius for radius in radii], axis=0)

        self.is_underground = self.altitudes[:,None] <= self.surface.elevation
        self.lowest_layer_above_surface = np.argmin(self.is_underground, axis=0)

        self.layered_adjacency_matrix = self.build_layered_adjacency_matrix(self.surface.adjacency_matrix)
        self.laplacian_matrix = self.build_laplacian_matrix(self.layered_adjacency_matrix)


    @staticmethod
    def load_material(material_name) -> dict:
        with open('materials.json', 'r') as f:
            materials = json.load(f)['materials']
            material = next((m for m in materials if m['name'] == material_name), None)
            if not material:
                raise ValueError(f"Material '{material_name}' not found in library.")
        return material


    def initialize_atmosphere(self, altitudes, max_iterations=100, tol=1e-6) \
            -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        R_specific = constants.R / self.molar_mass  # Specific gas constant (J/kg·K)
        delta_h = np.diff(altitudes, prepend=0)  # Height differences between layers

        # Initialize arrays
        shp = (self.n_layers, self.surface.n_vertices)
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
                    return self.initialize_atmosphere(altitudes)

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


    def exchange_heat_with_surface(self, delta_t: float):
        """
        Exchange heat between the surface and the lowest atmospheric layer.

        :param delta_t: Time step in seconds.
        """
        # Constants and parameters
        k_surface_atmosphere = self.material['convective_heat_transfer_coefficient']  # W/m²·K
        specific_heat_air = self.material['isobaric_mass_heat_capacity']  # J/kg·K

        surface_temperature = self.surface.temperature

        layer_indices = self.lowest_layer_above_surface
        vertex_indices = np.arange(self.surface.n_vertices)

        # Lowest atmospheric layer properties
        atmospheric_temperature = self.temperature[layer_indices, vertex_indices]
        atmospheric_density = self.density[layer_indices, vertex_indices]
        atmospheric_layer_thickness = self.altitudes[layer_indices] - self.altitudes[layer_indices - 1]

        # Heat flux between surface and atmosphere
        heat_flux = k_surface_atmosphere * (surface_temperature - atmospheric_temperature)

        # Update surface temperature
        surface_heat_loss = heat_flux  # Total energy lost by the surface
        self.surface.subsurface_temperature[:, 0] -= (
            surface_heat_loss / (self.surface.density * self.surface.specific_heat_capacity)
        ) * delta_t

        # Update atmospheric temperature
        is_vac = atmospheric_density < 1e-9
        self.temperature[layer_indices, vertex_indices][~is_vac] += heat_flux[~is_vac] / (
            specific_heat_air * (atmospheric_density * atmospheric_layer_thickness)[~is_vac]
        ) * delta_t

        # Ensure temperatures remain physical (e.g., above 0 K)
        self.temperature[~np.isnan(self.temperature)] = np.fmax(self.temperature[~np.isnan(self.temperature)], 0.0)
        self.surface.subsurface_temperature[:, 0] = np.fmax(self.surface.subsurface_temperature[:, 0], 0.0)


    def build_layered_adjacency_matrix(self, horizontal_adjacency_matrix: sparse.coo_matrix):
        """
        Given the horizontal adjacency matrix, build another one that also represents vertical connections.

        :param horizontal_adjacency_matrix: NxN adjacency matrix for the surface where N = surface.n_vertices.
                                            Only off-diagonal entries represent connection weights. The weights are the
                                            inverse Euclidean distances.
                                            Note: horizontal_adjacency_matrix should not contain diagonal entries.
                                            These will be computed separately when constructing the Laplacian.

        :return A: The layered adjacency matrix, which includes not only all the information in
        `horizontal_adjacency_matrix`, but also vertical adjacency connections. It has shape NxN, where
        N = n_layers * surface.n_vertices
        """
        # Horizontal adjacency (block diagonal matrix)
        A_blocks = [horizontal_adjacency_matrix] * self.n_layers
        A_block_diag = sparse.block_diag(A_blocks)

        # Add vertical adjacency
        row_indices = []
        col_indices = []
        data = []

        for layer_idx in range(self.n_layers - 1):
            dz = (self.altitudes[layer_idx + 1] - self.altitudes[layer_idx])
            vertical_weight = 1.0 / dz

            for v_idx in range(self.surface.n_vertices):
                i = layer_idx * self.surface.n_vertices + v_idx
                j = (layer_idx + 1) * self.surface.n_vertices + v_idx

                # Add vertical adjacency (symmetric)
                row_indices.extend([i, j])
                col_indices.extend([j, i])
                data.extend([vertical_weight, vertical_weight])

        V = sparse.coo_matrix((data, (row_indices, col_indices)))

        # Combine horizontal and vertical adjacency
        A = A_block_diag + V
        return A

    @staticmethod
    def build_laplacian_matrix(A: sparse.coo_matrix):
        """
        Given the layered adjacency matrix (which represents vertical adjacency as well), build the corresponding
        Laplacian matrix.

        :param A: NxN adjacency matrix for the entire atmosphere, where N = n_layers * surface.n_vertices.
                  Only off-diagonal entries represent connection weights. The weights are the inverse Euclidean
                  distances
        :return L: The Laplacian matrix
        """
        # Calculate the degree matrix as the sum of each row
        row_sum = np.array(A.sum(axis=1))
        D = sparse.diags(row_sum.ravel(), format='csr')

        # Compute the Laplacian
        L = D - A
        return L

    def conduct_heat(self, delta_t: float):
        """
        Perform a conduction step using the sparse Laplacian matrix:
        T_new = T + Δt * D * L * T
        """
        k_air = self.material['thermal_conductivity']
        cp_air = self.material['isobaric_mass_heat_capacity']

        # D = diag(k/(ρcp)), where ρ varies by cell.
        D = sparse.diags(k_air / (self.density.ravel() * cp_air))

        T_flat = self.temperature.ravel()
        dT = delta_t * D.dot(self.laplacian_matrix.dot(T_flat))
        T_new = np.fmax(T_flat + dT, 0.0)
        self.temperature = T_new.reshape(self.n_layers, self.surface.n_vertices)
