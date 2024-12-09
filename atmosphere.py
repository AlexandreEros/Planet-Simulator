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

        # Layer structure: axis 0 = layer, axis 1 = vertices, axis 2 = vector components (for position and velocity)
        self.altitudes, self.pressure, self.density, self.temperature = self.initialize_atmosphere(temp_altitudes)
        self.top_of_atmosphere = self.altitudes[-1]
        normalized_vertices = normalize(self.surface.vertices)
        radii = self.altitudes + self.surface.radius
        self.vertices = np.stack([normalized_vertices * radius for radius in radii], axis=0)

        self.is_underground = self.altitudes[:,None] <= self.surface.elevation
        self.lowest_layer_above_surface = np.argmin(self.is_underground, axis=0)

        self.neighbors = self.build_neighbors()

        # Sparse Laplacian matrix for heat conduction
        self.L = self.build_laplacian_matrix(self.neighbors, self.n_layers, self.surface.n_vertices)


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

        # Compute specific heat capacity (cp) assuming diatomic gas (CO2)
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
                vacuum = scale_height[layer_idx] < 1e-15

                # Update pressure based on hydrostatic equilibrium
                pressure[layer_idx][vacuum] = 0.0
                pressure[layer_idx][~vacuum] = pressure[layer_idx-1][~vacuum] * np.exp(
                                                            -delta_h[layer_idx] / scale_height[layer_idx][~vacuum]
                )

                # Update density using the ideal gas law
                density[layer_idx][vacuum] = 0.0
                density[layer_idx][~vacuum] = pressure[layer_idx][~vacuum] / (R_specific * temperature[layer_idx][~vacuum])

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
        vacuum = atmospheric_density < 1e-9
        self.temperature[layer_indices, vertex_indices][~vacuum] += heat_flux[~vacuum] / (
            specific_heat_air * (atmospheric_density * atmospheric_layer_thickness)[~vacuum]
        ) * delta_t

        # Ensure temperatures remain physical (e.g., above 0 K)
        self.temperature[~np.isnan(self.temperature)] = np.fmax(self.temperature[~np.isnan(self.temperature)], 0.0)
        self.surface.subsurface_temperature[:, 0] = np.fmax(self.surface.subsurface_temperature[:, 0], 0.0)


    def compute_interface_weight(self, node_i, node_j):
        """
        Compute the interface area Aij between node_i and node_j based on adjacency type.
        :param node_i: Tuple (layer_idx, vertex_idx) for node i
        :param node_j: Tuple (layer_idx, vertex_idx) for node j
        :return: Interface area divided by distance and cell volume Aij / (dij * Vi)
        """
        R = self.surface.radius
        layer_idx = max(node_i[0], node_j[0])
        h_l = self.altitudes[layer_idx]
        if node_i[0] != node_j[0]:
            h_min = self.altitudes[min(node_i[0], node_j[0])]
            A_layer = 4 * np.pi * (R + h_l) ** 2
            A_node = A_layer / self.surface.n_vertices
            cell_volume = 4/3 * np.pi * ((R+h_l)**3 - (R+h_min)**3) / self.surface.n_vertices
            if cell_volume==0:
                raise ZeroDivisionError(f"{h_l=};\n{h_min=}")
            return A_node / ((h_l-h_min) * cell_volume)

        else:
            # Compute the angular distance between adjacent vertices
            theta = 2 * np.pi / (self.surface.n_vertices / 10)  # Example for icosahedral grid
            # Vertical thickness (average between layers)
            if layer_idx < self.n_layers - 1:
                h_upper = self.altitudes[layer_idx + 1] - self.altitudes[layer_idx]
            else:
                h_upper = self.altitudes[layer_idx] - self.altitudes[layer_idx - 1]
            A_ij = h_upper * (R + h_l) * theta
            cell_volume = 4/3 * np.pi * ((R+h_upper)**3 - R**3) / self.surface.n_vertices
            return A_ij / (R*theta * cell_volume)

    def build_laplacian_matrix(self, neighbors: dict, n_layers: int, n_vertices: int):
        """
        Construct a sparse weighted Laplacian matrix L for the entire atmosphere.
        :param neighbors: Dictionary containing neighbor nodes and adjacency types
        :param n_layers: Number of atmospheric layers
        :param n_vertices: Number of vertices per layer
        :return: Sparse CSR Laplacian matrix
        """
        n = n_layers * n_vertices
        row_indices = []
        col_indices = []
        data = []

        # Helper to get the flat index
        flat_idx = lambda l_idx, v_idx: l_idx * n_vertices + v_idx

        for layer_idx in range(n_layers):
            for vertex_idx in range(n_vertices):
                i = flat_idx(layer_idx, vertex_idx)
                current_node = (layer_idx, vertex_idx)
                nbrs = neighbors[current_node]

                degree = 0.0  # To accumulate the weighted degrees
                for neighbor in nbrs:
                    j = flat_idx(neighbor[0], neighbor[1])
                    weight = self.compute_interface_weight(current_node, neighbor)

                    # Off-diagonal entry
                    row_indices.append(i)
                    col_indices.append(j)
                    data.append(weight)

                    # Accumulate degree
                    degree += weight

                # Diagonal entry
                row_indices.append(i)
                col_indices.append(i)
                data.append(-degree)

        # Create the sparse Laplacian matrix
        L = sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        return L

    def conduct_heat(self, delta_t: float):
        """
        Perform a conduction step using the sparse Laplacian matrix.
        T_new = T + delta_t * D * L * T
        :param delta_t: Time step in seconds.
        """
        k_air = self.material['thermal_conductivity']
        cp_air = self.material['isobaric_mass_heat_capacity']

        D = sparse.diags_array(k_air / (self.density.flatten() * cp_air))

        # Apply the Laplacian
        T_flat = self.temperature.ravel()
        dT = delta_t * D.dot(self.L.dot(T_flat))

        # Update temperature and ensure non-negative
        T_new = np.fmax(T_flat + dT, 0.0)
        self.temperature = T_new.reshape(self.n_layers, self.surface.n_vertices)