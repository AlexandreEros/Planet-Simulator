import noise
import numpy as np
from scipy import constants

from .materials import Materials
from src.math_utils.geodesic_grid import GeodesicGrid
from src.math_utils.vector_utils import cartesian_to_spherical, normalize

class Surface(GeodesicGrid):
    def __init__(self, **kwargs):
        self.resolution = 0 if 'resolution' not in kwargs else int(kwargs['resolution'])
        self.radius = 1.0 if 'radius' not in kwargs else float(kwargs['radius'])
        super().__init__(self.resolution, self.radius)

        self.noise_scale = 1.0 if 'noise_scale' not in kwargs else float(kwargs['noise_scale'])
        self.noise_octaves = 4 if 'noise_octaves' not in kwargs else int(kwargs['noise_octaves'])
        self.noise_amplitude = 0.05 if 'noise_amplitude' not in kwargs else float(kwargs['noise_amplitude'])
        self.noise_bias = 0.0 if 'noise_bias' not in kwargs else float(kwargs['noise_bias'])
        self.noise_offset = (0.0, 0.0, 0.0) if 'noise_offset' not in kwargs else \
            tuple([float(n.strip(' ')) for n in kwargs['noise_offset'][1:-1].split(',')])

        self.relative_distance = self.elevate_terrain()
        self.vertices *= self.relative_distance[:,None]
        self.elevation = (self.relative_distance - 1.0) * self.radius

        self.coordinates = np.empty_like(self.vertices)
        self.coordinates[:,:2] = np.apply_along_axis(cartesian_to_spherical, -1, self.vertices)
        self.coordinates[:,2] = self.elevation

        self.normals = self.calculate_normals()


        self.irradiance = np.zeros(shape=len(self.vertices), dtype=np.float64)
        self.emissivity = 0.95

        self.material = Materials.load(kwargs['material_name'])
        self.albedo = self.material['albedo']
        self.thermal_conductivity = self.material['thermal_conductivity']
        self.density = self.material['density']
        self.specific_heat_capacity = self.material['specific_heat_capacity']

        self.n_layers = 10 if 'n_layers' not in kwargs else kwargs['n_layers']
        self.max_depth = 4.0 if 'max_depth' not in kwargs else kwargs['max_depth']
        self.layer_depths = self.max_depth * (np.logspace(0, 1, self.n_layers, base=2) - 1)
        self.vertex_area = 4 * np.pi * self.radius ** 2 / len(self.vertices)
        self.subsurface_temperature = np.full((len(self.vertices), self.n_layers), kwargs['blackbody_temperature'], dtype=np.float64)

        self.f_GH = 0.0  # Greenhouse factor; will be updated by `Planet` if the planet has an atmosphere.


    def elevate_terrain(self):
        # Generate elevation using Perlin noise for each vertex
        try:
            elevations = []
            for vertex in self.vertices / self.radius:
                elevation = noise.pnoise3(vertex[0] * self.noise_scale + self.noise_offset[0],
                                          vertex[1] * self.noise_scale + self.noise_offset[1],
                                          vertex[2] * self.noise_scale + self.noise_offset[2],
                                          octaves=self.noise_octaves)
                elevations.append(elevation)
            elevations -= (np.amin(elevations) + np.amax(elevations)) / 2
            elevations *= 0.5 / np.amax(elevations)
            relative_distances = 1 + self.noise_amplitude * (np.array(elevations) + self.noise_bias/2)
            return relative_distances

        except Exception as err:
            raise Exception(f"Error calculating surface elevations:\n{err}")


    def calculate_normals(self):
        try:
            # Calculate normal vectors for each vertex
            normals = np.zeros(shape=self.vertices.shape, dtype=np.float64)

            for face in self.faces:
                v1, v2, v3 = [self.vertices[vert_idx] for vert_idx in face]

                face_normal = np.cross(v2 - v1, v3 - v1)
                normal_magnitude = np.linalg.norm(face_normal)
                if normal_magnitude == 0:
                    raise ZeroDivisionError(f"Division by zero identified while normalizing normal vectors; "
                                            f"the face formed by vertices {face[0]}, {face[1]}, and {face[2]} (located "
                                            f"at {v1:.2f}, {v2:.2f}, and {v3:.2f}) has a null normal vector.")
                face_normal /= normal_magnitude  # Normalize the vector

                for vert_idx in face:
                    # The normal vector at a vertex is the normalized sum of the normals of neighboring faces
                    normals[vert_idx] += face_normal

            normal_magnitudes = np.linalg.norm(normals, axis=-1)
            normals /= normal_magnitudes[..., None]

            is_water = self.elevation<=0.0
            normals[is_water] = normalize(self.vertices[is_water])

            return normals

        except Exception as err:
            raise Exception(f"Error calculating normal vectors at vertices on the surface:\n{err}")


    def update_irradiance(self, sunlight: np.ndarray):
        self.irradiance = -np.einsum('j, ij -> i', sunlight, self.normals)
        self.irradiance = np.fmax(self.irradiance, 0.0)

    def surface_heat_flux(self):
        # W/m²
        Q_absorbed = self.irradiance * (1 - self.albedo)
        Q_emitted = self.emissivity * constants.Stefan_Boltzmann * self.temperature ** 4
        return Q_absorbed - Q_emitted * (1 - self.f_GH)

    def update_temperature(self, delta_t: float):
        k = self.thermal_conductivity  # (W/m·K)
        rho = self.density  # (kg/m³)
        c = self.specific_heat_capacity  # (J/kg·K)
        alpha = k / (rho * c)  # Thermal diffusivity (m²/s)

        dz = np.diff(self.layer_depths, prepend=0)  # Layer thicknesses

        # Top layer (surface interaction)
        self.subsurface_temperature[:, 0] += delta_t * self.surface_heat_flux() / (rho * c * dz[1] )#* self.vertex_area)
        heat_loss_to_subsurface = alpha * (self.subsurface_temperature[:,0]-self.subsurface_temperature[:,1]) / dz[1]**2
        self.subsurface_temperature[:, 0] -= delta_t * heat_loss_to_subsurface

        # Middle layers
        self.subsurface_temperature[:, 1:-1] += delta_t * alpha * (
                  (self.subsurface_temperature[:, :-2]
                  -self.subsurface_temperature[:, 1:-1]) / dz[1:-1] ** 2
                + (self.subsurface_temperature[:, 2:]
                  -self.subsurface_temperature[:, 1:-1]) / dz[2:] ** 2
        )

        # Bottom layer
        Q_geothermal = 0.02  # W/m²
        heat_flux_from_below = Q_geothermal / (rho * c * dz[-1])
        self.subsurface_temperature[:, -1] += heat_flux_from_below
        heat_loss_to_above = alpha * (self.subsurface_temperature[:, -2] - self.subsurface_temperature[:, -1]) / dz[-1]**2
        self.subsurface_temperature[:, -1] -= delta_t * heat_loss_to_above


    @property
    def temperature(self):
        return self.subsurface_temperature[:,0]

    # def update_temperature(self, delta_t: float):
    #     self.temperature += delta_t * self.surface_heat_flux() / self.heat_capacity