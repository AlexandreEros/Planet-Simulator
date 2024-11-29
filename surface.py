import numpy as np
import noise
from scipy import constants
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from geodesic_grid import GeodesicGrid
from vector_utils import cartesian_to_spherical, normalize

class Surface(GeodesicGrid):
    def __init__(self, radius: float, **kwargs): #resolution: int = 0, noise_scale: float = 1.0, noise_octaves: int = 4,
                 #noise_amplitude: float = 0.05, noise_bias: float = 0.0, noise_offset: list[float] = (0.0, 0.0, 0.0)):
        try:
            self.radius = radius
            self.resolution = 0 if 'resolution' not in kwargs else int(kwargs['resolution'])
            self.noise_scale = 1.0 if 'noise_scale' not in kwargs else float(kwargs['noise_scale'])
            self.noise_octaves = 4 if 'noise_octaves' not in kwargs else int(kwargs['noise_octaves'])
            self.noise_amplitude = 0.05 if 'noise_amplitude' not in kwargs else float(kwargs['noise_amplitude'])
            self.noise_bias = 0.0 if 'noise_bias' not in kwargs else float(kwargs['noise_bias'])
            self.noise_offset = (0.0, 0.0, 0.0) if 'noise_offset' not in kwargs else \
                tuple([float(n.strip(' ')) for n in kwargs['noise_offset'][1:-1].split(',')])
            super().__init__(self.resolution)

            self.distance = self.elevate_terrain()
            self.vertices *= self.distance[:,None]
            self.elevation = self.distance - self.radius

            self.cmap = self.get_cmap(self.elevation)
            normalized_elevation = (self.elevation - np.amin(self.elevation)) / (np.amax(self.elevation) - np.amin(self.elevation))
            self.color = self.cmap(normalized_elevation)
            self.surface_type = np.apply_along_axis(self.get_surface_type, -1, self.color)

            self.coordinates = np.empty_like(self.vertices)
            self.coordinates[:,:2] = np.apply_along_axis(cartesian_to_spherical, -1, self.vertices)
            self.coordinates[:,2] = self.elevation

            self.normals = self.calculate_normals()

            self.irradiance = np.zeros(shape=len(self.vertices), dtype=np.float64)

            self.Stefan_Boltzmann = constants.Stefan_Boltzmann
            self.temperature = 180.0 + 120.0 * np.cos(np.arcsin(self.vertices[:,2] / np.linalg.norm(self.vertices, axis=-1)))
            self.emissivity = 0.95

            if 'albedo' in kwargs: self.albedo = kwargs['albedo']
            else:
                # OCEAN, DESERT, VEGETATION, SNOW, LAND
                albedoes = np.array([0.08, 0.8, 0.2, 0.35, 0.25])
                self.albedo = albedoes[self.surface_type]  # 0.1 + 0.8 * ((2*self.color[:,0] + 1.5*self.color[:,1] + self.color[:,2])  / 4.5) ** 3

            if 'heat_capacity' in kwargs: self.heat_capacity = kwargs['heat_capacity']
            else:
                heat_capacities = np.array([4.18e6, 2.0e5, 2.5e6, 1.0e6, 1.5e6])
                self.heat_capacity = heat_capacities[self.surface_type]  # 1e5 + 1e6 * ((1-self.color[:,0]) + (1-self.color[:,1])) ** 2

        except Exception as err:
            raise Exception(f"Error in the constructor of `Surface`:\n{err}")



    def elevate_terrain(self):
        # Generate elevation using Perlin noise for each vertex
        try:
            elevations = []
            for vertex in self.vertices:
                elevation = noise.pnoise3(vertex[0] * self.noise_scale + self.noise_offset[0],
                                          vertex[1] * self.noise_scale + self.noise_offset[1],
                                          vertex[2] * self.noise_scale + self.noise_offset[2],
                                          octaves=self.noise_octaves)
                elevations.append(elevation)
            elevations -= (np.amin(elevations) + np.amax(elevations)) / 2
            elevations *= 0.5 / np.amax(elevations)
            relative_distances = 1 + self.noise_amplitude * (np.array(elevations) + self.noise_bias/2)
            distances = self.radius * relative_distances
            return distances

        except Exception as err:
            raise Exception(f"Error calculating surface elevations:\n{err}")


    @staticmethod
    def get_cmap(elevation):
        underwater_fraction = (0.0 - np.amin(elevation)) / (np.amax(elevation) - np.amin(elevation))
        if underwater_fraction > 0.0:
            sea_num = int(256 * underwater_fraction)
            colors_undersea = plt.cm.Blues_r(np.linspace(start=0, stop=0.25, num=sea_num))
            colors_land = plt.cm.terrain(np.linspace(start=0.25, stop=1, num=256-sea_num))
            all_colors = np.vstack((colors_undersea, colors_land))
        else:
            all_colors = plt.cm.terrain(np.linspace(0.25, 1, 256))
        return mcolors.LinearSegmentedColormap.from_list('world_cmap', all_colors)

    @staticmethod
    def get_surface_type(rgba) -> int:
        OCEAN = 0
        DESERT = 1
        VEGETATION = 2
        SNOW = 3
        LAND = 4

        r, g, b = rgba[0], rgba[1], rgba[2]
        if b > r and b > g:
            return OCEAN  # Blue-ish colors represent oceans
        elif r > g and r > b:
            return DESERT  # Brownish colors represent desert/high elevation
        elif g > r and g > b:
            return VEGETATION  # Green-ish colors represent vegetation
        elif r > 0.8 and g > 0.8 and b > 0.8:
            return SNOW  # White colors represent snow or ice
        else:
            return LAND  # Default to land for other color values


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


    def update_temperature(self, delta_t: float):
        Q_absorbed = self.irradiance * (1 - self.albedo)
        Q_emitted = self.emissivity * self.Stefan_Boltzmann * self.temperature ** 4
        Q_net = Q_absorbed - Q_emitted

        self.temperature += delta_t * Q_net / self.heat_capacity