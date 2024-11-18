import numpy as np
from geodesic_grid import GeodesicGrid
import noise
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

            self.coordinates = np.empty_like(self.vertices)
            self.coordinates[:,:2] = np.apply_along_axis(cartesian_to_spherical, -1, self.vertices)
            self.coordinates[:,2] = self.elevation

            self.normals = self.calculate_normals()

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
