import numpy as np
from geodesic_grid import GeodesicGrid
import noise
from vector_utils import cartesian_to_spherical

class Surface(GeodesicGrid):
    def __init__(self, radius: float, resolution: int = 0, noise_scale: float = 1.0, noise_octaves: int = 4, noise_amplitude: float = 0.05):
        try:
            super().__init__(resolution)
            self.radius = radius
            self.noise_scale = noise_scale
            self.noise_octaves = noise_octaves
            self.noise_amplitude = noise_amplitude

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
                elevation = noise.pnoise3(vertex[0] * self.noise_scale,
                                          vertex[1] * self.noise_scale,
                                          vertex[2] * self.noise_scale,
                                          octaves=self.noise_octaves)
                elevations.append(elevation)
            relative_distances = 1 + self.noise_amplitude * np.array(elevations)
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

            return normals

        except Exception as err:
            raise Exception(f"Error calculating normal vectors at vertices on the surface:\n{err}")
