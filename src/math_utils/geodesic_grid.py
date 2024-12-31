import cupy as cp
from cupyx.scipy import sparse
import numpy as np
from scipy import sparse

from .vector_utils import normalize, cartesian_to_spherical, rotation_mat_x, rotation_mat_y


class GeodesicGrid:
# Create a basic geodesic grid (icosahedron-based)

    # Create the vertices of an icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices: list[list[float]] = (np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ]) / np.hypot(1, phi)       ).tolist()

    # Define the icosahedron faces (triangles)
    ico_faces: list[list[int]] = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]


    def __init__(self, resolution: int = 0, radius: float = 1.0):
        try:
            self.resolution = resolution
            self.radius = radius

            self.mesh = self.geodesic_subdivide()
            self.vertices = cp.array(self.radius * self.mesh[0])
            self.faces = cp.array(self.mesh[1])

            self.n_vertices = len(self.vertices)
            self.n_faces = len(self.faces)

            coordinates = cp.apply_along_axis(cartesian_to_spherical, -1, self.vertices)
            self.longitude = coordinates[:, 0]
            self.latitude = coordinates[:, 1]

            self.adjacency_matrix = self.build_adjacency_matrix()
            self.dx, self.dy, self.dz = self.build_dxdydz_matrices()

        except Exception as err:
            raise Exception(f"Error in the constructor of `GeodesicGrid`:\n{err}")


    def geodesic_subdivide(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Subdivide each triangle in the faces list to increase the resolution.
        """
        vertices = [np.array(vertex) for vertex in self.ico_vertices]
        faces = self.ico_faces.copy()

        for _ in range(self.resolution):
            try:
                new_faces = []

                # Subdivide each triangle into four smaller triangles
                for tri in faces:
                    try:
                        v1, v2, v3 = tri
                        mid_1_2 = normalize((vertices[v1] + vertices[v2]) / 2)  # self.add_and_return_midpoint(v1, v2, edge2midpoint)
                        mid_2_3 = normalize((vertices[v2] + vertices[v3]) / 2)  # self.add_and_return_midpoint(v2, v3, edge2midpoint)
                        mid_3_1 = normalize((vertices[v3] + vertices[v1]) / 2)  # self.add_and_return_midpoint(v3, v1, edge2midpoint)
                        vertices.extend([mid_1_2, mid_2_3, mid_3_1])

                        nv = len(vertices)
                        vm12, vm23, vm31 = nv-3, nv-2, nv-1

                        # Create four new triangles
                        new_faces.extend([
                            [v1, vm12, vm31],
                            [v2, vm23, vm12],
                            [v3, vm31, vm23],
                            [vm12, vm23, vm31]
                        ])
                    except ValueError as err:
                        raise ValueError(f"Error subdividing triangle {tri}: {err}")

                # Replace old faces with new ones
                faces = new_faces

            except Exception as e:
                print(f"An error occurred during geodesic subdivision at depth {_}: {e}")
                raise

        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int32)

        vertices = rotation_mat_x(1e-1).dot(vertices.T).T  # Prevent points from being exactly at the poles
        vertices = rotation_mat_y(1e-1).dot(vertices.T).T
        return vertices, faces


    def build_adjacency_matrix(self) -> sparse.coo_matrix:
        """
        Build an adjacency matrix for the geodesic grid using the inverse of the distance between vertices as weights.
        """

        row_indices = []
        col_indices = []
        data = []

        for face in self.faces:
            # Each face is a tuple of three vertex indices
            v1, v2, v3 = face

            for (a, b) in [(v1, v2), (v2, v3), (v3, v1)]:
                dist = cp.linalg.norm(self.vertices[a] - self.vertices[b])
                weight = 1.0 / dist if dist > 0 else 0

                row_indices.extend([a, b])
                col_indices.extend([b, a])
                data.extend([weight, weight])

        return sparse.coo_matrix((data, (row_indices, col_indices)), shape=(self.n_vertices, self.n_vertices))


    def build_dxdydz_matrices(self) -> tuple[sparse.coo_matrix, sparse.coo_matrix, sparse.coo_matrix]:
        """
        Build an adjacency matrix for the geodesic grid using the inverse of the distance between vertices as weights.
        """

        row_indices = []
        col_indices = []
        datax = []
        datay = []
        dataz = []

        for face in self.faces:
            # Each face is a tuple of three vertex indices
            v1, v2, v3 = face

            for (a, b) in [(v1, v2), (v2, v3), (v3, v1)]:
                dx = self.vertices[b, 0] - self.vertices[a, 0]
                dy = self.vertices[b, 1] - self.vertices[a, 1]
                dz = self.vertices[b, 2] - self.vertices[a, 2]

                row_indices.extend([a, b])
                col_indices.extend([b, a])
                datax.extend([dx, -dx])
                datay.extend([dy, -dy])
                dataz.extend([dz, -dz])

        dx_mat = sparse.coo_matrix((datax, (row_indices, col_indices)), shape=(self.n_vertices, self.n_vertices))
        dy_mat = sparse.coo_matrix((datay, (row_indices, col_indices)), shape=(self.n_vertices, self.n_vertices))
        dz_mat = sparse.coo_matrix((dataz, (row_indices, col_indices)), shape=(self.n_vertices, self.n_vertices))
        return dx_mat, dy_mat, dz_mat
