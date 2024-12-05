import numpy as np
from vector_utils import normalize

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
            self.vertices = self.radius * self.mesh[0]
            self.faces = self.mesh[1]

            self.neighbors: dict[int, set[int]] = self.build_neighbors()

        except Exception as err:
            raise Exception(f"Error in the constructor of `GeodesicGrid`:\n{err}")



    def add_and_return_midpoint(self, v1: int, v2: int, edge2midpoint: dict[tuple[int,int], int]) -> np.ndarray:
        edge: tuple[int, int] = (min(v1,v2), max(v1,v2))
        if edge not in edge2midpoint:
            try:
                midpoint = (np.array(self.vertices[v1]) + np.array(self.vertices[v2])) / 2.0
                norm = np.linalg.norm(midpoint)
                midpoint = midpoint / norm  # Normalize to keep on sphere
                return midpoint
            except ZeroDivisionError as err:
                raise ZeroDivisionError(f"Midpoint between vertices {v1} and {v2} has norm 0: {err}")
            except IndexError as err:
                wrong_idx = edge[1]
                if edge[0] >= len(self.vertices): wrong_idx = edge
                raise IndexError(f"Vertex index(ices) {wrong_idx} out of bounds: {err}")
            except Exception as err:
                raise ValueError(f"Error calculating midpoint for vertices {v1} and {v2}: {err}")

        # return edge2midpoint[edge]


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
        return vertices, faces



    def build_neighbors(self) -> dict[int, set[int]]:
        """
        Build an adjacency list where each vertex maps to its neighboring vertices.
        """
        neighbors = {i: set() for i in range(len(self.vertices))}
        faces: list[list[int]] = self.faces.tolist()
        for tri in faces:
            v1, v2, v3 = tri
            neighbors[v1].update([v2, v3])
            neighbors[v2].update([v1, v3])
            neighbors[v3].update([v1, v2])
        return neighbors
