import numpy as np
from numpy.typing import NDArray

class GeodesicGrid:
    # Create a basic geodesic grid (icosahedron-based)
    def __init__(self, resolution: int = 0):
        icosahedron_mesh = self.build_icosahedron()
        ico_vertices: list[list[float]] = icosahedron_mesh[0]
        ico_faces: list[list[int]] = icosahedron_mesh[1]

        geo_mesh = self.geodesic_subdivide(ico_vertices, ico_faces, resolution)
        self.vertices = np.array(geo_mesh[0], dtype=float)
        self.faces = np.array(geo_mesh[1], dtype=int)

        self.neighbors: dict[int, set[int]] = self.build_neighbors()


    @staticmethod
    def build_icosahedron() -> tuple[list[list[float]], list[list[int]]]:
        # Create the vertices of an icosahedron
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]) / np.hypot(1, phi)

        # Define the icosahedron faces (triangles)
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        return vertices.tolist(), faces

    @staticmethod
    def geodesic_subdivide(vertices: list[list[float]], faces: list[list[int]], depth: int = 0)\
            -> tuple[list[list[float]], list[list[int]]]:
        """
        Subdivide each triangle in the faces list to increase the resolution.
        """
        def add_and_return_midpoint(v1: int, v2: int, vertices: list[list[float]],
                                    edge2midpoint: dict[tuple[int,int], int]) -> int:
            edge: tuple[int, int] = (min(v1,v2), max(v1,v2))
            if edge not in edge2midpoint:
                try:
                    midpoint = (np.array(vertices[v1]) + np.array(vertices[v2])) / 2.0
                    norm = np.linalg.norm(midpoint)
                    midpoint = midpoint / norm  # Normalize to keep on sphere
                    edge2midpoint[edge] = len(vertices)  # Index of the vertex that will be appended next
                    vertices.append(midpoint.tolist())
                except ZeroDivisionError as err:
                    raise ZeroDivisionError(f"Midpoint between vertices {v1} and {v2} has norm 0: {err}")
                except IndexError as err:
                    wrong_idx = edge[1]
                    if edge[0] >= len(vertices): wrong_idx = edge
                    raise IndexError(f"Vertex index(ices) {wrong_idx} out of bounds: {err}")
                except Exception as err:
                    raise ValueError(f"Error calculating midpoint for vertices {v1} and {v2}: {err}")
            return edge2midpoint[edge]

        for _ in range(depth):
            try:
                new_faces = []
                edge2midpoint: dict[tuple[int,int], int] = {}

                # Subdivide each triangle into four smaller triangles
                for tri in faces:
                    try:
                        v1, v2, v3 = tri
                        mid_1_2 = add_and_return_midpoint(v1, v2, vertices, edge2midpoint)
                        mid_2_3 = add_and_return_midpoint(v2, v3, vertices, edge2midpoint)
                        mid_3_1 = add_and_return_midpoint(v3, v1, vertices, edge2midpoint)

                        # Create four new triangles
                        new_faces.extend([
                            [v1, mid_1_2, mid_3_1],
                            [v2, mid_2_3, mid_1_2],
                            [v3, mid_3_1, mid_2_3],
                            [mid_1_2, mid_2_3, mid_3_1]
                        ])
                    except ValueError as err:
                        raise ValueError(f"Error subdividing triangle {tri}: {err}")

                # Replace old faces with new ones
                faces = new_faces

            except Exception as e:
                print(f"An error occurred during geodesic subdivision at depth {_}: {e}")
                raise

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
