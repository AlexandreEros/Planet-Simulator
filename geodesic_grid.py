import numpy as np



class GeodesicGrid:
    # Create a basic geodesic grid (icosahedron-based)
    def __init__(self, resolution: int = 0):
        self.vertices, self.faces = self.build_icosahedron()
        self.vertices, self.faces = self.geodesic_subdivide(resolution)
        self.neighbors = self.build_neighbors()


    @staticmethod
    def build_icosahedron():
        # Create the vertices of an icosahedron
        phi = (1.0 + np.sqrt(5.0)) / 2.0

        vertices = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        vertices /= np.linalg.norm(vertices[0])

        # Define the icosahedron faces (triangles)
        faces = np.array([
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ])

        return vertices, faces


    def geodesic_subdivide(self, depth=0):
        """
        Subdivide each triangle in the faces list to increase the resolution.
        """
        try:
            # Convert vertices to list for easier manipulation
            vertices = self.vertices.tolist()  # Convert to list for easier handling

            def get_midpoint(v1, v2, vertex_dict):
                key = tuple(sorted((v1, v2)))
                if key not in vertex_dict:
                    try:
                        midpoint = (np.array(vertices[v1]) + np.array(vertices[v2])) / 2.0
                        norm = np.linalg.norm(midpoint)
                        if norm == 0:
                            raise ValueError
                        midpoint = midpoint / norm  # Normalize to keep on sphere
                        vertex_dict[key] = len(vertices)
                        vertices.append(midpoint.tolist())
                    except ZeroDivisionError as err:
                        raise ZeroDivisionError(f"Midpoint between vertices {v1} and {v2} at depth {_} "
                                                f"has norm 0: {err}")
                    except IndexError as err:
                        raise IndexError(f"Vertex index out of bounds: {err}")
                    except Exception as err:
                        raise ValueError(f"Error calculating midpoint for vertices {v1} and {v2}: {err}")
                return vertex_dict[key]

            for _ in range(depth):
                new_faces = []
                vertex_dict = {}

                # Subdivide each triangle into four smaller triangles
                for tri in self.faces:
                    try:
                        v1, v2, v3 = tri
                        a = get_midpoint(v1, v2, vertex_dict)
                        b = get_midpoint(v2, v3, vertex_dict)
                        c = get_midpoint(v3, v1, vertex_dict)

                        # Create four new triangles
                        new_faces.extend([
                            [v1, a, c],
                            [v2, b, a],
                            [v3, c, b],
                            [a, b, c]
                        ])
                    except ValueError as err:
                        raise ValueError(f"Error subdividing triangle {tri} at depth {_}: {err}")

                # Replace old faces with new ones
                self.faces = new_faces

            return np.array(vertices), np.array(self.faces)

        except Exception as e:
            print(f"An error occurred during geodesic subdivision: {e}")
            raise


    def build_neighbors(self):
        """
        Build an adjacency list where each vertex maps to its neighboring vertices.
        """
        neighbors = {i: set() for i in range(len(self.vertices))}
        for tri in self.faces:
            v1, v2, v3 = tri
            neighbors[v1].update([v2, v3])
            neighbors[v2].update([v1, v3])
            neighbors[v3].update([v1, v2])
        return neighbors
