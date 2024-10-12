import numpy as np



class GeodesicGrid:
    # Create a basic geodesic grid (icosahedron-based)
    def __init__(self, resolution=0):
        self.vertices, self.faces = self.make_icosahedron()
        self.vertices, self.faces = self.geodesic_subdivide(resolution)


    @staticmethod
    def make_icosahedron():
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
        vertices = self.vertices.tolist()  # Convert to list for easier handling
        for _ in range(depth):
            new_faces = []
            vertex_dict = {}

            def get_midpoint(v1, v2):
                key = tuple(sorted((v1, v2)))
                if key not in vertex_dict:
                    midpoint = (np.array(vertices[v1]) + np.array(vertices[v2])) / 2.0
                    vertex_dict[key] = len(vertices)
                    vertices.append((midpoint / np.linalg.norm(midpoint)).tolist())  # Normalize to keep on sphere
                return vertex_dict[key]

            # Subdivide each triangle
            for tri in self.faces:
                v1, v2, v3 = tri
                a = get_midpoint(v1, v2)
                b = get_midpoint(v2, v3)
                c = get_midpoint(v3, v1)

                # Create four new triangles
                new_faces.extend([
                    [v1, a, c],
                    [v2, b, a],
                    [v3, c, b],
                    [a, b, c]
                ])

            self.faces = new_faces

        return np.array(vertices), np.array(self.faces)
