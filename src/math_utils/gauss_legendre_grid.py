import numpy as np
from scipy import sparse
from .vector_utils import spherical_to_cartesian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



class GaussLegendreGrid:
    def __init__(self, resolution=9, radius=1.0):
        #n_points_latitude=20, n_points_longitude=40
        if resolution==0: resolution=90
        self.n_lat = int(np.ceil(360 / resolution))
        self.n_lon = int(np.ceil(180 / resolution))
        self.radius = radius

        # Generate grid points
        latitude, longitude = self.gauss_legendre_sphere(self.n_lat, self.n_lon)
        latitude, longitude = np.degrees(latitude), np.degrees(longitude)

        # Convert to Cartesian coordinates
        # self.vertices = spherical_to_cartesian(
        #     np.column_stack((self.latitudes.ravel(), self.longitudes.ravel(), np.full_like(self.latitudes.ravel(), radius)))
        # )
        coordinates = np.stack((latitude, longitude, np.full_like(latitude, radius)), axis=-1).reshape((-1,3))
        self.latitude, self.longitude = coordinates[:,0], coordinates[:,1]
        self.vertices = np.apply_along_axis(spherical_to_cartesian, -1, coordinates)
        self.n_vertices = len(self.vertices)
        self.coordinates = np.stack((self.longitude, self.latitude, np.full_like(self.latitude, radius)), axis=-1).reshape((-1,3))


        # Create quadrilateral faces
        self.quads = self.build_quadrilaterals(self.n_lat, self.n_lon)

        # Convert quads to triangles for rendering
        self.faces = self.quadrilateral_to_triangles(self.quads)

        # Create adjacency matrix
        self.adjacency_matrix = self.build_adjacency_matrix(self.n_lat, self.n_lon, self.radius)


    def gauss_legendre_sphere(self, n_points_latitude, n_points_longitude):
        """
        Generate a Gauss-Legendre grid on a sphere.

        Parameters:
        - n_points_latitude: Number of points in the latitudinal direction.
        - n_points_longitude: Number of points in the longitudinal direction.

        Returns:
        - latitudes: 1D array of latitude points (radians).
        - longitudes: 1D array of longitude points (radians).
        """
        # Gauss-Legendre quadrature for latitudes
        x, w = np.polynomial.legendre.leggauss(n_points_latitude)
        latitudes = np.arcsin(x)  # Transform to spherical coordinates

        # Uniform spacing for longitudes
        longitudes = np.linspace(0, 2 * np.pi, n_points_longitude, endpoint=False)

        return np.meshgrid(latitudes, longitudes, indexing='ij')


    def build_quadrilaterals(self, n_lat, n_lon):
        """
        Constructs the quadrilateral faces for a Gauss-Legendre spherical grid.

        Parameters:
        - n_lat: Number of latitude points
        - n_lon: Number of longitude points

        Returns:
        - faces: List of quadrilateral faces (each face is a list of 4 vertex indices)
        """
        faces = []

        # Convert 2D (i, j) indices to 1D vertex index
        def index(lat_idx, lon_idx):
            return lat_idx * n_lon + lon_idx

        for i in range(n_lat - 1):  # Exclude the last latitude row (no neighbor below)
            for j in range(n_lon):  # Loop over all longitudes
                top_left = index(i, j)
                top_right = index(i, (j + 1) % n_lon)  # Wrap around longitudinally
                bottom_left = index(i + 1, j)
                bottom_right = index(i + 1, (j + 1) % n_lon)

                # Each quadrilateral face is defined by 4 corner points
                faces.append([top_left, top_right, bottom_right, bottom_left])

        return np.array(faces, dtype=np.int32)


    def quadrilateral_to_triangles(self, faces):
        """
        Convert quadrilateral faces into two triangular faces each.

        Parameters:
        - faces: Nx4 array where each row represents a quadrilateral as four vertex indices.

        Returns:
        - triangles: Mx3 array where each row represents a triangle as three vertex indices.
        """
        triangles = []
        for quad in faces:
            v1, v2, v3, v4 = quad
            triangles.append([v1, v2, v3])  # First triangle
            triangles.append([v1, v3, v4])  # Second triangle

        return np.array(triangles, dtype=np.int32)


    def build_adjacency_matrix(self, n_lat, n_lon, radius):
        """
        Constructs the adjacency matrix for a Gauss-Legendre spherical grid.

        Parameters:
        - n_lat: Number of latitude points
        - n_lon: Number of longitude points
        - radius: Radius of the sphere

        Returns:
        - adjacency_matrix: Sparse adjacency matrix (scipy.sparse.csr_matrix)
        """
        num_points = n_lat * n_lon
        row_indices, col_indices, weights = [], [], []

        # Compute the 2D index mapping to 1D
        def index(lat_idx, lon_idx):
            return lat_idx * n_lon + lon_idx

        # Iterate through all grid points
        for i in range(n_lat):
            for j in range(n_lon):
                current_idx = index(i, j)

                # Connect to the north neighbor (except at poles)
                if i > 0:
                    neighbor_idx = index(i - 1, j)
                    row_indices.append(current_idx)
                    col_indices.append(neighbor_idx)
                    weights.append(1 / radius)  # Inverse Euclidean distance

                # Connect to the south neighbor (except at poles)
                if i < n_lat - 1:
                    neighbor_idx = index(i + 1, j)
                    row_indices.append(current_idx)
                    col_indices.append(neighbor_idx)
                    weights.append(1 / radius)

                # Connect to the east neighbor (longitude wraps around)
                neighbor_idx = index(i, (j + 1) % n_lon)
                row_indices.append(current_idx)
                col_indices.append(neighbor_idx)
                weights.append(1 / radius)

                # Connect to the west neighbor
                neighbor_idx = index(i, (j - 1) % n_lon)
                row_indices.append(current_idx)
                col_indices.append(neighbor_idx)
                weights.append(1 / radius)

        # Construct sparse adjacency matrix
        adjacency_matrix = sparse.coo_matrix((weights, (row_indices, col_indices)), shape=(num_points, num_points))

        return adjacency_matrix.tocsr()


    def plot_gauss_legendre_grid(self):
        """
        Visualizes a Gauss-Legendre grid by rendering triangular faces.

        Parameters:
        - vertices: Nx3 array of Cartesian coordinates of grid points.
        - faces: Mx4 array where each row represents a quadrilateral as four vertex indices.
        """
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Convert index-based triangles into actual coordinate arrays
        triangle_vertices = self.vertices[self.faces]

        # Create a collection of triangular faces for 3D plotting
        poly_collection = Poly3DCollection(triangle_vertices, alpha=0.3, edgecolor="k")

        ax.add_collection3d(poly_collection)

        # Set axis limits
        max_range = np.max(np.linalg.norm(self.vertices, axis=1))
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Gauss-Legendre Grid Visualization")

        plt.show()