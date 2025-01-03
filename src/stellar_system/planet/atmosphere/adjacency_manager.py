import numpy as np
from scipy import sparse

from .air_data import AirData
from src.math_utils import VectorOperatorsSpherical
from src.math_utils.vector_utils import spherical_to_cartesian

class AdjacencyManager:
    def __init__(self, air_data: AirData, horizontal_adjacency_matrix: sparse.coo_matrix):
        self.air_data = air_data
        self.horizontal_adjacency_matrix = horizontal_adjacency_matrix

        self.n_layers = self.air_data.n_layers
        self.n_columns = self.air_data.n_columns
        self.atmosphere_shape = (self.n_layers, self.n_columns)
        # self.is_underground = self.air_data.is_underground
        # self.lowest_layer_above_surface = self.air_data.lowest_layer_above_surface

        self.latitude = np.full(self.atmosphere_shape, fill_value=self.air_data.surface.latitude)
        self.longitude = np.full(self.atmosphere_shape, fill_value=self.air_data.surface.longitude)
        self.radius = self.air_data.surface.radius + self.air_data.altitudes
        self.coordinates = np.stack([self.longitude.ravel(), self.latitude.ravel(), self.radius.ravel()], axis=-1)
        self.cartesian = spherical_to_cartesian(self.coordinates)

        self.adjacency_matrix = self.build_layered_adjacency_matrix(horizontal_adjacency_matrix)
        self.vector_operators = VectorOperatorsSpherical(self.longitude.ravel(), self.latitude.ravel(), self.radius.ravel(), self.adjacency_matrix)
        self.laplacian_matrix = self.vector_operators.laplacian_operator



    def build_layered_adjacency_matrix(self, horizontal_adjacency_matrix: sparse.coo_matrix):
        """
        Given the horizontal adjacency matrix, build another one that also represents vertical connections.

        :param horizontal_adjacency_matrix: NxN adjacency matrix for the surface where N = surface.n_vertices.
                                            Only off-diagonal entries represent connection weights. The weights are the
                                            inverse Euclidean distances.
                                            Note: horizontal_adjacency_matrix should not contain diagonal entries.
                                            These will be computed separately when constructing the Laplacian.

        :return A: The layered adjacency matrix, which includes not only all the information in
        `horizontal_adjacency_matrix`, but also vertical adjacency connections. It has shape NxN, where
        N = n_layers * surface.n_vertices
        """
        # Horizontal adjacency (block diagonal matrix)
        A_blocks = [horizontal_adjacency_matrix] * self.n_layers
        A_block_diag = sparse.block_diag(A_blocks)

        # Add vertical adjacency
        row_indices = []
        col_indices = []
        data = []

        for c_idx in range(self.n_columns):
            # bottom_layer = self.lowest_layer_above_surface[v_idx]
            for layer_idx in range(0, self.n_layers - 1):
                dz = (self.radius[layer_idx + 1, c_idx] - self.radius[layer_idx, c_idx])
                vertical_weight = 1.0 / dz
                i = layer_idx * self.n_columns + c_idx
                j = (layer_idx + 1) * self.n_columns + c_idx

                # Add vertical adjacency (symmetric)
                row_indices.extend([i, j])
                col_indices.extend([j, i])
                data.extend([vertical_weight, vertical_weight])

        V = sparse.coo_matrix((data, (row_indices, col_indices)))

        # Combine horizontal and vertical adjacency
        A = A_block_diag + V
        A.eliminate_zeros()

        return A.tocsc()
