import numpy as np
from scipy import sparse

from air_data import AirData

class AdjacencyManager:
    def __init__(self, layer_manager: AirData, horizontal_adjacency_matrix: sparse.coo_matrix):
        self.layer_manager = layer_manager
        self.adjacency_matrix = self.build_layered_adjacency_matrix(horizontal_adjacency_matrix)
        self.laplacian_matrix = self.build_laplacian_matrix(self.adjacency_matrix)


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
        A_blocks = [horizontal_adjacency_matrix] * self.layer_manager.n_layers
        A_block_diag = sparse.block_diag(A_blocks)

        # Add vertical adjacency
        row_indices = []
        col_indices = []
        data = []

        for layer_idx in range(self.layer_manager.n_layers - 1):
            dz = (self.layer_manager.altitudes[layer_idx + 1] - self.layer_manager.altitudes[layer_idx])
            vertical_weight = 1.0 / dz

            for v_idx in range(self.layer_manager.n_vertices):
                i = layer_idx * self.layer_manager.n_vertices + v_idx
                j = (layer_idx + 1) * self.layer_manager.n_vertices + v_idx

                # Add vertical adjacency (symmetric)
                row_indices.extend([i, j])
                col_indices.extend([j, i])
                data.extend([vertical_weight, vertical_weight])

        V = sparse.coo_matrix((data, (row_indices, col_indices)))

        # Combine horizontal and vertical adjacency
        A = A_block_diag + V
        return A


    @staticmethod
    def build_laplacian_matrix(A: sparse.coo_matrix):
        """
        Given the layered adjacency matrix (which represents vertical adjacency as well), build the corresponding
        Laplacian matrix.

        :param A: NxN adjacency matrix for the entire atmosphere, where N = n_layers * surface.n_vertices.
                  Only off-diagonal entries represent connection weights. The weights are the inverse Euclidean
                  distances
        :return L: The Laplacian matrix
        """
        # Calculate the degree matrix as the sum of each row
        row_sum = np.array(A.sum(axis=1))
        D = sparse.diags(row_sum.ravel(), format='csr')

        # Compute the Laplacian
        L = D - A
        return L
