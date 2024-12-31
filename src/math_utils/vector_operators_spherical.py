import numpy as np
from scipy import sparse



class VectorOperatorsSpherical:
    def __init__(self, latitude: np.ndarray, longitude: np.ndarray, radius: np.ndarray, weights: sparse.csr_matrix):
        self.latitude = latitude
        self.longitude = longitude
        self.radius = radius
        self.weights = weights

        self.partial_derivative_operators = self.build_partial_derivative_operators()
        self.zonal_operator, self.meridional_operator, self.vertical_operator = self.partial_derivative_operators
        self.laplacian_operator = self.build_laplacian_matrix(self.weights)

    def build_partial_derivative_operators(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        """
        Constructs gradient operators for partial derivatives with respect to latitude, longitude, and elevation.
    
        Returns:
        - M_lambda: scipy.sparse.csr_matrix of shape (N, N) for ∂f/∂lambda.
        - M_phi: scipy.sparse.csr_matrix of shape (N, N) for ∂f/∂phi.
        - M_h: scipy.sparse.csr_matrix of shape (N, N) for ∂f/∂h.
        """
        N = self.latitude.shape[0]
    
        # Initialize lists to construct sparse matrices
        data_lambda = []
        rows_lambda = []
        cols_lambda = []
    
        data_phi = []
        rows_phi = []
        cols_phi = []
    
        data_h = []
        rows_h = []
        cols_h = []
    
        # Precompute cos(latitude) for scaling longitude differences
        cos_lat = np.cos(np.deg2rad(self.latitude))  # Convert degrees to radians if necessary
    
        for i in range(N):
            # Extract neighbor indices for vertex i
            neighbors_indices = self.weights[i].nonzero()[1]
    
            # Extract differences
            delta_lambda = np.deg2rad(self.longitude[neighbors_indices] - self.longitude[i])  # (M,)
            delta_phi = np.deg2rad(self.latitude[neighbors_indices] - self.latitude[i])  # (M,)
            delta_h = self.radius[neighbors_indices] - self.radius[i]  # (M,)
    
            # Optional: Handle longitude wrapping (e.g., -180 to 180)
            delta_lambda = (delta_lambda + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-180, 180]
    
            # Scale delta_lambda by cos(phi) to account for spherical geometry
            delta_lambda_scaled = delta_lambda * cos_lat[i] * (self.radius[neighbors_indices] + self.radius[i]) / 2
            delta_phi_scaled = delta_phi * (self.radius[neighbors_indices] + self.radius[i]) / 2
            delta_h_scaled = delta_h  # No additional scaling for elevation
    
            # Extract weights
            w = self.weights[i].data  # (M,)
    
            # Form matrix A (M x 3)
            A = np.vstack((delta_lambda_scaled, delta_phi_scaled, delta_h_scaled)).T  # Shape (M, 3)
    
            # Form weight matrix W (M x M), but we'll apply weights directly
            # Compute A^T W A
            AtW = A.T * w  # Each column of A.T multiplied by w
            AtWA = AtW @ A  # Shape (3, 3)
    
            # Check if AtWA is invertible
            if np.linalg.cond(AtWA) > 1 / np.finfo(AtWA.dtype).eps:
                # Singular or ill-conditioned; skip or handle appropriately
                # Here, we choose to skip and leave derivatives as zero
                continue
    
            # Compute C = (A^T W A)^-1 A^T W
            C = np.linalg.inv(AtWA) @ AtW  # Shape (3, M)
    
            # Extract coefficients for lambda, phi, and h
            C_lambda = C[0, :]  # Shape (M,)
            C_phi = C[1, :]  # Shape (M,)
            C_h = C[2, :]  # Shape (M,)
    
            # Assign coefficients to the sparse matrices
            for idx, j in enumerate(neighbors_indices):
                rows_lambda.append(i)
                cols_lambda.append(j)
                data_lambda.append(C_lambda[idx])
    
                rows_phi.append(i)
                cols_phi.append(j)
                data_phi.append(C_phi[idx])
    
                rows_h.append(i)
                cols_h.append(j)
                data_h.append(C_h[idx])
    
            # Central point
            rows_lambda.append(i)
            cols_lambda.append(i)
            data_lambda.append(-np.sum(C_lambda))  # C_i_lambda multiplies f_i
    
            rows_phi.append(i)
            cols_phi.append(i)
            data_phi.append(-np.sum(C_phi))  # C_i_phi multiplies f_i
    
            rows_h.append(i)
            cols_h.append(i)
            data_h.append(-np.sum(C_h))  # C_i_h multiplies f_i
    
        # Create sparse matrices in COO format, then convert to CSR
        M_lambda = sparse.coo_matrix((data_lambda, (rows_lambda, cols_lambda)), shape=(N, N)).tocsr()
        M_phi = sparse.coo_matrix((data_phi, (rows_phi, cols_phi)), shape=(N, N)).tocsr()
        M_h = sparse.coo_matrix((data_h, (rows_h, cols_h)), shape=(N, N)).tocsr()
    
        return M_lambda, M_phi, M_h



    def calculate_gradient(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of a scalar field defined on a spherical surface.
    
        :param values: NumPy array of shape (N,) containing scalar values at vertices.
        :return grad: NumPy array of shape (N,2) representing (∂f/∂lambda, ∂f/∂phi) for each vertex.
        """
        shape = values.shape
        values = values.flatten()

        # Perform matrix-vector multiplication to compute the gradients
        grad_lambda = self.zonal_operator.dot(values)  # ∂f/∂lambda for each vertex
        grad_phi = self.meridional_operator.dot(values)  # ∂f/∂phi for each vertex
        grad_h = self.vertical_operator.dot(values)  # ∂f/∂h for each vertex
        gradient = np.stack([grad_lambda, grad_phi, grad_h], axis=-1)
        return gradient.reshape(shape + (3,))



    def calculate_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Computes the divergence of a vector field defined on a spherical surface.
    
        Parameters:
        - vector_field: NumPy array of shape (..., 3) representing the vector field (v_lambda, v_phi, v_h) at each vertex.
        Returns:
        - divergence: Flat NumPy array representing the divergence of the vector field at each vertex.
        """
        shape = vector_field.shape
        vector_field = vector_field.reshape((-1, 3))
    
        # Split the vector field into components
        v_lambda = vector_field[:, 0]  # ∂f/∂lambda
        v_phi = vector_field[:, 1]  # ∂f/∂phi
        v_h = vector_field[:, 2]  # ∂f/∂h
    
        # Compute divergence using matrix-vector multiplication
        div_lambda = self.zonal_operator.dot(v_lambda)  # Partial derivative of v_lambda with respect to lambda
        div_phi = self.meridional_operator.dot(v_phi)  # Partial derivative of v_phi with respect to phi
        div_h = self.vertical_operator.dot(v_h)  # Partial derivative of v_h with respect to h
        divergence = div_lambda + div_phi + div_h
        return divergence.reshape(shape[:-1])



    def calculate_curl(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Computes the curl of a vector field defined on a spherical surface.
        Parameters:
        - vector_field: NumPy array of shape (..., 3) representing the vector field (v_lambda, v_phi, v_h) at each vertex.
        Returns:
        - curl: NumPy array of shape (...,3) representing the curl of the vector field at each vertex.
        """
        shape = vector_field.shape
        vector_field = vector_field.reshape((-1,3))

        # Split the vector field into components
        v_lambda = vector_field[:, 0]  # Zonal component
        v_phi = vector_field[:, 1]  # Meridional component
        v_h = vector_field[:, 2]  # Vertical component

        # Compute partial derivatives
        curl_lambda = self.meridional_operator.dot(v_h) - self.vertical_operator.dot(v_phi)  # ∂v_h/∂phi - ∂v_phi/∂h
        curl_phi = self.vertical_operator.dot(v_lambda) - self.zonal_operator.dot(v_h)  # ∂v_lambda/∂h - ∂v_h/∂lambda
        curl_h = self.zonal_operator.dot(v_phi) - self.meridional_operator.dot(v_lambda)  # ∂v_phi/∂lambda - ∂v_lambda/∂phi

        # Combine curl components
        curl = np.stack([curl_lambda, curl_phi, curl_h], axis=-1)
        return curl.reshape(shape)



    def calculate_vector_gradient(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of a vector field defined on a spherical surface.

        :param vector_field: NumPy array of shape (..., 3) containing the vector field components
                             (v_lambda, v_phi, v_h) at each vertex.
        :return: NumPy array of shape (N, 3, 3) representing the gradient tensor with
                 ∂v/∂lambda, ∂v/∂phi, and ∂v/∂h for each vertex.
        """
        shape = vector_field.shape
        vector_field = vector_field.reshape((-1, 3))
        N = vector_field.shape[0]
        gradient_tensor = np.zeros((N, 3, 3))

        # Split the vector field into zonal, meridional, and vertical components
        v_lambda = vector_field[:, 0]  # Zonal component of the vector field
        v_phi = vector_field[:, 1]  # Meridional component of the vector field
        v_h = vector_field[:, 2]  # Vertical component of the vector field

        # Compute partial derivatives
        gradient_tensor[:,0,0] = self.zonal_operator.dot(v_lambda)  # ∂v_lambda/∂lambda
        gradient_tensor[:,0,1] = self.meridional_operator.dot(v_lambda)  # ∂v_lambda/∂phi
        gradient_tensor[:,0,2] = self.vertical_operator.dot(v_lambda)  # ∂v_lambda/∂h

        gradient_tensor[:,1,0] = self.zonal_operator.dot(v_phi)  # ∂v_phi/∂lambda
        gradient_tensor[:,1,1] = self.meridional_operator.dot(v_phi)  # ∂v_phi/∂phi
        gradient_tensor[:,1,2] = self.vertical_operator.dot(v_phi)  # ∂v_phi/∂h

        gradient_tensor[:,2,0] = self.zonal_operator.dot(v_h)  # ∂v_h/∂lambda
        gradient_tensor[:,2,1] = self.meridional_operator.dot(v_h)  # ∂v_h/∂phi
        gradient_tensor[:,2,2] = self.vertical_operator.dot(v_h)  # ∂v_h/∂h

        return gradient_tensor.reshape(shape + (3,))



    @staticmethod
    def build_laplacian_matrix(A: sparse.csr_matrix):
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

    def calculate_laplacian(self, values: np.ndarray) -> np.ndarray:
        shape = values.shape
        N = self.laplacian_operator.shape[0]
        values = values.reshape((N,-1))
        laplacian = self.laplacian_operator.dot(values)
        return laplacian.reshape(shape)
