import numpy as np
from scipy import sparse



class VectorOperatorsSpherical:
    def __init__(self, latitude: np.ndarray, longitude: np.ndarray, weights: sparse.csr_matrix, radius: float = 1.0):
        self.latitude = latitude
        self.longitude = longitude
        self.weights = weights
        self.radius = radius

        self.zonal_operator, self.meridional_operator = self.build_partial_derivative_operators()    

    def build_partial_derivative_operators(self) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """
        Constructs gradient operators for partial derivatives with respect to latitude and longitude.
    
        Returns:
        - M_lambda: scipy.sparse.csr_matrix of shape (N, N) for ∂f/∂lambda.
        - M_phi: scipy.sparse.csr_matrix of shape (N, N) for ∂f/∂phi.
        """
        N = self.latitude.shape[0]
    
        # Initialize lists to construct sparse matrices
        data_lambda = []
        rows_lambda = []
        cols_lambda = []
    
        data_phi = []
        rows_phi = []
        cols_phi = []
    
        # Precompute cos(latitude) for scaling longitude differences
        cos_lat = np.cos(np.deg2rad(self.latitude))  # Convert degrees to radians if necessary
    
        for i in range(N):
            # Extract neighbor indices for vertex i
            neighbors_indices = self.weights[i].nonzero()[1]
    
            # Extract differences
            delta_lambda = np.deg2rad(self.longitude[neighbors_indices] - self.longitude[i])  # (M,)
            delta_phi = np.deg2rad(self.latitude[neighbors_indices] - self.latitude[i])  # (M,)
    
            # Optional: Handle longitude wrapping (e.g., -180 to 180)
            delta_lambda = (delta_lambda + np.pi) % (2*np.pi) - np.pi  # Wrap to [-180, 180]
    
            # Scale delta_lambda by cos(phi) to account for spherical geometry
            delta_lambda_scaled = delta_lambda * cos_lat[i] * self.radius
            delta_phi_scaled = delta_phi * self.radius
    
            # Extract weights
            w = self.weights[i].data  # (M,)
    
            # Form matrix A (M x 2)
            A = np.vstack((delta_lambda_scaled, delta_phi_scaled)).T  # Shape (M, 2)
    
            # Form weight matrix W (M x M), but we'll apply weights directly
            # Compute A^T W A
            AtW = A.T * w  # Each column of A.T multiplied by w
            AtWA = AtW @ A  # Shape (2, 2)
    
            # Check if AtWA is invertible
            if np.linalg.cond(AtWA) > 1 / np.finfo(AtWA.dtype).eps:
                # Singular or ill-conditioned; skip or handle appropriately
                # Here, we choose to skip and leave derivatives as zero
                continue
    
            # Compute C = (A^T W A)^-1 A^T W
            C = np.linalg.inv(AtWA) @ AtW  # Shape (2, M)
    
            # Extract coefficients for lambda and phi
            C_lambda = C[0, :]  # Shape (M,)
            C_phi = C[1, :]  # Shape (M,)
    
            # Assign coefficients to the sparse matrices
            for idx, j in enumerate(neighbors_indices):
                rows_lambda.append(i)
                cols_lambda.append(j)
                data_lambda.append(C_lambda[idx])
    
                rows_phi.append(i)
                cols_phi.append(j)
                data_phi.append(C_phi[idx])
    
            # Central point
            rows_lambda.append(i)
            cols_lambda.append(i)
            data_lambda.append(-np.sum(C_lambda))  # C_i_lambda multiplies f_i
    
            rows_phi.append(i)
            cols_phi.append(i)
            data_phi.append(-np.sum(C_phi))  # C_i_phi multiplies f_i
    
        # Create sparse matrices in COO format, then convert to CSR
        M_lambda = sparse.coo_matrix((data_lambda, (rows_lambda, cols_lambda)), shape=(N, N)).tocsr()
        M_phi = sparse.coo_matrix((data_phi, (rows_phi, cols_phi)), shape=(N, N)).tocsr()
    
        return M_lambda, M_phi


    def calculate_gradient(self, values: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of a scalar field defined on a spherical surface.
    
        :param values: NumPy array of shape (N,) containing scalar values at vertices.
        :return grad: NumPy array of shape (N,2) representing (∂f/∂lambda, ∂f/∂phi) for each vertex.
        """
        # Perform matrix-vector multiplication to compute the gradients
        grad_lambda = self.zonal_operator.dot(values)  # ∂f/∂lambda for each vertex
        grad_phi = self.meridional_operator.dot(values)  # ∂f/∂phi for each vertex
        return np.stack([grad_lambda, grad_phi], axis=-1)


    def calculate_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Computes the divergence of a vector field defined on a spherical surface.
    
        Parameters:
        - vector_field: NumPy array of shape (N, 2) representing the vector field (v_lambda, v_phi) at each vertex.
        Returns:
        - divergence: NumPy array of shape (N,) representing the divergence of the vector field at each vertex.
        """
    
        # Split the vector field into components
        v_lambda = vector_field[:, 0]  # ∂f/∂lambda
        v_phi = vector_field[:, 1]  # ∂f/∂phi
    
        # Compute divergence using matrix-vector multiplication
        div_lambda = self.zonal_operator.dot(v_lambda)  # Partial derivative of v_lambda with respect to lambda
        div_phi = self.meridional_operator.dot(v_phi)  # Partial derivative of v_phi with respect to phi
        divergence = div_lambda + div_phi
        return divergence


    def calculate_curl(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Computes the curl of a vector field defined on a spherical surface.
        Parameters:
        - latitude: NumPy array of shape (N,) containing latitudes of vertices, in degrees.
        - longitude: NumPy array of shape (N,) containing longitudes of vertices, in degrees.
        - vector_field: NumPy array of shape (N, 2) representing the vector field (v_lambda, v_phi) at each vertex.
        - weights: Sparse csr array of shape (N, N) containing weights for each neighbor.
        Returns:
        - curl: NumPy array of shape (N,) representing the curl of the vector field at each vertex.
        """
    
        # Split the vector field into components
        v_lambda = vector_field[:, 0]  # v_lambda component
        v_phi = vector_field[:, 1]  # v_phi component

        # Compute partial derivatives
        curl_lambda = self.zonal_operator.dot(v_phi)  # Partial derivative of v_phi with respect to lambda
        curl_phi = self.meridional_operator.dot(v_lambda)  # Partial derivative of v_lambda with respect to phi
    
        # Compute curl as the difference of components
        curl = curl_lambda - curl_phi
        return curl


    def calculate_vector_gradient(self, vector_field: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of a vector field defined on a spherical surface.

        :param vector_field: NumPy array of shape (N, 2) containing the vector field components
                             (v_lambda, v_phi) at each vertex.
        :return: NumPy array of shape (N, 2, 2) representing the gradient tensor with
                 ∂v/∂lambda and ∂v/∂phi for each vertex.
        """
        # Split the vector field into zonal and meridional components
        v_lambda = vector_field[:, 0]  # Zonal component of the vector field
        v_phi = vector_field[:, 1]  # Meridional component of the vector field

        # Compute partial derivatives
        grad_v_lambda_lambda = self.zonal_operator.dot(v_lambda)  # ∂v_lambda/∂lambda
        grad_v_lambda_phi = self.meridional_operator.dot(v_lambda)  # ∂v_lambda/∂phi
        grad_v_phi_lambda = self.zonal_operator.dot(v_phi)  # ∂v_phi/∂lambda
        grad_v_phi_phi = self.meridional_operator.dot(v_phi)  # ∂v_phi/∂phi

        # Combine results into gradient tensor with shape (N, 2, 2)
        gradient_tensor = np.stack(
            [[grad_v_lambda_lambda, grad_v_lambda_phi],
             [grad_v_phi_lambda, grad_v_phi_phi]],
            axis=-1
        ).transpose((1, 2, 0))

        return gradient_tensor
