import numpy as np
from scipy.spatial.transform import Rotation
from scipy import sparse


def rotation_mat_x(angle_rad: float) -> np.ndarray:
    """Rotation matrix to rotate a vector around the X-axis by a given angle (in radians)."""
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return rotation_matrix

def rotation_mat_y(angle_rad: float) -> np.ndarray:
    """Rotation matrix to rotate a vector around the Y-axis by a given angle (in radians)."""
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    return rotation_matrix

def rotation_mat_z(angle_rad: float) -> np.ndarray:
    """Rotation matrix to rotate a vector around the Z-axis by a given angle (in radians)."""
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    return rotation_matrix


def rotate_vector_rodrigues(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a vector around an arbitrary axis by a given angle (in radians) using Rodrigues' rotation formula."""
    axis = axis / np.linalg.norm(axis)  # Normalize the axis of rotation
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # Rodrigues' rotation formula
    rotated_vector = (vector * cos_theta +
                      np.cross(axis, vector) * sin_theta +
                      axis * np.dot(axis, vector) * (1 - cos_theta))
    return rotated_vector


def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate a vector around an arbitrary axis using quaternions."""
    axis = axis / np.linalg.norm(axis)  # Normalize the axis of rotation
    rotation = Rotation.from_rotvec(angle_rad * axis)
    return rotation.apply(vector)

deg2rad = lambda ang_deg: np.pi * ang_deg / 180.0

normalize = lambda vec: vec / np.linalg.norm(vec, axis=-1)[..., None]


def cartesian_to_spherical(vertex: np.ndarray):
    x, y, z = tuple(vertex.tolist())
    r = np.linalg.norm(vertex)
    longitude = np.degrees(np.arctan2(y, x))
    latitude = np.degrees(np.arcsin(z / r))
    return latitude, longitude, r


def subsolar_point(sunlight_vector: np.ndarray) -> tuple[float, float]:
    sunlight_vector /= np.linalg.norm(sunlight_vector)
    x, y, z = tuple(sunlight_vector.tolist())
    latitude = np.rad2deg(np.arcsin(z))
    longitude = np.rad2deg(np.arctan2(y, x))
    return float(latitude), float(longitude)



def build_gradient_operators(latitude, longitude, weights: sparse.csr_matrix, radius: float = 1.0):
    """
    Constructs gradient operators for partial derivatives with respect to latitude and longitude.

    Parameters:
    - latitude: NumPy array of shape (N,) containing latitudes of vertices, in degrees.
    - longitude: NumPy array of shape (N,) containing longitudes of vertices, in degrees.
    - weights: Sparse csr array of shape (N, N) containing weights for each neighbor.
    - radius: (optional) Radius of the sphere

    Returns:
    - M_phi: scipy.sparse.csr_matrix of shape (N, N) for ∂f/∂phi.
    - M_lambda: scipy.sparse.csr_matrix of shape (N, N) for ∂f/∂lambda.
    """
    N = latitude.shape[0]

    num_neighbors = np.bincount(weights.nonzero()[0])
    M_max = np.amax(num_neighbors)
    neighbors_array = np.full((N, M_max), -1, dtype=int)  # Use -1 as placeholder
    for i in range(N):
        neighbors_array[i, :num_neighbors[i]] = weights[i].nonzero()[1]

    # Initialize lists to construct sparse matrices
    data_phi = []
    rows_phi = []
    cols_phi = []

    data_lambda = []
    rows_lambda = []
    cols_lambda = []

    # Precompute cos(latitude) for scaling longitude differences
    cos_lat = np.cos(np.deg2rad(latitude))  # Convert degrees to radians if necessary

    for i in range(N):
        # Extract neighbor indices for vertex i
        nbr_indices = neighbors_array[i]
        valid = nbr_indices != -1  # Boolean mask for valid neighbors
        valid_nbrs = nbr_indices[valid]
        M = valid_nbrs.size  # Number of valid neighbors

        if M == 0:
            # No neighbors; skip derivative computation
            continue

        # Extract differences
        delta_phi = np.deg2rad(latitude[valid_nbrs] - latitude[i])  # (M,)
        delta_lambda = np.deg2rad(longitude[valid_nbrs] - longitude[i])  # (M,)

        # Optional: Handle longitude wrapping (e.g., -180 to 180)
        delta_lambda = (delta_lambda + np.pi) % (2*np.pi) - np.pi  # Wrap to [-180, 180]

        # Scale delta_lambda by cos(phi) to account for spherical geometry
        delta_lambda_scaled = delta_lambda * cos_lat[i] * radius
        delta_phi_scaled = delta_phi * radius

        # Extract weights
        w = weights[i].data  # (M,)

        # Form matrix A (M x 2)
        A = np.vstack((delta_phi_scaled, delta_lambda_scaled)).T  # Shape (M, 2)

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

        # Extract coefficients for phi and lambda
        C_phi = C[0, :]  # Shape (M,)
        C_lambda = C[1, :]  # Shape (M,)

        # Compute central coefficients to account for f_i
        C_i_phi = -np.sum(C_phi)
        C_i_lambda = -np.sum(C_lambda)

        # Assign coefficients to the sparse matrices
        # For M_phi
        for idx, j in enumerate(valid_nbrs):
            rows_phi.append(i)
            cols_phi.append(j)
            data_phi.append(C_phi[idx])

        # Central point
        rows_phi.append(i)
        cols_phi.append(i)
        # C_i_phi multiplies f_i
        data_phi.append(C_i_phi)

        # For M_lambda
        for idx, j in enumerate(valid_nbrs):
            rows_lambda.append(i)
            cols_lambda.append(j)
            data_lambda.append(C_lambda[idx])

        # Central point
        rows_lambda.append(i)
        cols_lambda.append(i)
        # C_i_lambda multiplies f_i
        data_lambda.append(C_i_lambda)

    # Create sparse matrices in COO format
    M_phi_coo = sparse.coo_matrix((data_phi, (rows_phi, cols_phi)), shape=(N, N))
    M_lambda_coo = sparse.coo_matrix((data_lambda, (rows_lambda, cols_lambda)), shape=(N, N))

    # Convert to CSR format for efficient arithmetic operations
    M_phi = M_phi_coo.tocsr()
    M_lambda = M_lambda_coo.tocsr()

    return M_phi, M_lambda