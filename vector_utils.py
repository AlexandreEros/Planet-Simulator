import numpy as np
from scipy.spatial.transform import Rotation


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


