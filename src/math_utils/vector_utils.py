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

normalize = lambda vec: vec / np.linalg.norm(vec, axis=-1)[..., None]


def cartesian_to_spherical(vertex: np.ndarray):
    x, y, z = tuple(vertex.tolist())
    r = np.linalg.norm(vertex)
    longitude = np.degrees(np.arctan2(y, x))
    latitude = np.degrees(np.arcsin(z / r))
    return latitude, longitude, r


def spherical_to_cartesian(coordinates: np.ndarray):
    latitude, longitude, r = tuple(coordinates.T.tolist())
    x = r * np.cos(np.deg2rad(latitude)) * np.cos(np.deg2rad(longitude))
    y = r * np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(longitude))
    z = r * np.sin(np.deg2rad(latitude))
    return np.stack([x.T, y.T, z.T], axis=-1)


def subsolar_point(sunlight_vector: np.ndarray) -> tuple[float, float]:
    sunlight_vector /= np.linalg.norm(sunlight_vector)
    x, y, z = tuple(sunlight_vector.tolist())
    latitude = np.rad2deg(np.arcsin(z))
    longitude = np.rad2deg(np.arctan2(y, x))
    return float(latitude), float(longitude)



def polar_to_cartesian_velocity(zonal, meridional, vertical, cartesian_coords) -> np.ndarray:
    """
    Convert velocity components from polar coordinates to Cartesian coordinates.

    Args:
        zonal: Velocity components along the eastward direction (m/s) (scalar or array).
        meridional: Velocity components along the northward direction (m/s) (scalar or array).
        vertical: Velocity components along the vertical direction (m/s) (scalar or array).
        cartesian_coords: Cartesian coordinates (x, y, z) of the points (array-like of shape (..., 3)).

    Returns:
        np.ndarray: Velocities in Cartesian coordinates (vx, vy, vz) for all input points.
    """
    
    # Normalize Cartesian coordinates to get the 'up' direction
    up = cartesian_coords / np.linalg.norm(cartesian_coords, axis=-1, keepdims=True)
    
    # Calculate the east vector as the cross product of up and the Z-axis (0, 0, 1)
    z_axis = np.array([0, 0, 1])
    east = np.cross(z_axis, up)
    east = east / np.linalg.norm(east, axis=-1, keepdims=True)
    
    # Calculate the north vector as the cross product of up and east
    north = np.cross(up, east)
    north = north / np.linalg.norm(north, axis=-1, keepdims=True)
    
    # Combine velocity components into Cartesian coordinates
    cartesian_velocity = (zonal[..., None] * east +
                          meridional[..., None] * north +
                          vertical[..., None] * up)
    
    return cartesian_velocity


def cartesian_to_polar_velocity(cartesian_velocity: np.ndarray, cartesian_coords: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert velocity from Cartesian coordinates to polar velocity components.

    Args:
        cartesian_velocity: Cartesian velocity components (vx, vy, vz) (array-like of shape (..., 3)).
        cartesian_coords: Cartesian coordinates (x, y, z) of the points (array-like of shape (..., 3)).

    Returns:
        tuple: Zonal (eastward), meridional (northward), and vertical components of velocity 
               as arrays of the same shape as the input components.
    """

    # Normalize Cartesian coordinates to get the 'up' direction
    up = cartesian_coords / np.linalg.norm(cartesian_coords, axis=-1, keepdims=True)

    # Calculate the east vector as the cross product of up and the Z-axis (0, 0, 1)
    z_axis = np.array([0, 0, 1])
    east = np.cross(z_axis, up)
    east = east / np.linalg.norm(east, axis=-1, keepdims=True)

    # Calculate the north vector as the cross product of up and east
    north = np.cross(up, east)
    north = north / np.linalg.norm(north, axis=-1, keepdims=True)

    # Compute zonal, meridional, and vertical components
    zonal = np.sum(cartesian_velocity * east, axis=-1)
    meridional = np.sum(cartesian_velocity * north, axis=-1)
    vertical = np.sum(cartesian_velocity * up, axis=-1)

    return zonal, meridional, vertical
