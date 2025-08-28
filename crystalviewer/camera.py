"""
Camera utilities for Crystal Viewer
Camera positioning, presets, and animation
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .config import CAMERA_PRESETS, get_camera_preset
from .geometry import Vector3D, calculate_distance, calculate_center_of_mass


def create_camera_position(
    center: Union[Vector3D, List[float], Tuple[float, float, float]],
    distance: float,
    azimuth: float = 45.0,
    elevation: float = 30.0,
) -> List[float]:
    """
    Create camera position from center, distance, and angles

    Args:
        center: Center point of the scene
        distance: Distance from center
        azimuth: Azimuth angle in degrees
        elevation: Elevation angle in degrees

    Returns:
        Camera position as [x, y, z]
    """
    if isinstance(center, (list, tuple)):
        center = Vector3D(*center)
    elif not isinstance(center, Vector3D):
        center = Vector3D(center.x, center.y, center.z)

    # Convert angles to radians
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)

    # Calculate position
    x = center.x + distance * math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = center.y + distance * math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = center.z + distance * math.sin(elevation_rad)

    return [x, y, z]


def calculate_optimal_camera_distance(
    structure_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    multiplier: float = 2.0
) -> float:
    """
    Calculate optimal camera distance based on structure bounds

    Args:
        structure_bounds: ((min_x, max_x), (min_y, max_y), (min_z, max_z))
        multiplier: Distance multiplier

    Returns:
        Optimal camera distance
    """
    if not HAS_NUMPY:
        # Simple calculation without numpy
        size_x = structure_bounds[0][1] - structure_bounds[0][0]
        size_y = structure_bounds[1][1] - structure_bounds[1][0]
        size_z = structure_bounds[2][1] - structure_bounds[2][0]
        max_size = max(size_x, size_y, size_z)
        return max_size * multiplier

    # Calculate diagonal of bounding box
    bounds = np.array(structure_bounds)
    diagonal = np.linalg.norm(bounds[:, 1] - bounds[:, 0])
    return diagonal * multiplier


def create_rotated_camera_position(
    center: Union[Vector3D, List[float], Tuple[float, float, float]],
    distance: float,
    rotation_angle: float,
    axis: str = "z",
) -> List[float]:
    """
    Create camera position rotated around an axis

    Args:
        center: Center point
        distance: Distance from center
        rotation_angle: Rotation angle in degrees
        axis: Rotation axis ('x', 'y', 'z')

    Returns:
        Camera position as [x, y, z]
    """
    if isinstance(center, (list, tuple)):
        center = Vector3D(*center)

    angle_rad = math.radians(rotation_angle)

    if axis == "x":
        x = center.x
        y = center.y + distance * math.cos(angle_rad)
        z = center.z + distance * math.sin(angle_rad)
    elif axis == "y":
        x = center.x + distance * math.cos(angle_rad)
        y = center.y
        z = center.z + distance * math.sin(angle_rad)
    elif axis == "z":
        x = center.x + distance * math.cos(angle_rad)
        y = center.y + distance * math.sin(angle_rad)
        z = center.z
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

    return [x, y, z]


def interpolate_camera_positions(
    start_config: List[List[float]],
    end_config: List[List[float]],
    steps: int = 10
) -> List[List[List[float]]]:
    """
    Interpolate between two camera configurations

    Args:
        start_config: Starting camera configuration [position, look_at, up_direction]
        end_config: Ending camera configuration [position, look_at, up_direction]
        steps: Number of interpolation steps

    Returns:
        List of interpolated camera configurations
    """
    if not HAS_NUMPY:
        raise ImportError("NumPy required for camera interpolation")

    start_pos = np.array(start_config[0])
    start_look_at = np.array(start_config[1])
    start_up = np.array(start_config[2])

    end_pos = np.array(end_config[0])
    end_look_at = np.array(end_config[1])
    end_up = np.array(end_config[2])

    configs = []
    for i in range(steps):
        t = i / (steps - 1)

        # Linear interpolation for position and look at
        position = start_pos + t * (end_pos - start_pos)
        look_at = start_look_at + t * (end_look_at - start_look_at)

        # Spherical interpolation for up_direction (to maintain smoothness)
        start_view = start_up / np.linalg.norm(start_up)
        end_view = end_up / np.linalg.norm(end_up)

        dot_product = np.clip(np.dot(start_view, end_view), -1, 1)
        angle = np.arccos(dot_product)

        if np.abs(angle) < 1e-6:
            up_direction = start_up
        else:
            sin_angle = np.sin(angle)
            up_direction = (
                np.sin((1 - t) * angle) / sin_angle * start_view +
                np.sin(t * angle) / sin_angle * end_view
            )

        configs.append([
            position.tolist(),
            look_at.tolist(),
            up_direction.tolist()
        ])

    return configs


def create_camera_path(
    center: Union[Vector3D, List[float], Tuple[float, float, float]],
    distance: float,
    num_points: int = 36,
    axis: str = "z",
) -> List[List[float]]:
    """
    Create a circular camera path around the center

    Args:
        center: Center point
        distance: Distance from center
        num_points: Number of points in the path
        axis: Rotation axis ('x', 'y', 'z')

    Returns:
        List of camera positions
    """
    positions = []
    for i in range(num_points):
        angle = 360.0 * i / num_points
        pos = create_rotated_camera_position(center, distance, angle, axis)
        positions.append(pos)

    return positions


def get_camera_preset(preset: str) -> Dict[str, Any]:
    """
    Get camera preset configuration

    Args:
        preset: Preset name

    Returns:
        Camera configuration dictionary
    """
    return CAMERA_PRESETS.get(preset, CAMERA_PRESETS["isometric"])


def calculate_camera_bounds(
    positions: List[List[float]],
    padding: float = 0.1
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate camera bounds for a set of positions

    Args:
        positions: List of 3D positions
        padding: Padding factor

    Returns:
        Bounds as ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    """
    if not positions:
        return ((0, 0), (0, 0), (0, 0))

    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)
    min_z = min(p[2] for p in positions)
    max_z = max(p[2] for p in positions)

    # Add padding
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    min_x -= x_range * padding
    max_x += x_range * padding
    min_y -= y_range * padding
    max_y += y_range * padding
    min_z -= z_range * padding
    max_z += z_range * padding

    return ((min_x, max_x), (min_y, max_y), (min_z, max_z))


def create_orbital_camera_path(
    center: Union[Vector3D, List[float], Tuple[float, float, float]],
    distance: float,
    num_points: int = 60,
    inclination: float = 30.0,
) -> List[List[float]]:
    """
    Create an orbital camera path with variable inclination

    Args:
        center: Center point
        distance: Distance from center
        num_points: Number of points in the path
        inclination: Inclination angle in degrees

    Returns:
        List of camera positions
    """
    positions = []
    for i in range(num_points):
        azimuth = 360.0 * i / num_points
        pos = create_camera_position(center, distance, azimuth, inclination)
        positions.append(pos)

    return positions


# Export functions
__all__ = [
    "create_camera_position",
    "calculate_optimal_camera_distance",
    "create_rotated_camera_position",
    "interpolate_camera_positions",
    "create_camera_path",
    "get_camera_preset",
    "calculate_camera_bounds",
    "create_orbital_camera_path",
]
