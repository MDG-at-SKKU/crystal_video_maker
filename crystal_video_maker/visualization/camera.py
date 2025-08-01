"""
Camera positioning and view management for 3D crystal structure visualization
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Sequence
from ..common_types import AnyStructure


def calculate_optimal_camera_position(
    structures: dict,
    margin_factor: float = 1.2,
    default_distance: float = 2.0,
    view_angle: str = "isometric"
) -> Dict[str, float]:
    """
    Calculate optimal camera position to view all atoms in the structures
    
    Args:
        structures: Dictionary of structures to analyze
        margin_factor: Factor to add margin around the bounding box
        default_distance: Default camera distance if calculation fails
        view_angle: Camera angle preset ('isometric', 'front', 'top', 'side')
        
    Returns:
        Dictionary with optimal camera eye position {x, y, z}
    """
    try:
        all_coords = []
        
        # Collect all atomic coordinates from all structures
        for struct in structures.values():
            for site in struct.sites:
                all_coords.append(site.coords)
        
        if not all_coords:
            # Fallback to default position
            return _get_default_camera_position(view_angle, default_distance)
        
        coords_array = np.array(all_coords)
        
        # Calculate bounding box
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        center = (min_coords + max_coords) / 2
        
        # Calculate the extent of the structure
        extent = max_coords - min_coords
        max_extent = np.max(extent)
        
        # Calculate optimal camera distance
        # Use margin factor to ensure all atoms are visible
        camera_distance = max_extent * margin_factor
        
        # If structure is too small, use minimum distance
        camera_distance = max(camera_distance, default_distance)
        
        # Get camera direction based on view angle
        camera_direction = _get_camera_direction(view_angle)
        
        # Calculate camera position relative to structure center
        camera_position = center + camera_direction * camera_distance
        
        return dict(x=camera_position[0], y=camera_position[1], z=camera_position[2])
        
    except Exception as e:
        # Fallback to default position if calculation fails
        print(f"Warning: Camera position calculation failed: {e}")
        return _get_default_camera_position(view_angle, default_distance)


def calculate_scene_center(structures: dict) -> Dict[str, float]:
    """
    Calculate the center point of all structures for camera targeting
    
    Args:
        structures: Dictionary of structures to analyze
        
    Returns:
        Dictionary with center coordinates {x, y, z}
    """
    try:
        all_coords = []
        
        for struct in structures.values():
            for site in struct.sites:
                all_coords.append(site.coords)
        
        if not all_coords:
            return dict(x=0, y=0, z=0)
        
        coords_array = np.array(all_coords)
        center = np.mean(coords_array, axis=0)
        
        return dict(x=center[0], y=center[1], z=center[2])
        
    except Exception:
        return dict(x=0, y=0, z=0)


def _get_camera_direction(view_angle: str) -> np.ndarray:
    """
    Get normalized camera direction vector based on view angle preset
    
    Args:
        view_angle: Camera angle preset
        
    Returns:
        Normalized direction vector
    """
    directions = {
        "isometric": np.array([1, 1, 1]),
        "front": np.array([0, -1, 0]),
        "back": np.array([0, 1, 0]),
        "top": np.array([0, 0, 1]),
        "bottom": np.array([0, 0, -1]),
        "left": np.array([-1, 0, 0]),
        "right": np.array([1, 0, 0]),
        "side": np.array([1, 0, 0])
    }
    
    direction = directions.get(view_angle, directions["isometric"])
    return direction / np.linalg.norm(direction)


def _get_default_camera_position(view_angle: str, distance: float) -> Dict[str, float]:
    """
    Get default camera position for fallback cases
    
    Args:
        view_angle: Camera angle preset
        distance: Camera distance from origin
        
    Returns:
        Dictionary with camera position {x, y, z}
    """
    direction = _get_camera_direction(view_angle)
    position = direction * distance
    
    return dict(x=position[0], y=position[1], z=position[2])


def create_camera_config(
    eye: Optional[Dict[str, float]] = None,
    center: Optional[Dict[str, float]] = None,
    up: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Create a complete camera configuration dictionary
    
    Args:
        eye: Camera position {x, y, z}
        center: Point camera looks at {x, y, z}
        up: Up vector {x, y, z}
        
    Returns:
        Complete camera configuration dictionary
    """
    config = {}
    
    if eye is not None:
        config['eye'] = eye.copy()
    
    if center is not None:
        config['center'] = center.copy()
    
    if up is not None:
        config['up'] = up.copy()
    else:
        config['up'] = dict(x=0, y=0, z=1)  # Default: Z-axis up
    
    return config


def get_preset_camera_configs() -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Get dictionary of preset camera configurations
    
    Returns:
        Dictionary of named camera presets
    """
    return {
        "isometric": create_camera_config(
            eye=dict(x=1.5, y=1.5, z=1.5),
            center=dict(x=0, y=0, z=0)
        ),
        "front": create_camera_config(
            eye=dict(x=0, y=-3, z=0),
            center=dict(x=0, y=0, z=0)
        ),
        "back": create_camera_config(
            eye=dict(x=0, y=3, z=0),
            center=dict(x=0, y=0, z=0)
        ),
        "top": create_camera_config(
            eye=dict(x=0, y=0, z=3),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)  # Y-axis up for top view
        ),
        "bottom": create_camera_config(
            eye=dict(x=0, y=0, z=-3),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=1, z=0)
        ),
        "left": create_camera_config(
            eye=dict(x=-3, y=0, z=0),
            center=dict(x=0, y=0, z=0)
        ),
        "right": create_camera_config(
            eye=dict(x=3, y=0, z=0),
            center=dict(x=0, y=0, z=0)
        )
    }


def get_camera_preset(preset_name: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Get a specific camera preset configuration
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Camera configuration dictionary or None if preset not found
    """
    presets = get_preset_camera_configs()
    return presets.get(preset_name)


def optimize_camera_for_structures(
    structures: dict,
    preset: str = "isometric",
    margin_factor: float = 1.2,
    auto_center: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Create optimized camera configuration for given structures
    
    Args:
        structures: Dictionary of structures to optimize for
        preset: Base camera angle preset
        margin_factor: Margin factor for camera distance
        auto_center: Whether to automatically center on structures
        
    Returns:
        Optimized camera configuration dictionary
    """
    # Get base preset
    base_config = get_camera_preset(preset)
    if base_config is None:
        base_config = get_camera_preset("isometric")
    
    # Calculate optimal camera position
    optimal_eye = calculate_optimal_camera_position(
        structures, 
        margin_factor=margin_factor,
        view_angle=preset
    )
    
    # Calculate scene center if auto_center is enabled
    if auto_center:
        optimal_center = calculate_scene_center(structures)
    else:
        optimal_center = base_config.get('center', dict(x=0, y=0, z=0))
    
    # Create optimized configuration
    return create_camera_config(
        eye=optimal_eye,
        center=optimal_center,
        up=base_config.get('up', dict(x=0, y=0, z=1))
    )
