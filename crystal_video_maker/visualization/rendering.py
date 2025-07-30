"""Low-level rendering functions with vectorized operations for maximum performance"""

import numpy as np
from typing import Any, Dict, List
from functools import lru_cache
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor

@lru_cache(maxsize=1000)
def get_cached_marker_config(
    size: float, 
    color: str, 
    opacity: float = 1.0,
    line_width: float = 1.0
) -> Dict[str, Any]:
    """Get cached marker configuration for consistent styling

    Args:
        size: Marker size
        color: Marker color
        opacity: Marker opacity
        line_width: Border line width

    Returns:
        Dictionary with marker configuration
    """
    return dict(
        size=size,
        color=color, 
        opacity=opacity,
        line=dict(width=line_width, color="rgba(0,0,0,0.4)")
    )

def batch_add_scatter3d(
    fig: go.Figure,
    coordinates: np.ndarray,
    colors: List[str],
    sizes: List[float],
    names: List[str],
    scene: str,
    **kwargs
) -> None:
    """Add multiple 3D scatter points efficiently in batch

    Args:
        fig: Plotly figure to add to
        coordinates: Nx3 array of coordinates
        colors: List of colors for each point
        sizes: List of sizes for each point  
        names: List of names for each point
        scene: Scene identifier
        **kwargs: Additional scatter3d arguments
    """
    if len(coordinates) == 0:
        return

    # Use vectorized operations for better performance
    x_coords, y_coords, z_coords = coordinates.T

    for i, (x, y, z, color, size, name) in enumerate(
        zip(x_coords, y_coords, z_coords, colors, sizes, names)
    ):
        marker_config = get_cached_marker_config(size, color)

        fig.add_scatter3d(
            x=[x], y=[y], z=[z],
            mode="markers",
            marker=marker_config,
            name=name,
            scene=scene,
            **kwargs
        )

@lru_cache(maxsize=100)
def get_optimized_line_config(
    color: str = "black",
    width: float = 2.0,
    dash: str = "solid"
) -> Dict[str, Any]:
    """Get cached line configuration for bonds and edges

    Args:
        color: Line color
        width: Line width
        dash: Line dash style

    Returns:
        Dictionary with line configuration
    """
    return dict(color=color, width=width, dash=dash)

def vectorized_distance_calculation(
    coords1: np.ndarray,
    coords2: np.ndarray
) -> np.ndarray:
    """Calculate distances between coordinate arrays efficiently

    Args:
        coords1: First set of coordinates
        coords2: Second set of coordinates

    Returns:
        Array of distances
    """
    return np.linalg.norm(coords1 - coords2, axis=1)
