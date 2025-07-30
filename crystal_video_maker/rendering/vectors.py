"""
Vector rendering functions for forces, magnetic moments, etc.
"""

import plotly.graph_objects as go
from typing import Any, Dict, List, Optional
import numpy as np
from pymatgen.core import PeriodicSite
from functools import lru_cache

@lru_cache(maxsize=300)
def _get_cached_arrow_style(
    color: str = "red",
    width: float = 3.0
) -> Dict[str, Any]:
    """
    Get cached arrow line styling

    Args:
        color: Arrow color
        width: Arrow line width

    Returns:
        Dictionary with arrow styling
    """
    return dict(color=color, width=width)

def draw_vector(
    fig: go.Figure,
    site: PeriodicSite,
    vector: List[float],
    *,
    vector_property: str = "force",
    scale: float = 1.0,
    is_3d: bool = True,
    scene: str = "scene",
    color: str = "red",
    width: float = 3.0,
    show_magnitude: bool = False,
    row: Optional[int] = None,
    col: Optional[int] = None,
    **kwargs
) -> None:
    """
    Draw a vector (force, magnetic moment, etc.) from a site

    Args:
        fig: Plotly figure to add vector to
        site: Site to draw vector from
        vector: Vector components [x, y, z]
        vector_property: Name of the vector property
        scale: Scaling factor for vector length
        is_3d: Whether this is a 3D plot
        scene: Scene identifier for 3D plots
        color: Vector color
        width: Vector line width
        show_magnitude: Whether to show vector magnitude in hover
        row: Subplot row for 2D plots
        col: Subplot column for 2D plots
        **kwargs: Additional scatter arguments
    """
    # Skip zero or very small vectors
    vector_magnitude = np.linalg.norm(vector)
    if vector_magnitude < 1e-6:
        return

    # Scale vector
    scaled_vector = np.array(vector) * scale

    # Get start and end points
    start_point = site.coords
    end_point = start_point + scaled_vector

    # Get cached arrow styling
    line_style = _get_cached_arrow_style(color, width)

    # Prepare coordinates for the arrow shaft
    x_coords = [start_point[0], end_point[0]]
    y_coords = [start_point[1], end_point[1]]

    # Prepare hover text
    hover_text = f"{vector_property}: [{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}]"
    if show_magnitude:
        hover_text += f"<br>Magnitude: {vector_magnitude:.3f}"

    scatter_kwargs = dict(
        x=x_coords,
        y=y_coords,
        mode="lines",
        line=line_style,
        showlegend=False,
        name=f"{vector_property}_vector",
        hovertext=hover_text,
        hoverinfo="text",
        **kwargs
    )

    if is_3d:
        z_coords = [start_point[2], end_point[2]]
        scatter_kwargs['z'] = z_coords
        scatter_kwargs['scene'] = scene
        fig.add_scatter3d(**scatter_kwargs)

        # Add arrowhead for 3D (using a cone)
        _add_3d_arrowhead(fig, start_point, end_point, color, scene)
    else:
        if row is not None and col is not None:
            scatter_kwargs.update(row=row, col=col)
        fig.add_scatter(**scatter_kwargs)

        # Add arrowhead for 2D using annotation
        _add_2d_arrowhead(fig, start_point, end_point, color, row, col)

def _add_3d_arrowhead(
    fig: go.Figure,
    start_point: np.ndarray,
    end_point: np.ndarray,
    color: str,
    scene: str
) -> None:
    """
    Add 3D arrowhead using cone

    Args:
        fig: Plotly figure
        start_point: Vector start coordinates
        end_point: Vector end coordinates  
        color: Arrow color
        scene: Scene identifier
    """
    # Calculate arrow direction
    direction = end_point - start_point
    direction_norm = direction / np.linalg.norm(direction)

    # Position arrowhead slightly before the end point
    arrowhead_pos = end_point - 0.1 * direction_norm

    fig.add_cone(
        x=[arrowhead_pos[0]],
        y=[arrowhead_pos[1]], 
        z=[arrowhead_pos[2]],
        u=[direction_norm[0]],
        v=[direction_norm[1]],
        w=[direction_norm[2]],
        sizemode="absolute",
        sizeref=0.2,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        scene=scene,
        showlegend=False,
        hoverinfo="skip"
    )

def _add_2d_arrowhead(
    fig: go.Figure,
    start_point: np.ndarray,
    end_point: np.ndarray,
    color: str,
    row: Optional[int] = None,
    col: Optional[int] = None
) -> None:
    """
    Add 2D arrowhead using annotation

    Args:
        fig: Plotly figure
        start_point: Vector start coordinates
        end_point: Vector end coordinates
        color: Arrow color
        row: Subplot row
        col: Subplot column
    """
    # Add annotation with arrow
    annotation_kwargs = dict(
        x=end_point[0],
        y=end_point[1],
        ax=start_point[0],
        ay=start_point[1],
        xref="x", yref="y",
        axref="x", ayref="y",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor=color,
        showarrow=True,
        text="",
    )

    if row is not None and col is not None:
        annotation_kwargs.update(
            xref=f"x{col}" if col > 1 else "x",
            yref=f"y{row}" if row > 1 else "y",
            axref=f"x{col}" if col > 1 else "x", 
            ayref=f"y{row}" if row > 1 else "y"
        )

    fig.add_annotation(**annotation_kwargs)

def draw_site_vectors(
    fig: go.Figure,
    site: PeriodicSite,
    vector_properties: List[str],
    *,
    vector_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs
) -> None:
    """
    Draw multiple vectors from a single site

    Args:
        fig: Plotly figure
        site: Site to draw vectors from
        vector_properties: List of vector property names to draw
        vector_kwargs: Per-property vector styling options
        **kwargs: Common arguments for all vectors
    """
    vector_kwargs = vector_kwargs or {}

    # Default colors for different vector types
    default_colors = {
        'force': 'red',
        'magmom': 'blue', 
        'velocity': 'green',
        'displacement': 'orange'
    }

    for prop in vector_properties:
        if prop in site.properties:
            vector_data = site.properties[prop]
            if isinstance(vector_data, (list, tuple, np.ndarray)) and len(vector_data) == 3:
                # Get property-specific styling
                prop_kwargs = vector_kwargs.get(prop, {})
                color = prop_kwargs.get('color', default_colors.get(prop, 'red'))

                # Merge kwargs
                merged_kwargs = {**kwargs, **prop_kwargs}
                merged_kwargs['color'] = color
                merged_kwargs['vector_property'] = prop

                draw_vector(fig, site, vector_data, **merged_kwargs)

def batch_draw_vectors(
    fig: go.Figure,
    sites: List[PeriodicSite],
    vector_property: str,
    **kwargs
) -> None:
    """
    Draw vectors for multiple sites efficiently

    Args:
        fig: Plotly figure
        sites: List of sites to draw vectors for
        vector_property: Vector property name
        **kwargs: Arguments passed to draw_vector
    """
    for site in sites:
        if vector_property in site.properties:
            vector_data = site.properties[vector_property]
            if isinstance(vector_data, (list, tuple, np.ndarray)) and len(vector_data) == 3:
                draw_vector(fig, site, vector_data, vector_property=vector_property, **kwargs)
