"""
Bond rendering functions with performance improvements
"""

import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Set
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import NearNeighbors, CrystalNN
from functools import lru_cache

@lru_cache(maxsize=500)
def _get_cached_bond_style(
    color: str = "black",
    width: float = 2.0,
    dash: str = "solid"
) -> Dict[str, Any]:
    """
    Get cached bond line styling

    Args:
        color: Bond line color
        width: Bond line width  
        dash: Line dash style

    Returns:
        Dictionary with line styling configuration
    """
    return dict(color=color, width=width, dash=dash)

def draw_bonds(
    fig: go.Figure,
    structure: Structure,
    nn: NearNeighbors | bool,
    *,
    is_3d: bool = True,
    bond_kwargs: Optional[Dict[str, Any]] = None,
    scene: str = "scene",
    elem_colors: Optional[Dict[str, str]] = None,
    plotted_sites_coords: Optional[Set] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> None:
    """
    Draw bonds between sites with performance improvements

    Args:
        fig: Plotly figure to add bonds to
        structure: Crystal structure
        nn: Neighbor finding algorithm or True for CrystalNN
        is_3d: Whether this is a 3D plot
        bond_kwargs: Bond styling options
        scene: Scene identifier for 3D plots
        elem_colors: Element color mapping
        plotted_sites_coords: Set of plotted site coordinates for filtering
        row: Subplot row for 2D plots
        col: Subplot column for 2D plots
    """
    if nn is True:
        nn = CrystalNN()

    bond_kwargs = bond_kwargs or {}
    bond_color = bond_kwargs.get('color', 'black')
    bond_width = bond_kwargs.get('width', 2.0)

    # Get cached line style
    line_style = _get_cached_bond_style(bond_color, bond_width)

    # Collect all bond coordinates for batch rendering
    bond_coords = {'x': [], 'y': [], 'z': []}

    sites = structure.sites
    n_sites = len(sites)

    # Pre-compute site coordinates if filtering is needed
    if plotted_sites_coords is not None:
        plotted_coords_array = np.array([
            list(coord_tuple) for coord_tuple in plotted_sites_coords
        ])

    # Find bonds efficiently
    for i, site in enumerate(sites):
        try:
            # Get neighbors for this site
            neighbors_info = nn.get_nn_info(structure, i)

            for neighbor_info in neighbors_info:
                neighbor_site = neighbor_info['site']

                # Filter bonds if plotted sites are specified
                if plotted_sites_coords is not None:
                    site_coord_key = tuple(np.round(site.coords, 5))
                    neighbor_coord_key = tuple(np.round(neighbor_site.coords, 5))

                    if (site_coord_key not in plotted_sites_coords or 
                        neighbor_coord_key not in plotted_sites_coords):
                        continue

                # Add bond coordinates
                bond_coords['x'].extend([site.coords[0], neighbor_site.coords[0], None])
                bond_coords['y'].extend([site.coords[1], neighbor_site.coords[1], None])
                bond_coords['z'].extend([site.coords[2], neighbor_site.coords[2], None])

        except Exception as e:
            # Skip sites that cause errors in neighbor finding
            continue

    # Add all bonds in a single trace for better performance
    if bond_coords['x']:  # Only add if we have bonds to draw
        scatter_kwargs = dict(
            x=bond_coords['x'],
            y=bond_coords['y'],
            mode="lines",
            line=line_style,
            showlegend=False,
            name="bonds",
            hoverinfo="skip"
        )

        if is_3d:
            scatter_kwargs['z'] = bond_coords['z']
            scatter_kwargs['scene'] = scene
            fig.add_scatter3d(**scatter_kwargs)
        else:
            if row is not None and col is not None:
                scatter_kwargs.update(row=row, col=col)
            fig.add_scatter(**scatter_kwargs)

def draw_bond_between_sites(
    fig: go.Figure,
    site1_coords: List[float],
    site2_coords: List[float], 
    *,
    is_3d: bool = True,
    color: str = "black",
    width: float = 2.0,
    scene: str = "scene",
    **kwargs
) -> None:
    """
    Draw a single bond between two specific sites

    Args:
        fig: Plotly figure to add bond to
        site1_coords: First site coordinates [x, y, z]
        site2_coords: Second site coordinates [x, y, z]
        is_3d: Whether this is a 3D plot
        color: Bond color
        width: Bond width
        scene: Scene identifier
        **kwargs: Additional scatter arguments
    """
    line_style = _get_cached_bond_style(color, width)

    x_coords = [site1_coords[0], site2_coords[0]]
    y_coords = [site1_coords[1], site2_coords[1]]

    scatter_kwargs = dict(
        x=x_coords,
        y=y_coords,
        mode="lines",
        line=line_style,
        showlegend=False,
        hoverinfo="skip",
        **kwargs
    )

    if is_3d:
        z_coords = [site1_coords[2], site2_coords[2]]
        scatter_kwargs['z'] = z_coords
        scatter_kwargs['scene'] = scene
        fig.add_scatter3d(**scatter_kwargs)
    else:
        fig.add_scatter(**scatter_kwargs)

def batch_draw_bonds(
    fig: go.Figure,
    bond_pairs: List[tuple],
    *,
    is_3d: bool = True,
    colors: Optional[List[str]] = None,
    scene: str = "scene"
) -> None:
    """
    Draw multiple bonds efficiently in batch

    Args:
        fig: Plotly figure to add bonds to
        bond_pairs: List of (site1_coords, site2_coords) tuples
        is_3d: Whether this is a 3D plot
        colors: Optional list of colors for each bond
        scene: Scene identifier
    """
    if not bond_pairs:
        return

    # Group bonds by color for efficient rendering
    if colors is None:
        colors = ["black"] * len(bond_pairs)

    bonds_by_color = {}
    for (site1, site2), color in zip(bond_pairs, colors):
        if color not in bonds_by_color:
            bonds_by_color[color] = []
        bonds_by_color[color].append((site1, site2))

    # Draw bonds grouped by color
    for color, color_bonds in bonds_by_color.items():
        bond_coords = {'x': [], 'y': [], 'z': []}

        for site1, site2 in color_bonds:
            bond_coords['x'].extend([site1[0], site2[0], None])
            bond_coords['y'].extend([site1[1], site2[1], None])
            if is_3d:
                bond_coords['z'].extend([site1[2], site2[2], None])

        line_style = _get_cached_bond_style(color)

        scatter_kwargs = dict(
            x=bond_coords['x'],
            y=bond_coords['y'],
            mode="lines",
            line=line_style,
            showlegend=False,
            hoverinfo="skip"
        )

        if is_3d:
            scatter_kwargs['z'] = bond_coords['z']
            scatter_kwargs['scene'] = scene
            fig.add_scatter3d(**scatter_kwargs)
        else:
            fig.add_scatter(**scatter_kwargs)
