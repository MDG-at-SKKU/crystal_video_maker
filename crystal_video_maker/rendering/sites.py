"""
Site rendering functions with performance improvements
"""

import plotly.graph_objects as go
from typing import Any, Optional, Dict, List
from pymatgen.core import PeriodicSite
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1000)
def _get_cached_site_marker(
    size: float,
    color: str,
    opacity: float = 1.0
) -> Dict[str, Any]:
    """
    Get cached marker configuration for site rendering

    Args:
        size: Marker size
        color: Marker color
        opacity: Marker opacity

    Returns:
        Dictionary with marker configuration
    """
    return dict(
        size=size,
        color=color,
        opacity=opacity,
        line=dict(width=1, color="rgba(0,0,0,0.4)")
    )

def draw_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: List[float],
    site_idx: int,
    site_labels: Any,
    elem_colors: Dict[str, str],
    atomic_radii: Dict[str, float],
    atom_size: float,
    scale: float,
    site_kwargs: Dict[str, Any],
    *,
    is_image: bool = False,
    is_3d: bool = True,
    row: Optional[int] = None,
    col: Optional[int] = None,
    scene: Optional[str] = None,
    hover_text: str = "",
    float_fmt: str = ".4",
    legendgroup: Optional[str] = None,
    showlegend: bool = False,
    legend: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Add a site (regular or image) to the plot with performance improvements

    Args:
        fig: Plotly figure to add site to
        site: Periodic site object
        coords: Site coordinates [x, y, z]  
        site_idx: Index of the site
        site_labels: Site labeling configuration
        elem_colors: Element color mapping
        atomic_radii: Atomic radii mapping
        atom_size: Base atom size
        scale: Scaling factor
        site_kwargs: Additional site styling options
        is_image: Whether this is an image site
        is_3d: Whether this is a 3D plot
        row: Subplot row (for 2D plots)
        col: Subplot column (for 2D plots) 
        scene: Scene identifier (for 3D plots)
        hover_text: Hover tooltip text
        float_fmt: Float formatting for coordinates
        legendgroup: Legend group identifier
        showlegend: Whether to show in legend
        legend: Legend identifier
        name: Site name for legend
        **kwargs: Additional scatter arguments
    """
    # Get element symbol
    if hasattr(site, 'specie'):
        element = site.specie.symbol
    elif hasattr(site, 'species_string'):
        element = site.species_string
    else:
        element = str(site.species)

    # Get color and size
    color = elem_colors.get(element, "gray")
    radius = atomic_radii.get(element, 0.8)
    marker_size = radius * scale * atom_size

    # Get cached marker configuration
    marker_config = _get_cached_site_marker(marker_size, color)

    # Prepare text
    text = None
    if site_labels == "symbol":
        text = element
    elif site_labels == "species":
        text = str(site.species)

    # Prepare hover text
    if not hover_text and hasattr(site, 'coords'):
        hover_text = f"Element: {element}<br>Coords: [{coords[0]:.3f}, {coords[1]:.3f}, {coords[2]:.3f}]"

    # Add to figure
    scatter_kwargs = dict(
        x=[coords[0]],
        y=[coords[1]], 
        z=[coords[2]] if is_3d else None,
        mode="markers+text" if text else "markers",
        marker=marker_config,
        text=[text] if text else None,
        textposition="middle center",
        hovertext=hover_text,
        hoverinfo="text",
        name=name or element,
        legendgroup=legendgroup,
        showlegend=showlegend,
        **kwargs
    )

    # Remove None values
    scatter_kwargs = {k: v for k, v in scatter_kwargs.items() if v is not None}

    if is_3d:
        scatter_kwargs['scene'] = scene or "scene"
        fig.add_scatter3d(**scatter_kwargs)
    else:
        if row is not None and col is not None:
            scatter_kwargs['row'] = row
            scatter_kwargs['col'] = col
        fig.add_scatter(**scatter_kwargs)

def draw_disordered_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: List[float],
    site_idx: int,
    **kwargs
) -> None:
    """
    Draw a disordered site with special styling

    Args:
        fig: Plotly figure to add to
        site: Disordered periodic site
        coords: Site coordinates
        site_idx: Site index
        **kwargs: Additional arguments passed to draw_site
    """
    # For disordered sites, we can add special styling
    kwargs['site_kwargs'] = kwargs.get('site_kwargs', {})
    kwargs['site_kwargs']['opacity'] = 0.7  # Make disordered sites semi-transparent

    # Draw each species in the disordered site
    for species, occupancy in site.species.items():
        if occupancy > 0.01:  # Only draw species with significant occupancy
            # Create a temporary site for this species
            temp_site = PeriodicSite(
                species, 
                site.frac_coords, 
                site.lattice,
                coords_are_cartesian=False
            )

            # Adjust marker size by occupancy
            original_atom_size = kwargs.get('atom_size', 20)
            kwargs['atom_size'] = original_atom_size * occupancy

            draw_site(fig, temp_site, coords, site_idx, **kwargs)

def batch_draw_sites(
    fig: go.Figure,
    sites_data: List[Dict[str, Any]]
) -> None:
    """
    Draw multiple sites efficiently in batch

    Args:
        fig: Plotly figure to add sites to
        sites_data: List of dictionaries with site drawing parameters
    """
    # Group sites by element for more efficient rendering
    sites_by_element = {}

    for site_data in sites_data:
        site = site_data['site']
        element = getattr(site, 'species_string', str(site.species))

        if element not in sites_by_element:
            sites_by_element[element] = []

        sites_by_element[element].append(site_data)

    # Draw sites grouped by element
    for element, element_sites in sites_by_element.items():
        for site_data in element_sites:
            draw_site(fig, **site_data)
