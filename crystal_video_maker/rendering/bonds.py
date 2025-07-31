"""
Chemical bond rendering functions with advanced bond analysis
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from functools import lru_cache
import plotly.graph_objects as go

from pymatgen.core import Structure, PeriodicSite
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from ..utils.colors import get_elem_colors, interpolate_colors

def calculate_bond_distances(structure: Structure, cutoff: float = 5.0) -> List[Dict[str, Any]]:
    """
    Calculate all bond distances in structure within cutoff
    
    Args:
        structure: Crystal structure
        cutoff: Maximum distance to consider for bonds
        
    Returns:
        List of bond information dictionaries
    """
    bonds = []
    sites = structure.sites
    
    for i, site1 in enumerate(sites):
        for j, site2 in enumerate(sites[i+1:], i+1):
            distance = site1.distance(site2)
            if distance <= cutoff:
                bonds.append({
                    'site1_index': i,
                    'site2_index': j,
                    'site1': site1,
                    'site2': site2,
                    'distance': distance,
                    'midpoint': (np.array(site1.coords) + np.array(site2.coords)) / 2,
                    'vector': np.array(site2.coords) - np.array(site1.coords)
                })
    
    return bonds

def get_bond_midpoint(site1: PeriodicSite, site2: PeriodicSite) -> np.ndarray:
    """
    Calculate midpoint between two sites
    
    Args:
        site1: First site
        site2: Second site
        
    Returns:
        Midpoint coordinates
    """
    return (np.array(site1.coords) + np.array(site2.coords)) / 2

def get_bond_color(
    site1: PeriodicSite, 
    site2: PeriodicSite, 
    elem_colors: Dict[str, str],
    bond_coloring: str = "element_average"
) -> str:
    """
    Get color for a bond based on bonded elements
    
    Args:
        site1: First bonded site
        site2: Second bonded site
        elem_colors: Element color mapping
        bond_coloring: Coloring scheme ("element_average", "gradient", "single")
        
    Returns:
        Bond color as hex string
    """
    from ..rendering.sites import get_site_symbol
    
    symbol1 = get_site_symbol(site1)
    symbol2 = get_site_symbol(site2)
    
    color1 = elem_colors.get(symbol1, "#808080")
    color2 = elem_colors.get(symbol2, "#808080")
    
    if bond_coloring == "element_average":
        # Average the two element colors
        colors = interpolate_colors(color1, color2, 3)
        return colors[1]  # Middle color
    elif bond_coloring == "gradient":
        # Return both colors for gradient effect
        return [color1, color2]
    else:
        # Single color (use first element)
        return color1

def filter_bonds_by_distance(
    bonds: List[Dict[str, Any]], 
    min_distance: float = 0.1, 
    max_distance: float = 3.0
) -> List[Dict[str, Any]]:
    """
    Filter bonds by distance criteria
    
    Args:
        bonds: List of bond dictionaries
        min_distance: Minimum bond distance
        max_distance: Maximum bond distance
        
    Returns:
        Filtered list of bonds
    """
    return [
        bond for bond in bonds 
        if min_distance <= bond['distance'] <= max_distance
    ]

def draw_bonds(
    fig: go.Figure,
    structure: Structure,
    nn: Union[bool, NearNeighbors] = True,
    *,
    is_3d: bool = True,
    bond_kwargs: Optional[Dict[str, Any]] = None,
    scene: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    elem_colors: Optional[Dict[str, str]] = None,
    plotted_sites_coords: Optional[Set[Tuple[float, float, float]]] = None,
    bond_coloring: str = "element_average",
    show_bond_labels: bool = False,
    **kwargs
) -> None:
    """
    Draw chemical bonds between atoms
    
    Args:
        fig: Plotly figure object
        structure: Crystal structure
        nn: Nearest neighbor finder (True for CrystalNN, or custom NearNeighbors)
        is_3d: Whether this is a 3D plot
        bond_kwargs: Bond line customization options
        scene: Scene identifier for 3D plots
        row: Subplot row for 2D plots
        col: Subplot column for 2D plots
        elem_colors: Element color mapping
        plotted_sites_coords: Set of plotted site coordinates for filtering
        bond_coloring: Bond coloring scheme
        show_bond_labels: Whether to show bond length labels
        **kwargs: Additional arguments
    """
    if not nn:
        return
    
    # Set up nearest neighbor finder
    if nn is True:
        nn_finder = CrystalNN()
    else:
        nn_finder = nn
    
    # Default bond styling
    default_bond_kwargs = {
        "color": "#808080",
        "width": 2,
        "dash": "solid"
    }
    if bond_kwargs:
        default_bond_kwargs.update(bond_kwargs)
    
    # Get element colors if not provided
    if elem_colors is None:
        elem_colors = get_elem_colors("VESTA")
    
    # Collect all bonds
    all_bonds = []
    bond_lines_x, bond_lines_y, bond_lines_z = [], [], []
    bond_colors = []
    bond_labels = []
    
    for i, site in enumerate(structure.sites):
        try:
            nn_info = nn_finder.get_nn_info(structure, i)
            
            for neighbor_data in nn_info:
                neighbor_site = neighbor_data['site']
                distance = neighbor_data.get('weight', site.distance(neighbor_site))
                
                # Filter by plotted sites if specified
                if plotted_sites_coords is not None:
                    site_coord_key = tuple(np.round(site.coords, 5))
                    neighbor_coord_key = tuple(np.round(neighbor_site.coords, 5))
                    
                    if (site_coord_key not in plotted_sites_coords or 
                        neighbor_coord_key not in plotted_sites_coords):
                        continue
                
                # Get bond color
                bond_color = get_bond_color(
                    site, neighbor_site, elem_colors, bond_coloring
                )
                
                # Add bond line coordinates
                bond_lines_x.extend([site.coords[0], neighbor_site.coords[0], None])
                bond_lines_y.extend([site.coords[1], neighbor_site.coords[1], None])
                if is_3d:
                    bond_lines_z.extend([site.coords[2], neighbor_site.coords[2], None])
                
                if isinstance(bond_color, list):
                    # Gradient coloring - use average for now
                    bond_colors.append(interpolate_colors(bond_color[0], bond_color[1], 3)[1])
                else:
                    bond_colors.append(bond_color)
                
                # Add bond label if requested
                if show_bond_labels:
                    midpoint = get_bond_midpoint(site, neighbor_site)
                    bond_labels.append({
                        'coords': midpoint,
                        'text': f"{distance:.2f} Å",
                        'color': bond_color if isinstance(bond_color, str) else bond_color[0]
                    })
                
                all_bonds.append({
                    'site1': site,
                    'site2': neighbor_site,
                    'distance': distance,
                    'color': bond_color
                })
        
        except Exception as e:
            # Skip bonds that can't be calculated
            continue
    
    if not bond_lines_x:
        return
    
    # Create bond trace
    trace_kwargs = {
        "mode": "lines",
        "line": dict(
            color=default_bond_kwargs["color"],
            width=default_bond_kwargs["width"],
            dash=default_bond_kwargs.get("dash", "solid")
        ),
        "showlegend": False,
        "hoverinfo": "skip",
        "name": "bonds"
    }
    
    if is_3d:
        trace_kwargs.update({
            "x": bond_lines_x,
            "y": bond_lines_y,
            "z": bond_lines_z
        })
        if scene:
            trace_kwargs["scene"] = scene
        fig.add_scatter3d(**trace_kwargs)
    else:
        trace_kwargs.update({
            "x": bond_lines_x,
            "y": bond_lines_y
        })
        if row and col:
            fig.add_scatter(row=row, col=col, **trace_kwargs)
        else:
            fig.add_scatter(**trace_kwargs)
    
    # Add bond labels if requested
    if show_bond_labels and bond_labels:
        _add_bond_labels(fig, bond_labels, is_3d, scene, row, col)

def _add_bond_labels(
    fig: go.Figure,
    bond_labels: List[Dict[str, Any]],
    is_3d: bool,
    scene: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None
) -> None:
    """
    Add bond length labels to the plot
    
    Args:
        fig: Plotly figure object
        bond_labels: List of bond label dictionaries
        is_3d: Whether this is a 3D plot
        scene: Scene identifier
        row: Subplot row
        col: Subplot column
    """
    coords = np.array([label['coords'] for label in bond_labels])
    texts = [label['text'] for label in bond_labels]
    colors = [label['color'] for label in bond_labels]
    
    trace_kwargs = {
        "mode": "text",
        "text": texts,
        "textfont": dict(size=8, color="black"),
        "textposition": "middle center",
        "showlegend": False,
        "hoverinfo": "skip",
        "name": "bond_labels"
    }
    
    if is_3d:
        trace_kwargs.update({
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2]
        })
        if scene:
            trace_kwargs["scene"] = scene
        fig.add_scatter3d(**trace_kwargs)
    else:
        trace_kwargs.update({
            "x": coords[:, 0],
            "y": coords[:, 1]
        })
        if row and col:
            fig.add_scatter(row=row, col=col, **trace_kwargs)
        else:
            fig.add_scatter(**trace_kwargs)

def batch_draw_bonds(
    fig: go.Figure,
    bonds: List[Dict[str, Any]],
    elem_colors: Dict[str, str],
    is_3d: bool = True,
    **kwargs
) -> None:
    """
    Draw multiple bonds efficiently in batch
    
    Args:
        fig: Plotly figure object
        bonds: List of bond dictionaries
        elem_colors: Element color mapping
        is_3d: Whether this is a 3D plot
        **kwargs: Additional arguments
    """
    if not bonds:
        return
    
    # Group bonds by color for efficient rendering
    bonds_by_color = {}
    for bond in bonds:
        color = get_bond_color(
            bond['site1'], bond['site2'], elem_colors,
            kwargs.get('bond_coloring', 'element_average')
        )
        if isinstance(color, list):
            color = color[0]  # Use first color for grouping
        
        if color not in bonds_by_color:
            bonds_by_color[color] = []
        bonds_by_color[color].append(bond)
    
    # Draw each color group
    for color, color_bonds in bonds_by_color.items():
        x_coords, y_coords, z_coords = [], [], []
        
        for bond in color_bonds:
            site1_coords = bond['site1'].coords
            site2_coords = bond['site2'].coords
            
            x_coords.extend([site1_coords[0], site2_coords[0], None])
            y_coords.extend([site1_coords[1], site2_coords[1], None])
            if is_3d:
                z_coords.extend([site1_coords[2], site2_coords[2], None])
        
        trace_kwargs = {
            "mode": "lines",
            "line": dict(
                color=color,
                width=kwargs.get('line_width', 2)
            ),
            "showlegend": False,
            "hoverinfo": "skip",
            "name": f"bonds_{color}"
        }
        
        if is_3d:
            trace_kwargs.update({
                "x": x_coords, "y": y_coords, "z": z_coords
            })
            if kwargs.get('scene'):
                trace_kwargs["scene"] = kwargs['scene']
            fig.add_scatter3d(**trace_kwargs)
        else:
            trace_kwargs.update({
                "x": x_coords, "y": y_coords
            })
            fig.add_scatter(**trace_kwargs)

def analyze_bond_statistics(bonds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze statistical properties of bonds
    
    Args:
        bonds: List of bond dictionaries
        
    Returns:
        Dictionary with bond statistics
    """
    from ..rendering.sites import get_site_symbol
    
    if not bonds:
        return {}
    
    distances = [bond['distance'] for bond in bonds]
    
    # Group by bond types
    bond_types = {}
    for bond in bonds:
        symbol1 = get_site_symbol(bond['site1'])
        symbol2 = get_site_symbol(bond['site2'])
        bond_type = f"{min(symbol1, symbol2)}-{max(symbol1, symbol2)}"
        
        if bond_type not in bond_types:
            bond_types[bond_type] = []
        bond_types[bond_type].append(bond['distance'])
    
    # Calculate statistics for each bond type
    bond_type_stats = {}
    for bond_type, type_distances in bond_types.items():
        bond_type_stats[bond_type] = {
            'count': len(type_distances),
            'min_distance': min(type_distances),
            'max_distance': max(type_distances),
            'mean_distance': np.mean(type_distances),
            'std_distance': np.std(type_distances) if len(type_distances) > 1 else 0
        }
    
    return {
        'total_bonds': len(bonds),
        'distance_range': (min(distances), max(distances)),
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'bond_type_statistics': bond_type_stats,
        'unique_bond_types': list(bond_types.keys())
    }

def create_bond_distance_histogram(bonds: List[Dict[str, Any]]) -> go.Figure:
    """
    Create histogram of bond distances
    
    Args:
        bonds: List of bond dictionaries
        
    Returns:
        Plotly figure with histogram
    """
    distances = [bond['distance'] for bond in bonds]
    
    fig = go.Figure()
    fig.add_histogram(
        x=distances,
        nbinsx=20,
        name="Bond Distances",
        opacity=0.7
    )
    
    fig.update_layout(
        title="Distribution of Bond Distances",
        xaxis_title="Distance (Å)",
        yaxis_title="Count",
        showlegend=False
    )
    
    return fig
