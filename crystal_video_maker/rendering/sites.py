"""
Atomic site rendering functions with advanced features
"""

import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Union, Set, Tuple
from functools import lru_cache
import plotly.graph_objects as go

from pymatgen.core import PeriodicSite, Structure
from pymatgen.core.periodic_table import Element
from ..constants import ELEMENT_RADIUS
from ..common_enum import SiteCoords
from ..utils.colors import get_elem_colors, normalize_color
from collections import defaultdict


def _make_unit_sphere(u_res: int = 8, v_res: int = 12):
    """
    Create a unit sphere vertex grid and faces (triangulated) on parameter grid.
    Returns (verts: (N,3), faces: (M,3))
    """
    u = np.linspace(0, np.pi, u_res)
    v = np.linspace(0, 2 * np.pi, v_res, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    x = (np.sin(uu) * np.cos(vv)).ravel()
    y = (np.sin(uu) * np.sin(vv)).ravel()
    z = (np.cos(uu)).ravel()
    verts = np.vstack([x, y, z]).T

    faces = []
    for vi in range(v_res):
        for ui in range(u_res - 1):
            a = vi * u_res + ui
            b = vi * u_res + (ui + 1)
            c = ((vi + 1) % v_res) * u_res + ui
            d = ((vi + 1) % v_res) * u_res + (ui + 1)
            faces.append([a, b, c])
            faces.append([b, d, c])
    faces = np.array(faces, dtype=int)
    return verts, faces


def build_mesh3d_for_group(
    group_sites: Sequence[Tuple[PeriodicSite, np.ndarray]],
    atomic_radii: Dict[str, float],
    elem_color: str,
    atom_size: float = 20,
    scale: float = 1.0,
    u_res: int = 8,
    v_res: int = 12,
    scene: str | None = None,
    name: str | None = None,
    showlegend: bool = True,
):
    """
    Build a single Mesh3d trace representing multiple spheres of same color.

    group_sites: list of (PeriodicSite, coords)
    Returns: plotly.graph_objects.Mesh3d
    """
    if not group_sites:
        return None

    verts_unit, faces = _make_unit_sphere(u_res=u_res, v_res=v_res)

    xs = []
    ys = []
    zs = []
    i_idx = []
    j_idx = []
    k_idx = []
    vert_offset = 0

    for site, coords in group_sites:
        sym = get_site_symbol(site)
        r = atomic_radii.get(sym, 0.8)
        scale_factor = r * scale * (atom_size / 20.0)
        transformed = verts_unit * scale_factor + np.array(coords)
        n_verts = transformed.shape[0]
        xs.extend(transformed[:, 0].tolist())
        ys.extend(transformed[:, 1].tolist())
        zs.extend(transformed[:, 2].tolist())
        # faces
        for f in faces:
            i_idx.append(int(f[0] + vert_offset))
            j_idx.append(int(f[1] + vert_offset))
            k_idx.append(int(f[2] + vert_offset))
        vert_offset += n_verts

    mesh = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color=elem_color,
        opacity=1.0,
        flatshading=True,
        name=name,
        showlegend=showlegend,
    )
    # Disable hover for mesh traces to avoid inconsistent hover behavior
    try:
        mesh.update(hoverinfo='skip', hovertemplate=None)
    except Exception:
        # Some plotly versions may not support hoverinfo on Mesh3d; ignore
        pass
    if scene:
        mesh.update(scene=scene)
    return mesh

@lru_cache(maxsize=1000)
def get_site_symbol(site: PeriodicSite) -> str:
    """
    Get element symbol from a site, handling disordered sites
    
    Args:
        site: PeriodicSite object
        
    Returns:
        Element symbol string
    """
    if hasattr(site.species, 'elements'):  # Disordered site (Composition)
        el_amt_dict = site.species.get_el_amt_dict()
        if el_amt_dict:
            return max(el_amt_dict, key=el_amt_dict.get)  # Most abundant element
        if site.species.elements:
            return site.species.elements[0].symbol
        return "X"
    
    if hasattr(site.species, 'symbol'):  # Element object
        return site.species.symbol
    
    if hasattr(site.species, 'element'):  # Specie object (e.g., Fe2+)
        return site.species.element.symbol
    
    # Fallback
    try:
        return site.species_string.split()[0]  # Remove oxidation state
    except (AttributeError, IndexError):
        return "X"

def get_site_coordinates(site: PeriodicSite, coord_type: str = "cartesian") -> np.ndarray:
    """
    Get site coordinates in specified format
    
    Args:
        site: PeriodicSite object
        coord_type: Type of coordinates ("cartesian" or "fractional")
        
    Returns:
        Coordinate array
    """
    if coord_type == "fractional":
        return np.array(site.frac_coords)
    else:
        return np.array(site.coords)

def get_site_marker_properties(
    site: PeriodicSite,
    atomic_radii: Dict[str, float],
    elem_colors: Dict[str, str],
    atom_size: float = 20,
    scale: float = 1.0,
    is_image: bool = False
) -> Dict[str, Any]:
    """
    Get marker properties for a site
    
    Args:
        site: PeriodicSite object
        atomic_radii: Dictionary of atomic radii
        elem_colors: Dictionary of element colors
        atom_size: Base atom size
        scale: Scaling factor
        is_image: Whether this is an image site
        
    Returns:
        Dictionary with marker properties
    """
    symbol = get_site_symbol(site)
    
    # Calculate size
    radius = atomic_radii.get(symbol, 0.8)
    size = radius * scale * atom_size
    
    # Get color
    color = elem_colors.get(symbol, "#808080")
    
    # Adjust opacity for image sites
    opacity = 0.6 if is_image else 1.0
    
    return {
        "size": size,
        "color": color,
        "opacity": opacity,
        "line": dict(width=1, color="rgba(0,0,0,0.4)")
    }

def format_site_hover_text(
    site: PeriodicSite,
    hover_format: Union[SiteCoords, callable],
    float_fmt: Union[str, callable] = ".4"
) -> str:
    """
    Format hover text for a site
    
    Args:
        site: PeriodicSite object
        hover_format: Hover text format specification
        float_fmt: Float formatting
        
    Returns:
        Formatted hover text string
    """
    symbol = get_site_symbol(site)
    
    if callable(hover_format):
        return hover_format(site)
    
    def format_coord(coord_val):
        if callable(float_fmt):
            return float_fmt(coord_val)
        return f"{float(coord_val):{float_fmt}}"
    
    cart_text = f"({', '.join(format_coord(c) for c in site.coords)})"
    frac_text = f"[{', '.join(format_coord(c) for c in site.frac_coords)}]"
    
    if hover_format == SiteCoords.cartesian:
        coords_str = cart_text
    elif hover_format == SiteCoords.fractional:
        coords_str = frac_text
    elif hover_format == SiteCoords.cartesian_fractional:
        coords_str = f"{cart_text} {frac_text}"
    else:
        coords_str = cart_text
    
    # Add additional site information
    info_parts = [f"<b>{symbol}</b>", coords_str]
    
    # Add oxidation state if available
    if hasattr(site.species, 'oxi_state') and site.species.oxi_state:
        info_parts.insert(1, f"Oxidation: {site.species.oxi_state:+}")
    
    # Add site properties if available
    if site.properties:
        for prop, value in site.properties.items():
            if prop not in ['is_image'] and isinstance(value, (int, float)):
                info_parts.append(f"{prop}: {value:.3f}")
    
    return "<br>".join(info_parts)

def draw_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: np.ndarray,
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
    hover_text: Union[SiteCoords, callable] = SiteCoords.cartesian_fractional,
    float_fmt: Union[str, callable] = ".4",
    legendgroup: Optional[str] = None,
    showlegend: bool = False,
    legend: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs
) -> None:
    """
    Draw a single atomic site on the plot
    
    Args:
        fig: Plotly figure object
        site: PeriodicSite object to draw
        coords: Site coordinates
        site_idx: Site index
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
        hover_text: Hover text format
        float_fmt: Float formatting
        legendgroup: Legend group identifier
        showlegend: Whether to show in legend
        legend: Legend identifier
        name: Trace name
        **kwargs: Additional trace arguments
    """
    symbol = get_site_symbol(site)
    
    # Get marker properties
    marker_props = get_site_marker_properties(
        site, atomic_radii, elem_colors, atom_size, scale, is_image
    )
    
    # Override with site_kwargs
    marker_props.update(site_kwargs.get('marker', {}))
    
    # Format hover text
    hover_txt = format_site_hover_text(site, hover_text, float_fmt)
    
    # Determine text display
    text_display = None
    if site_labels == "symbol":
        text_display = symbol
    elif site_labels == "species":
        text_display = str(site.species)
    elif isinstance(site_labels, dict):
        text_display = site_labels.get(site_idx, None)
    elif isinstance(site_labels, (list, tuple)):
        text_display = site_labels[site_idx] if site_idx < len(site_labels) else None
    
    # Create trace
    trace_kwargs = {
        "mode": "markers",
        "marker": marker_props,
        "hovertext": hover_txt,
        "hoverinfo": "text",
        "name": name or symbol,
        "showlegend": showlegend,
        **kwargs
    }
    
    if legendgroup:
        trace_kwargs["legendgroup"] = legendgroup
    
    if legend:
        trace_kwargs["legend"] = legend
    
    if text_display:
        trace_kwargs["text"] = [text_display]
        trace_kwargs["textposition"] = "middle center"
        trace_kwargs["textfont"] = dict(size=10, color="white")
    
    if is_3d:
        # 3D scatter plot
        trace_kwargs.update({
            "x": [coords[0]], "y": [coords[1]], "z": [coords[2]]
        })
        if scene:
            trace_kwargs["scene"] = scene
        fig.add_scatter3d(**trace_kwargs)
    else:
        # 2D scatter plot
        trace_kwargs.update({
            "x": [coords[0]], "y": [coords[1]]
        })
        if row and col:
            fig.add_scatter(row=row, col=col, **trace_kwargs)
        else:
            fig.add_scatter(**trace_kwargs)

def draw_disordered_site(
    fig: go.Figure,
    site: PeriodicSite,
    coords: np.ndarray,
    site_idx: int,
    elem_colors: Dict[str, str],
    atomic_radii: Dict[str, float],
    atom_size: float,
    scale: float,
    **kwargs
) -> None:
    """
    Draw a disordered site with multiple species representations
    
    Args:
        fig: Plotly figure object
        site: Disordered PeriodicSite object
        coords: Site coordinates
        site_idx: Site index
        elem_colors: Element color mapping
        atomic_radii: Atomic radii mapping
        atom_size: Base atom size
        scale: Scaling factor
        **kwargs: Additional arguments
    """
    if not hasattr(site.species, 'elements') or len(site.species.elements) <= 1:
        # Not actually disordered, use regular draw_site
        draw_site(fig, site, coords, site_idx, None, elem_colors, 
                 atomic_radii, atom_size, scale, {}, **kwargs)
        return
    
    # Get element amounts
    el_amt_dict = site.species.get_el_amt_dict()
    
    # Draw pie chart representation for disordered site
    total_amount = sum(el_amt_dict.values())
    angle_offset = 0
    
    for i, (element, amount) in enumerate(el_amt_dict.items()):
        fraction = amount / total_amount
        angle_span = 2 * np.pi * fraction
        
        # Create partial circle for this element
        n_points = max(10, int(20 * fraction))
        angles = np.linspace(angle_offset, angle_offset + angle_span, n_points)
        
        # Calculate radius
        radius = atomic_radii.get(element.symbol, 0.8) * scale * atom_size / 20
        
        # Create circular arc coordinates
        x_arc = coords[0] + radius * np.cos(angles)
        y_arc = coords[1] + radius * np.sin(angles)
        
        if kwargs.get('is_3d', True):
            z_arc = [coords[2]] * len(angles)
            fig.add_scatter3d(
                x=x_arc, y=y_arc, z=z_arc,
                mode="markers",
                marker=dict(
                    size=atom_size * scale * 0.3,
                    color=elem_colors.get(element.symbol, "#808080"),
                    opacity=0.8
                ),
                name=f"{element.symbol} ({fraction:.1%})",
                showlegend=i == 0,  # Only show legend for first element
                **{k: v for k, v in kwargs.items() if k not in ['is_3d']}
            )
        else:
            fig.add_scatter(
                x=x_arc, y=y_arc,
                mode="markers",
                marker=dict(
                    size=atom_size * scale * 0.3,
                    color=elem_colors.get(element.symbol, "#808080"),
                    opacity=0.8
                ),
                name=f"{element.symbol} ({fraction:.1%})",
                showlegend=i == 0,
                **{k: v for k, v in kwargs.items() if k not in ['is_3d']}
            )
        
        angle_offset += angle_span

def batch_draw_sites(
    fig: go.Figure,
    sites: List[PeriodicSite],
    coords_list: List[np.ndarray],
    elem_colors: Dict[str, str],
    atomic_radii: Dict[str, float],
    atom_size: float = 20,
    scale: float = 1.0,
    **kwargs
) -> None:
    """
    Draw multiple sites efficiently in batch
    
    Args:
        fig: Plotly figure object
        sites: List of PeriodicSite objects
        coords_list: List of coordinate arrays
        elem_colors: Element color mapping
        atomic_radii: Atomic radii mapping
        atom_size: Base atom size
        scale: Scaling factor
        **kwargs: Additional arguments
    """
    if not sites or len(sites) != len(coords_list):
        return
    
    # Group sites by element for batch processing
    sites_by_element = {}
    for i, (site, coords) in enumerate(zip(sites, coords_list)):
        symbol = get_site_symbol(site)
        if symbol not in sites_by_element:
            sites_by_element[symbol] = {
                'sites': [], 'coords': [], 'indices': []
            }
        sites_by_element[symbol]['sites'].append(site)
        sites_by_element[symbol]['coords'].append(coords)
        sites_by_element[symbol]['indices'].append(i)
    
    # Draw each element group as a single trace
    for symbol, group_data in sites_by_element.items():
        coords_array = np.array(group_data['coords'])
        
        # Calculate properties for this element
        radius = atomic_radii.get(symbol, 0.8)
        size = radius * scale * atom_size
        color = elem_colors.get(symbol, "#808080")
        
        # Create hover texts
        hover_texts = [
            format_site_hover_text(site, kwargs.get('hover_text', SiteCoords.cartesian_fractional))
            for site in group_data['sites']
        ]
        
        trace_kwargs = {
            "mode": "markers",
            "marker": dict(
                size=size,
                color=color,
                opacity=kwargs.get('opacity', 1.0),
                line=dict(width=1, color="rgba(0,0,0,0.4)")
            ),
            "hovertext": hover_texts,
            "hoverinfo": "text",
            "name": symbol,
            "showlegend": kwargs.get('showlegend', True)
        }
        
        if kwargs.get('is_3d', True):
            trace_kwargs.update({
                "x": coords_array[:, 0],
                "y": coords_array[:, 1], 
                "z": coords_array[:, 2]
            })
            if kwargs.get('scene'):
                trace_kwargs["scene"] = kwargs['scene']
            fig.add_scatter3d(**trace_kwargs)
        else:
            trace_kwargs.update({
                "x": coords_array[:, 0],
                "y": coords_array[:, 1]
            })
            fig.add_scatter(**trace_kwargs)

def get_site_properties_summary(sites: List[PeriodicSite]) -> Dict[str, Any]:
    """
    Get summary of site properties across multiple sites
    
    Args:
        sites: List of PeriodicSite objects
        
    Returns:
        Dictionary with property summaries
    """
    elements = [get_site_symbol(site) for site in sites]
    unique_elements = list(set(elements))
    
    # Collect all properties
    all_properties = set()
    for site in sites:
        all_properties.update(site.properties.keys())
    
    property_stats = {}
    for prop in all_properties:
        values = []
        for site in sites:
            if prop in site.properties:
                val = site.properties[prop]
                if isinstance(val, (int, float)):
                    values.append(val)
        
        if values:
            property_stats[prop] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values) if len(values) > 1 else 0
            }
    
    return {
        'num_sites': len(sites),
        'unique_elements': unique_elements,
        'element_counts': {elem: elements.count(elem) for elem in unique_elements},
        'property_statistics': property_stats
    }

def filter_sites_by_criteria(
    sites: List[PeriodicSite],
    coords_list: List[np.ndarray],
    criteria: Dict[str, Any]
) -> Tuple[List[PeriodicSite], List[np.ndarray]]:
    """
    Filter sites based on various criteria
    
    Args:
        sites: List of PeriodicSite objects
        coords_list: List of coordinate arrays
        criteria: Filtering criteria dictionary
        
    Returns:
        Tuple of filtered sites and coordinates
    """
    filtered_sites = []
    filtered_coords = []
    
    for site, coords in zip(sites, coords_list):
        include_site = True
        
        # Filter by element
        if 'elements' in criteria:
            symbol = get_site_symbol(site)
            if symbol not in criteria['elements']:
                include_site = False
        
        # Filter by coordinate bounds
        if 'coord_bounds' in criteria and include_site:
            bounds = criteria['coord_bounds']
            if not all(bounds['min'][i] <= coords[i] <= bounds['max'][i] for i in range(3)):
                include_site = False
        
        # Filter by properties
        if 'properties' in criteria and include_site:
            for prop, (min_val, max_val) in criteria['properties'].items():
                if prop in site.properties:
                    val = site.properties[prop]
                    if isinstance(val, (int, float)):
                        if not (min_val <= val <= max_val):
                            include_site = False
                            break
        
        if include_site:
            filtered_sites.append(site)
            filtered_coords.append(coords)
    
    return filtered_sites, filtered_coords
