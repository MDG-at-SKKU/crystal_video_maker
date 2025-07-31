"""
Label generation and formatting utilities
"""

from typing import Dict, List, Optional, Union, Callable
from functools import lru_cache
import numpy as np

from pymatgen.core import PeriodicSite, Structure
from ..common_enum import SiteCoords

@lru_cache(maxsize=500)
def generate_site_label(
    site: PeriodicSite,
    label_type: str = "symbol",
    include_index: bool = False,
    site_index: Optional[int] = None
) -> str:
    """
    Generate label for an atomic site
    
    Args:
        site: PeriodicSite object
        label_type: Type of label (symbol, species, oxidation)
        include_index: Whether to include site index
        site_index: Site index number
        
    Returns:
        Formatted site label
    """
    from ..rendering.sites import get_site_symbol
    
    if label_type == "symbol":
        label = get_site_symbol(site)
    elif label_type == "species":
        label = str(site.species)
    elif label_type == "oxidation":
        symbol = get_site_symbol(site)
        if hasattr(site.species, 'oxi_state') and site.species.oxi_state:
            label = f"{symbol}{site.species.oxi_state:+}"
        else:
            label = symbol
    else:
        label = get_site_symbol(site)
    
    if include_index and site_index is not None:
        label = f"{label}_{site_index}"
    
    return label

def get_site_hover_text(
    site: PeriodicSite,
    coord_format: SiteCoords = SiteCoords.cartesian_fractional,
    float_fmt: str = ".4",
    include_properties: bool = True
) -> str:
    """
    Generate hover text for a site
    
    Args:
        site: PeriodicSite object
        coord_format: Coordinate format for display
        float_fmt: Float formatting string
        include_properties: Whether to include site properties
        
    Returns:
        Formatted hover text
    """
    from ..rendering.sites import get_site_symbol
    
    symbol = get_site_symbol(site)
    parts = [f"<b>{symbol}</b>"]
    
    # Add coordinates
    coord_text = format_coordinate_text(site, coord_format, float_fmt)
    parts.append(coord_text)
    
    # Add oxidation state if available
    if hasattr(site.species, 'oxi_state') and site.species.oxi_state:
        parts.append(f"Oxidation: {site.species.oxi_state:+}")
    
    # Add properties if requested
    if include_properties and site.properties:
        for prop, value in site.properties.items():
            if prop not in ['is_image'] and isinstance(value, (int, float)):
                parts.append(f"{prop}: {value:.3f}")
    
    return "<br>".join(parts)

def format_coordinate_text(
    site: PeriodicSite,
    coord_format: SiteCoords,
    float_fmt: str = ".4"
) -> str:
    """
    Format coordinate text for display
    
    Args:
        site: PeriodicSite object
        coord_format: Coordinate format specification
        float_fmt: Float formatting string
        
    Returns:
        Formatted coordinate string
    """
    def format_coord(coord_val):
        return f"{float(coord_val):{float_fmt}}"
    
    cart_text = f"({', '.join(format_coord(c) for c in site.coords)})"
    frac_text = f"[{', '.join(format_coord(c) for c in site.frac_coords)}]"
    
    if coord_format == SiteCoords.cartesian:
        return cart_text
    elif coord_format == SiteCoords.fractional:
        return frac_text
    elif coord_format == SiteCoords.cartesian_fractional:
        return f"{cart_text} {frac_text}"
    else:
        return cart_text

def create_element_legend(
    elements: List[str],
    elem_colors: Dict[str, str],
    show_counts: bool = True,
    element_counts: Optional[Dict[str, int]] = None
) -> Dict[str, str]:
    """
    Create legend mapping for elements
    
    Args:
        elements: List of element symbols
        elem_colors: Element color mapping
        show_counts: Whether to show element counts
        element_counts: Dictionary of element counts
        
    Returns:
        Dictionary mapping elements to legend text
    """
    legend_map = {}
    
    for element in elements:
        legend_text = element
        
        if show_counts and element_counts and element in element_counts:
            count = element_counts[element]
            legend_text = f"{element} ({count})"
        
        legend_map[element] = legend_text
    
    return legend_map

def generate_structure_title(
    structure: Structure,
    include_formula: bool = True,
    include_spacegroup: bool = True,
    include_lattice: bool = False
) -> str:
    """
    Generate title for a structure
    
    Args:
        structure: Crystal structure
        include_formula: Whether to include chemical formula
        include_spacegroup: Whether to include space group
        include_lattice: Whether to include lattice parameters
        
    Returns:
        Formatted structure title
    """
    parts = []
    
    if include_formula:
        parts.append(structure.formula)
    
    if include_spacegroup:
        try:
            spg_info = structure.get_space_group_info()
            if spg_info:
                parts.append(f"({spg_info[0]}, #{spg_info[1]})")
        except:
            pass
    
    if include_lattice:
        lattice = structure.lattice
        lattice_text = f"a={lattice.a:.2f}, b={lattice.b:.2f}, c={lattice.c:.2f}"
        parts.append(lattice_text)
    
    return " ".join(parts) if parts else "Crystal Structure"
