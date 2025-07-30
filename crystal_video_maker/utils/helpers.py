"""
Helper functions with performance improvements
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pymatgen.core import PeriodicSite

@lru_cache(maxsize=1000)
def get_site_symbol(site: PeriodicSite) -> str:
    """
    Get element symbol from site with caching

    Args:
        site: Periodic site object

    Returns:
        Element symbol
    """
    if hasattr(site, 'specie'):
        return site.specie.symbol
    elif hasattr(site, 'species_string'):
        return site.species_string.split()[0]  # Handle cases like "Fe2+"
    else:
        species_str = str(site.species)
        # Extract element symbol from species string
        import re
        match = re.match(r'([A-Z][a-z]?)', species_str)
        return match.group(1) if match else species_str

def get_first_matching_site_prop(
    site: PeriodicSite,
    property_names: List[str]
) -> Any:
    """
    Get first available property from a list

    Args:
        site: Periodic site object
        property_names: List of property names to check

    Returns:
        First available property value or None
    """
    if not hasattr(site, 'properties'):
        return None

    for prop_name in property_names:
        if prop_name in site.properties:
            return site.properties[prop_name]

    return None

@lru_cache(maxsize=500)
def calculate_distance_3d(
    coord1: tuple,
    coord2: tuple
) -> float:
    """
    Calculate 3D distance between two points with caching

    Args:
        coord1: First coordinate tuple (x, y, z)
        coord2: Second coordinate tuple (x, y, z)

    Returns:
        Euclidean distance
    """
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    dz = coord1[2] - coord2[2]
    return float(np.sqrt(dx*dx + dy*dy + dz*dz))

def pick_max_contrast_color(
    background_color: str,
    color_options: Optional[List[str]] = None
) -> str:
    """
    Pick color with maximum contrast against background

    Args:
        background_color: Background color in hex format
        color_options: List of color options to choose from

    Returns:
        Color with maximum contrast
    """
    if color_options is None:
        # Default high-contrast colors
        color_options = ["#000000", "#FFFFFF", "#FF0000", "#00FF00", "#0000FF"]

    # Simple luminance-based contrast calculation
    def get_luminance(hex_color):
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return 0.299 * r + 0.587 * g + 0.114 * b

    bg_luminance = get_luminance(background_color)

    best_color = color_options[0]
    best_contrast = 0

    for color in color_options:
        color_luminance = get_luminance(color)
        contrast = abs(bg_luminance - color_luminance)

        if contrast > best_contrast:
            best_contrast = contrast
            best_color = color

    return best_color

def is_vector_property(prop_value: Any) -> bool:
    """
    Check if property value is a 3D vector

    Args:
        prop_value: Property value to check

    Returns:
        True if value is a 3D vector
    """
    if isinstance(prop_value, (list, tuple, np.ndarray)):
        return len(prop_value) == 3 and all(
            isinstance(x, (int, float, np.number)) for x in prop_value
        )
    return False

def batch_process_sites(
    sites: List[PeriodicSite],
    process_func: callable,
    **kwargs
) -> List[Any]:
    """
    Process multiple sites efficiently with same function

    Args:
        sites: List of sites to process
        process_func: Function to apply to each site
        **kwargs: Additional arguments for process_func

    Returns:
        List of processing results
    """
    return [process_func(site, **kwargs) for site in sites]

def group_sites_by_element(sites: List[PeriodicSite]) -> Dict[str, List[PeriodicSite]]:
    """
    Group sites by element type with performance improvements

    Args:
        sites: List of periodic sites

    Returns:
        Dictionary mapping element symbols to lists of sites
    """
    groups = {}

    for site in sites:
        element = get_site_symbol(site)
        if element not in groups:
            groups[element] = []
        groups[element].append(site)

    return groups

@lru_cache(maxsize=100)
def get_atomic_number(element_symbol: str) -> int:
    """
    Get atomic number for element with caching

    Args:
        element_symbol: Element symbol

    Returns:
        Atomic number
    """
    # Simple atomic number lookup
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
        'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
        'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
        'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
        'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36
        # Add more as needed
    }
    return atomic_numbers.get(element_symbol, 0)

def filter_sites_by_property(
    sites: List[PeriodicSite],
    property_name: str,
    property_value: Any = None,
    comparison: str = "equals"
) -> List[PeriodicSite]:
    """
    Filter sites by property value

    Args:
        sites: List of sites to filter
        property_name: Name of property to filter by
        property_value: Value to compare against (None to check existence)
        comparison: Comparison type ("equals", "greater", "less", "exists")

    Returns:
        Filtered list of sites
    """
    filtered_sites = []

    for site in sites:
        if not hasattr(site, 'properties') or property_name not in site.properties:
            continue

        site_prop_value = site.properties[property_name]

        if comparison == "exists":
            filtered_sites.append(site)
        elif comparison == "equals" and site_prop_value == property_value:
            filtered_sites.append(site)
        elif comparison == "greater" and site_prop_value > property_value:
            filtered_sites.append(site)
        elif comparison == "less" and site_prop_value < property_value:
            filtered_sites.append(site)

    return filtered_sites

def clear_helpers_cache():
    """
    Clear helper function caches for memory management
    """
    get_site_symbol.cache_clear()
    calculate_distance_3d.cache_clear()
    get_atomic_number.cache_clear()
