"""
Label and text generation utilities with performance improvements
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional, Callable
import numpy as np
from pymatgen.core import PeriodicSite

@lru_cache(maxsize=1000)
def _get_cached_element_info(element_str: str) -> Dict[str, Any]:
    """
    Get cached element information

    Args:
        element_str: Element string representation

    Returns:
        Dictionary with element information
    """
    # Parse element string to extract symbol and oxidation state
    import re

    # Match patterns like "Fe2+", "O2-", "Ca", etc.
    match = re.match(r'([A-Z][a-z]?)([+-]?\d*)', element_str)
    if match:
        symbol = match.group(1)
        charge_str = match.group(2)

        charge = 0
        if charge_str:
            if charge_str.endswith(('+', '-')):
                sign = 1 if charge_str.endswith('+') else -1
                number = charge_str[:-1]
                charge = sign * (int(number) if number else 1)
            else:
                charge = int(charge_str)

        return {
            'symbol': symbol,
            'charge': charge,
            'has_charge': charge != 0
        }

    return {'symbol': element_str, 'charge': 0, 'has_charge': False}

def generate_site_label(
    site: PeriodicSite,
    label_type: str = "symbol",
    include_index: bool = False,
    site_index: Optional[int] = None
) -> str:
    """
    Generate label for a site with caching

    Args:
        site: Periodic site object
        label_type: Type of label ("symbol", "species", "element", "index")
        include_index: Whether to include site index
        site_index: Site index number

    Returns:
        Generated label string
    """
    if label_type == "index":
        return str(site_index) if site_index is not None else "0"

    # Get element information
    if hasattr(site, 'specie'):
        element_str = str(site.specie)
    elif hasattr(site, 'species_string'):
        element_str = site.species_string
    else:
        element_str = str(site.species)

    element_info = _get_cached_element_info(element_str)

    if label_type == "symbol":
        label = element_info['symbol']
    elif label_type == "species":
        label = element_str
    elif label_type == "element":
        label = element_info['symbol']
        if element_info['has_charge']:
            charge = element_info['charge']
            if charge > 0:
                label += f"{charge}+" if charge > 1 else "+"
            elif charge < 0:
                label += f"{abs(charge)}-" if abs(charge) > 1 else "-"
    else:
        label = element_str

    if include_index and site_index is not None:
        label += f" ({site_index})"

    return label

def get_site_hover_text(
    site: PeriodicSite,
    coords_type: str = "cartesian",
    float_fmt: str = ".4f",
    include_properties: bool = True,
    site_index: Optional[int] = None
) -> str:
    """
    Generate hover text for a site with performance improvements

    Args:
        site: Periodic site object
        coords_type: Coordinate type ("cartesian", "fractional", "both")
        float_fmt: Float formatting string
        include_properties: Whether to include site properties
        site_index: Site index number

    Returns:
        Formatted hover text string
    """
    lines = []

    # Element/species information
    if hasattr(site, 'specie'):
        lines.append(f"Element: {site.specie}")
    elif hasattr(site, 'species_string'):
        lines.append(f"Species: {site.species_string}")
    else:
        lines.append(f"Species: {site.species}")

    # Site index
    if site_index is not None:
        lines.append(f"Site Index: {site_index}")

    # Coordinates
    if coords_type in ("cartesian", "both"):
        coords = site.coords
        coord_str = f"[{coords[0]:{float_fmt}}, {coords[1]:{float_fmt}}, {coords[2]:{float_fmt}}]"
        lines.append(f"Cartesian: {coord_str}")

    if coords_type in ("fractional", "both"):
        frac_coords = site.frac_coords
        frac_str = f"[{frac_coords[0]:{float_fmt}}, {frac_coords[1]:{float_fmt}}, {frac_coords[2]:{float_fmt}}]"
        lines.append(f"Fractional: {frac_str}")

    # Properties
    if include_properties and hasattr(site, 'properties') and site.properties:
        prop_lines = []
        for prop_name, prop_value in site.properties.items():
            if prop_name.startswith('_'):  # Skip private properties
                continue

            if isinstance(prop_value, (list, tuple, np.ndarray)):
                if len(prop_value) == 3:  # Vector property
                    val_str = f"[{prop_value[0]:{float_fmt}}, {prop_value[1]:{float_fmt}}, {prop_value[2]:{float_fmt}}]"
                else:
                    val_str = str(prop_value)
            elif isinstance(prop_value, float):
                val_str = f"{prop_value:{float_fmt}}"
            else:
                val_str = str(prop_value)

            prop_lines.append(f"{prop_name}: {val_str}")

        if prop_lines:
            lines.append("Properties:")
            lines.extend(f"  {line}" for line in prop_lines)

    return "<br>".join(lines)

@lru_cache(maxsize=500)
def format_coordinate_string(
    coords: tuple,
    coord_type: str = "cartesian",
    float_fmt: str = ".4f"
) -> str:
    """
    Format coordinate tuple as string with caching

    Args:
        coords: Coordinate tuple (x, y, z)
        coord_type: Type of coordinates for labeling
        float_fmt: Float format string

    Returns:
        Formatted coordinate string
    """
    x, y, z = coords
    coord_str = f"[{x:{float_fmt}}, {y:{float_fmt}}, {z:{float_fmt}}]"

    if coord_type == "cartesian":
        return f"Cart: {coord_str}"
    elif coord_type == "fractional":
        return f"Frac: {coord_str}"
    else:
        return coord_str

def generate_subplot_title(
    structure_key: str,
    structure_index: int,
    structure: Any,
    custom_formatter: Optional[Callable] = None
) -> str:
    """
    Generate title for structure subplot

    Args:
        structure_key: Structure identifier key
        structure_index: Structure index number
        structure: Structure object
        custom_formatter: Optional custom title formatter function

    Returns:
        Generated subplot title
    """
    if custom_formatter is not None:
        try:
            return custom_formatter(structure, structure_key)
        except Exception:
            pass  # Fall back to default formatting

    # Default formatting
    if hasattr(structure, 'formula'):
        formula = structure.formula
        if len(structure_key) > 20:  # Truncate very long keys
            title = f"{structure_index}: {formula}"
        else:
            title = f"{structure_key}"
    else:
        title = structure_key

    return title

def create_legend_labels(
    elements: List[str],
    include_counts: bool = False,
    element_counts: Optional[Dict[str, int]] = None
) -> Dict[str, str]:
    """
    Create legend labels for elements

    Args:
        elements: List of element symbols
        include_counts: Whether to include atom counts
        element_counts: Dictionary of element counts

    Returns:
        Dictionary mapping elements to legend labels
    """
    labels = {}

    for element in elements:
        if include_counts and element_counts and element in element_counts:
            count = element_counts[element]
            labels[element] = f"{element} ({count})"
        else:
            labels[element] = element

    return labels

def clear_labels_cache():
    """
    Clear label generation caches for memory management
    """
    _get_cached_element_info.cache_clear()
    format_coordinate_string.cache_clear()
