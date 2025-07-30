"""
Styling and color management functions with performance improvements
"""

import plotly.colors as pcolors
from typing import Dict, List, Optional, Any
from functools import lru_cache
import numpy as np

# Default color schemes
DEFAULT_COLOR_SCHEMES = {
    "VESTA": {
        "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
        "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
        "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
        "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
        "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
        "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7",
        "Mn": "#9C7AC7", "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050",
        "Cu": "#C88033", "Zn": "#7D80B0", "Ga": "#C28F8F", "Ge": "#668F8F",
        "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929", "Kr": "#5CB8D1",
        "Rb": "#702EB0", "Sr": "#00FF00", "Y": "#94FFFF", "Zr": "#94E0E0",
        "Nb": "#73C2C9", "Mo": "#54B5B5", "Tc": "#3B9E9E", "Ru": "#248F8F",
        "Rh": "#0A7D8C", "Pd": "#006985", "Ag": "#C0C0C0", "Cd": "#FFD98F",
        "In": "#A67573", "Sn": "#668080", "Sb": "#9E63B5", "Te": "#D47A00",
        "I": "#940094", "Xe": "#429EB0", "Cs": "#57178F", "Ba": "#00C900",
        "La": "#70D4FF", "Ce": "#FFFFC7", "Pr": "#D9FFC7", "Nd": "#C7FFC7",
        "Pm": "#A3FFC7", "Sm": "#8FFFC7", "Eu": "#61FFC7", "Gd": "#45FFC7",
        "Tb": "#30FFC7", "Dy": "#1FFFC7", "Ho": "#00FF9C", "Er": "#00E675",
        "Tm": "#00D452", "Yb": "#00BF38", "Lu": "#00AB24", "Hf": "#4DC2FF",
        "Ta": "#4DA6FF", "W": "#2194D6", "Re": "#267DAB", "Os": "#266696",
        "Ir": "#175487", "Pt": "#D0D0E0", "Au": "#FFD123", "Hg": "#B8B8D0",
        "Tl": "#A6544D", "Pb": "#575961", "Bi": "#9E4FB5", "Po": "#AB5C00",
        "At": "#754F45", "Rn": "#428296", "Fr": "#420066", "Ra": "#007D00",
        "Ac": "#70ABFA", "Th": "#00BAFF", "Pa": "#00A1FF", "U": "#008FFF",
        "Np": "#0080FF", "Pu": "#006BFF", "Am": "#545CF2", "Cm": "#785CE3",
        "Bk": "#8A4FE3", "Cf": "#A136D4", "Es": "#B31FD4", "Fm": "#B31FBA",
        "Md": "#B30DA6", "No": "#BD0D87", "Lr": "#C70066", "Rf": "#CC0059",
        "Db": "#D1004F", "Sg": "#D90045", "Bh": "#E00038", "Hs": "#E6002E",
        "Mt": "#EB0026", "Ds": "#FF0000", "Rg": "#FF0000", "Cn": "#FF0000",
        "Nh": "#FF0000", "Fl": "#FF0000", "Mc": "#FF0000", "Lv": "#FF0000",
        "Ts": "#FF0000", "Og": "#FF0000"
    },
    "Jmol": {
        "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
        "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
        "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
        "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
        "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00"
    }
}

@lru_cache(maxsize=100)
def get_cached_color_scheme(scheme_name: str) -> Dict[str, str]:
    """
    Get cached color scheme for performance

    Args:
        scheme_name: Name of color scheme ("VESTA", "Jmol", etc.)

    Returns:
        Dictionary mapping element symbols to hex colors
    """
    return DEFAULT_COLOR_SCHEMES.get(scheme_name, DEFAULT_COLOR_SCHEMES["VESTA"])

# For backwards compatibility
color_map_type = DEFAULT_COLOR_SCHEMES

def get_elem_colors(
    elem_colors: Optional[Dict[str, str]] = None,
    color_scheme: str = "VESTA"
) -> Dict[str, str]:
    """
    Get element color mapping with caching

    Args:
        elem_colors: Custom element colors or None for default
        color_scheme: Default color scheme to use

    Returns:
        Dictionary mapping element symbols to colors
    """
    if elem_colors is not None:
        return elem_colors

    return get_cached_color_scheme(color_scheme)

@lru_cache(maxsize=500)
def normalize_color(color: str) -> str:
    """
    Normalize color string format with caching

    Args:
        color: Color in various formats (name, hex, rgb)

    Returns:
        Normalized hex color string
    """
    # Handle hex colors
    if color.startswith('#'):
        return color.upper()

    # Handle named colors
    color_lower = color.lower()
    named_colors = {
        'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF',
        'yellow': '#FFFF00', 'cyan': '#00FFFF', 'magenta': '#FF00FF',
        'black': '#000000', 'white': '#FFFFFF', 'gray': '#808080',
        'orange': '#FFA500', 'purple': '#800080', 'brown': '#A52A2A',
        'pink': '#FFC0CB', 'lime': '#00FF00'
    }

    if color_lower in named_colors:
        return named_colors[color_lower]

    # Return as-is if not recognized
    return color

def get_contrasting_text_color(background_color: str) -> str:
    """
    Get contrasting text color (black or white) for a background

    Args:
        background_color: Background color in hex format

    Returns:
        Either "#000000" (black) or "#FFFFFF" (white)
    """
    # Convert hex to RGB
    if background_color.startswith('#'):
        hex_color = background_color[1:]
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:6], 16)

        # Calculate luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

        # Return black for light backgrounds, white for dark
        return "#000000" if luminance > 0.5 else "#FFFFFF"

    return "#000000"  # Default to black

@lru_cache(maxsize=200)
def generate_color_palette(n_colors: int, colorscale: str = "viridis") -> List[str]:
    """
    Generate a color palette with specified number of colors

    Args:
        n_colors: Number of colors to generate
        colorscale: Plotly colorscale name

    Returns:
        List of hex color strings
    """
    if n_colors <= 0:
        return []

    # Use plotly's color scales
    try:
        # Get colorscale colors
        colorscale_colors = getattr(pcolors.qualitative, colorscale.title(), None)
        if colorscale_colors is None:
            # Fallback to sequential colorscale
            colorscale_colors = pcolors.sample_colorscale(colorscale, n_colors)
            return [f"rgb{color}" if isinstance(color, tuple) else color for color in colorscale_colors]

        # Cycle through qualitative colors if needed
        palette = []
        for i in range(n_colors):
            palette.append(colorscale_colors[i % len(colorscale_colors)])

        return palette

    except Exception:
        # Fallback to simple color generation
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF']
        return [colors[i % len(colors)] for i in range(n_colors)]

def interpolate_colors(color1: str, color2: str, n_steps: int) -> List[str]:
    """
    Interpolate between two colors

    Args:
        color1: Start color in hex format
        color2: End color in hex format
        n_steps: Number of interpolation steps

    Returns:
        List of interpolated hex colors
    """
    if n_steps <= 0:
        return []
    if n_steps == 1:
        return [color1]

    # Convert hex to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def rgb_to_hex(rgb):
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    rgb1 = hex_to_rgb(color1)
    rgb2 = hex_to_rgb(color2)

    colors = []
    for i in range(n_steps):
        ratio = i / (n_steps - 1) if n_steps > 1 else 0
        r = int(rgb1[0] + (rgb2[0] - rgb1[0]) * ratio)
        g = int(rgb1[1] + (rgb2[1] - rgb1[1]) * ratio)
        b = int(rgb1[2] + (rgb2[2] - rgb1[2]) * ratio)
        colors.append(rgb_to_hex((r, g, b)))

    return colors

def get_element_category_colors() -> Dict[str, str]:
    """
    Get colors based on element categories (metals, nonmetals, etc.)

    Returns:
        Dictionary mapping element symbols to category colors
    """
    categories = {
        # Alkali metals
        "alkali": "#FF6B6B",
        # Alkaline earth metals  
        "alkaline_earth": "#4ECDC4",
        # Transition metals
        "transition": "#45B7D1", 
        # Post-transition metals
        "post_transition": "#96CEB4",
        # Metalloids
        "metalloid": "#FFEAA7",
        # Nonmetals
        "nonmetal": "#DDA0DD",
        # Halogens
        "halogen": "#98D8C8",
        # Noble gases
        "noble_gas": "#F7DC6F"
    }

    # Element categorization (simplified)
    element_categories = {
        "H": "nonmetal", "He": "noble_gas",
        "Li": "alkali", "Be": "alkaline_earth",
        "B": "metalloid", "C": "nonmetal", "N": "nonmetal", "O": "nonmetal",
        "F": "halogen", "Ne": "noble_gas",
        "Na": "alkali", "Mg": "alkaline_earth",
        "Al": "post_transition", "Si": "metalloid", "P": "nonmetal", "S": "nonmetal",
        "Cl": "halogen", "Ar": "noble_gas",
        # Add more as needed...
    }

    return {elem: categories.get(cat, "#808080") 
            for elem, cat in element_categories.items()}

def clear_styling_cache():
    """
    Clear styling-related caches for memory management
    """
    get_cached_color_scheme.cache_clear()
    normalize_color.cache_clear()
    generate_color_palette.cache_clear()
