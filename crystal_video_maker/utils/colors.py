"""
Color utilities and management functions - 순환 import 제거
"""

from functools import lru_cache
from typing import Dict, List, Union, Tuple
import colorsys
import warnings


# VESTA color scheme
VESTA_COLORS = {
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
    "Md": "#B30DA6", "No": "#BD0D87", "Lr": "#C70066", "XX": "#808080"
}

# Jmol color scheme
JMOL_COLORS = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00",
    "B": "#FFB5B5", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Fe": "#E06633", "Cu": "#C88033", "Zn": "#7D80B0", "Ag": "#C0C0C0",
    "Au": "#FFD123", "XX": "#FF1493"
}

# CPK color scheme
CPK_COLORS = {
    "H": "#FFFFFF", "C": "#000000", "N": "#8F8FFF", "O": "#FF0000",
    "F": "#00FF00", "P": "#FFA500", "S": "#FFFF00", "Cl": "#00FF00",
    "Fe": "#FFA500", "XX": "#FF1493"
}

# Color scheme registry
color_map_type = {
    "VESTA": VESTA_COLORS,
    "Jmol": JMOL_COLORS, 
    "CPK": CPK_COLORS,
    "default": VESTA_COLORS
}

@lru_cache(maxsize=1000)
def get_elem_colors(color_scheme: Union[str, Dict[str, str]] = "VESTA") -> Dict[str, str]:
    """
    Get element color mapping for a given color scheme
    
    Args:
        color_scheme: Color scheme name or custom color dictionary
        
    Returns:
        Dictionary mapping element symbols to hex colors
    """
    if isinstance(color_scheme, dict):
        return color_scheme
    
    scheme_name = color_scheme.upper() if isinstance(color_scheme, str) else "VESTA"
    return color_map_type.get(scheme_name, VESTA_COLORS)

@lru_cache(maxsize=100)
def get_cached_color_scheme(scheme_name: str) -> Dict[str, str]:
    """
    Get cached color scheme by name
    
    Args:
        scheme_name: Name of the color scheme
        
    Returns:
        Dictionary of element colors
    """
    return color_map_type.get(scheme_name.upper(), VESTA_COLORS)

def get_default_colors() -> Dict[str, str]:
    """
    Get default VESTA color scheme
    
    Returns:
        Dictionary of default element colors
    """
    return VESTA_COLORS.copy()

def normalize_color(color: Union[str, Tuple[float, float, float]]) -> str:
    """
    Normalize color input to hex string format
    
    Args:
        color: Color as hex string, RGB tuple, or color name
        
    Returns:
        Normalized hex color string
    """
    if isinstance(color, str):
        if color.startswith('#'):
            return color
        elif color.startswith('rgb'):
            # Parse rgb(r,g,b) format
            import re
            rgb_match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', color)
            if rgb_match:
                r, g, b = map(int, rgb_match.groups())
                return f"#{r:02x}{g:02x}{b:02x}"
        else:
            # Try to convert named color
            named_colors = {
                'red': '#FF0000', 'green': '#00FF00', 'blue': '#0000FF',
                'yellow': '#FFFF00', 'cyan': '#00FFFF', 'magenta': '#FF00FF',
                'black': '#000000', 'white': '#FFFFFF', 'gray': '#808080',
                'orange': '#FFA500', 'purple': '#800080', 'brown': '#A52A2A'
            }
            return named_colors.get(color.lower(), color)
    
    elif isinstance(color, (tuple, list)) and len(color) == 3:
        r, g, b = color
        # Convert from 0-1 range to 0-255 if needed
        if all(0 <= c <= 1 for c in color):
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    return "#808080"  # Default gray

def generate_color_palette(n_colors: int, scheme: str = "viridis") -> List[str]:
    """
    Generate a color palette with specified number of colors
    
    Args:
        n_colors: Number of colors to generate
        scheme: Color scheme name
        
    Returns:
        List of hex color strings
    """
    try:
        import plotly.colors as pcolors
        
        if scheme in ['viridis', 'plasma', 'inferno', 'magma', 'cividis']:
            # Use plotly colorscales
            colorscale = getattr(pcolors.sequential, scheme.title(), pcolors.sequential.Viridis)
            colors = pcolors.sample_colorscale(colorscale, n_colors)
            return [normalize_color(color) for color in colors]
        
        elif scheme == "rainbow":
            # Generate rainbow colors using HSV
            colors = []
            for i in range(n_colors):
                hue = i / n_colors
                rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                colors.append(normalize_color(rgb))
            return colors
        
        elif scheme == "categorical":
            # Use plotly's qualitative colors
            plotly_colors = pcolors.qualitative.Plotly
            colors = [plotly_colors[i % len(plotly_colors)] for i in range(n_colors)]
            return colors
    
    except ImportError:
        # Fallback without plotly
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append(normalize_color(rgb))
        return colors
    
    # Default to rainbow
    return generate_color_palette(n_colors, "rainbow")

def interpolate_colors(color1: str, color2: str, n_steps: int = 10) -> List[str]:
    """
    Interpolate between two colors
    
    Args:
        color1: Starting color (hex string)
        color2: Ending color (hex string)
        n_steps: Number of interpolation steps
        
    Returns:
        List of interpolated hex colors
    """
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    rgb1 = hex_to_rgb(normalize_color(color1))
    rgb2 = hex_to_rgb(normalize_color(color2))
    
    colors = []
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0
        interpolated_rgb = tuple(
            int(rgb1[j] + t * (rgb2[j] - rgb1[j])) for j in range(3)
        )
        colors.append(rgb_to_hex(interpolated_rgb))
    
    return colors

def get_contrasting_text_color(background_color: str) -> str:
    """
    Get contrasting text color (black or white) for a given background color
    
    Args:
        background_color: Background color as hex string
        
    Returns:
        Contrasting text color ("#000000" or "#FFFFFF")
    """
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def luminance(rgb: Tuple[int, int, int]) -> float:
        # Calculate relative luminance using WCAG formula
        def normalize_channel(c: int) -> float:
            c_norm = c / 255.0
            return c_norm / 12.92 if c_norm <= 0.03928 else ((c_norm + 0.055) / 1.055) ** 2.4
        
        r, g, b = rgb
        return 0.2126 * normalize_channel(r) + 0.7152 * normalize_channel(g) + 0.0722 * normalize_channel(b)
    
    bg_color = normalize_color(background_color)
    rgb = hex_to_rgb(bg_color)
    lum = luminance(rgb)
    
    # Use white text on dark backgrounds, black text on light backgrounds
    return "#FFFFFF" if lum < 0.5 else "#000000"

# Additional utility functions
@lru_cache(maxsize=100)
def get_color_scheme_names() -> List[str]:
    """Get list of available color scheme names"""
    return list(color_map_type.keys())

def validate_color_scheme(scheme: Union[str, Dict[str, str]]) -> bool:
    """Validate if color scheme is valid"""
    if isinstance(scheme, dict):
        return all(isinstance(k, str) and isinstance(v, str) for k, v in scheme.items())
    elif isinstance(scheme, str):
        return scheme.upper() in color_map_type
    return False

def get_element_colors_for_structure(elements: List[str], scheme: str = "VESTA") -> Dict[str, str]:
    """Get colors for specific elements in a structure"""
    all_colors = get_elem_colors(scheme)
    return {elem: all_colors.get(elem, "#808080") for elem in elements}

# Legacy aliases
pick_max_contrast_color = get_contrasting_text_color
get_element_color_cached = lambda element, scheme="VESTA": get_elem_colors(scheme).get(element, "#808080")

__all__ = [
    "get_elem_colors",
    "get_cached_color_scheme",
    "get_default_colors",
    "normalize_color",
    "generate_color_palette",
    "interpolate_colors",
    "get_contrasting_text_color",
    "color_map_type",
    "get_color_scheme_names",
    "validate_color_scheme",
    "get_element_colors_for_structure",
    "pick_max_contrast_color",
    "get_element_color_cached",
    "VESTA_COLORS",
    "JMOL_COLORS",
    "CPK_COLORS"
]
