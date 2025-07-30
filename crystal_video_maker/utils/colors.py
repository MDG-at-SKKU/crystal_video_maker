"""
Color utilities with performance improvements
"""

from functools import lru_cache
from typing import Dict, List, Optional
import plotly.colors as pcolors

# Import from rendering.styling for backwards compatibility
from ..rendering.styling import (
    get_elem_colors,
    get_cached_color_scheme,
    normalize_color,
    generate_color_palette,
    interpolate_colors,
    get_contrasting_text_color,
    color_map_type
)

@lru_cache(maxsize=200)
def get_plotly_colorscale(name: str) -> List:
    """
    Get cached Plotly colorscale

    Args:
        name: Colorscale name

    Returns:
        Plotly colorscale list
    """
    try:
        return getattr(pcolors.sequential, name.title(), pcolors.sequential.Viridis)
    except AttributeError:
        return pcolors.sequential.Viridis

def normalize_elem_color(colors: Dict[str, str]) -> Dict[str, str]:
    """
    Normalize element color dictionary

    Args:
        colors: Element color mapping

    Returns:
        Normalized color mapping with hex colors
    """
    return {elem: normalize_color(color) for elem, color in colors.items()}

def blend_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """
    Blend two colors with given ratio

    Args:
        color1: First color in hex format
        color2: Second color in hex format  
        ratio: Blending ratio (0.0 = color1, 1.0 = color2)

    Returns:
        Blended color in hex format
    """
    if ratio <= 0:
        return color1
    if ratio >= 1:
        return color2

    # Use interpolate_colors for single step
    return interpolate_colors(color1, color2, 2)[int(ratio)]

def get_random_color_palette(n_colors: int, seed: Optional[int] = None) -> List[str]:
    """
    Generate random color palette

    Args:
        n_colors: Number of colors
        seed: Random seed for reproducibility

    Returns:
        List of random hex colors
    """
    import random
    if seed is not None:
        random.seed(seed)

    colors = []
    for _ in range(n_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255) 
        b = random.randint(0, 255)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")

    return colors
