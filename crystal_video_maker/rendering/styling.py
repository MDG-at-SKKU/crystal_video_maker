"""
Advanced styling functions that don't create circular imports
"""

from typing import Dict, List, Tuple, Union
import numpy as np
from ..utils.colors import (
    normalize_color, interpolate_colors, get_contrasting_text_color
)

def blend_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """
    Blend two colors with given ratio
    
    Args:
        color1: First color (hex string)
        color2: Second color (hex string)
        ratio: Blending ratio (0.0 = color1, 1.0 = color2)
        
    Returns:
        Blended color as hex string
    """
    colors = interpolate_colors(color1, color2, 3)
    if ratio <= 0.0:
        return colors[0]
    elif ratio >= 1.0:
        return colors[-1]
    else:
        return colors[1]  # Middle color for 0.5 ratio

def adjust_color_brightness(color: str, factor: float) -> str:
    """
    Adjust color brightness
    
    Args:
        color: Input color as hex string
        factor: Brightness factor (>1.0 brighter, <1.0 darker)
        
    Returns:
        Adjusted color as hex string
    """
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    rgb = hex_to_rgb(normalize_color(color))
    adjusted_rgb = tuple(min(255, max(0, int(c * factor))) for c in rgb)
    return rgb_to_hex(adjusted_rgb)

def create_custom_colormap(elements: List[str], colors: List[str]) -> Dict[str, str]:
    """
    Create custom colormap from element list and color list
    
    Args:
        elements: List of element symbols
        colors: List of color values (hex strings or RGB tuples)
        
    Returns:
        Dictionary mapping elements to normalized colors
    """
    if len(elements) != len(colors):
        raise ValueError("Elements and colors lists must have same length")
    
    return {elem: normalize_color(color) for elem, color in zip(elements, colors)}

__all__ = [
    "blend_colors",
    "adjust_color_brightness", 
    "create_custom_colormap"
]
