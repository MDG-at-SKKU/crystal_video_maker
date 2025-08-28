"""
Styling utilities for Crystal Viewer
Color schemes, materials, and visual styling optimized for PyVista
"""

from typing import Dict, List, Tuple, Union, Optional, Any
import colorsys
from functools import lru_cache

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .config import COLOR_SCHEMES, get_element_color
from .utils import normalize_color, interpolate_colors, get_contrasting_color


class Material:
    """Material properties for rendering"""

    def __init__(
        self,
        ambient: float = 0.3,
        diffuse: float = 0.7,
        specular: float = 0.2,
        specular_power: float = 100.0,
        opacity: float = 1.0,
        color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    ):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.specular_power = specular_power
        self.opacity = opacity
        self.color = color

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PyVista"""
        return {
            "ambient": self.ambient,
            "diffuse": self.diffuse,
            "specular": self.specular,
            "specular_power": self.specular_power,
            "opacity": self.opacity,
            "color": self.color,
        }


class ColorScheme:
    """Color scheme management"""

    def __init__(
        self,
        name: str = "vesta",
        colors: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        self.name = name
        if colors:
            self.colors = colors
        else:
            self.colors = COLOR_SCHEMES.get(name, COLOR_SCHEMES["vesta"])

    def get_color(self, element: str) -> Tuple[float, float, float]:
        """Get color for element"""
        return self.colors.get(element, self.colors.get("XX", (0.5, 0.5, 0.5)))

    def set_color(self, element: str, color: Tuple[float, float, float]):
        """Set color for element"""
        self.colors[element] = color

    def get_all_colors(self) -> Dict[str, Tuple[float, float, float]]:
        """Get all colors in scheme"""
        return self.colors.copy()


class Gradient:
    """Color gradient for continuous data"""

    def __init__(
        self,
        colors: List[Tuple[float, float, float]],
        positions: Optional[List[float]] = None,
    ):
        self.colors = colors
        if positions is None:
            self.positions = [i / (len(colors) - 1) for i in range(len(colors))]
        else:
            self.positions = positions

    def get_color_at(self, position: float) -> Tuple[float, float, float]:
        """Get color at specific position (0-1)"""
        if position <= 0:
            return self.colors[0]
        if position >= 1:
            return self.colors[-1]

        for i in range(len(self.positions) - 1):
            if self.positions[i] <= position <= self.positions[i + 1]:
                t = (position - self.positions[i]) / (
                    self.positions[i + 1] - self.positions[i]
                )
                color1 = self.colors[i]
                color2 = self.colors[i + 1]

                return tuple(c1 + t * (c2 - c1) for c1, c2 in zip(color1, color2))

        return self.colors[-1]


def create_viridis_colormap(n_colors: int = 256) -> List[Tuple[float, float, float]]:
    """Create viridis colormap"""
    # Simplified viridis colormap
    colors = []
    for i in range(n_colors):
        t = i / (n_colors - 1)

        if t < 0.25:
            r = 0.267
            g = 0.004
            b = 0.329 + t * 2.5
        elif t < 0.5:
            r = 0.267 + (t - 0.25) * 2.5
            g = 0.004 + (t - 0.25) * 3.5
            b = 0.829 - (t - 0.25) * 0.5
        elif t < 0.75:
            r = 0.767 - (t - 0.5) * 1.5
            g = 0.504 + (t - 0.5) * 2.0
            b = 0.329 - (t - 0.5) * 0.5
        else:
            r = 0.267 - (t - 0.75) * 0.5
            g = 0.904 + (t - 0.75) * 0.5
            b = 0.129 + (t - 0.75) * 0.5

        colors.append((r, g, b))

    return colors


def create_plasma_colormap(n_colors: int = 256) -> List[Tuple[float, float, float]]:
    """Create plasma colormap"""
    colors = []
    for i in range(n_colors):
        t = i / (n_colors - 1)

        r = 0.050 + t * 0.950
        g = 0.030 + t * 0.310 if t < 0.5 else 0.340 + (t - 0.5) * 0.660
        b = 0.525 + t * 0.475 if t < 0.5 else 1.000 - (t - 0.5) * 0.475

        colors.append((r, g, b))

    return colors


def create_rainbow_colormap(n_colors: int = 256) -> List[Tuple[float, float, float]]:
    """Create rainbow colormap"""
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)
    return colors


def blend_colors(
    color1: Tuple[float, float, float],
    color2: Tuple[float, float, float],
    factor: float = 0.5,
) -> Tuple[float, float, float]:
    """Blend two colors"""
    factor = max(0.0, min(1.0, factor))
    return tuple(c1 * (1 - factor) + c2 * factor for c1, c2 in zip(color1, color2))


def adjust_color_brightness(
    color: Tuple[float, float, float], factor: float
) -> Tuple[float, float, float]:
    """Adjust color brightness"""
    h, s, v = colorsys.rgb_to_hsv(*color)
    new_v = max(0.0, min(1.0, v * factor))
    return colorsys.hsv_to_rgb(h, s, new_v)


def create_material_from_properties(properties: Dict[str, Any]) -> Material:
    """Create material from properties dictionary"""
    return Material(
        ambient=properties.get("ambient", 0.3),
        diffuse=properties.get("diffuse", 0.7),
        specular=properties.get("specular", 0.2),
        specular_power=properties.get("specular_power", 100.0),
        opacity=properties.get("opacity", 1.0),
        color=properties.get("color", (0.5, 0.5, 0.5)),
    )


def get_element_material(element: str, scheme: str = "default") -> Material:
    """Get default material for element"""
    color = get_element_color(element, scheme)
    return Material(color=color)


def create_transparent_material(
    color: Tuple[float, float, float], opacity: float = 0.5
) -> Material:
    """Create transparent material"""
    return Material(
        color=color, opacity=opacity, ambient=0.4, diffuse=0.6, specular=0.1
    )


def create_metal_material(color: Tuple[float, float, float]) -> Material:
    """Create metallic material"""
    return Material(
        color=color, ambient=0.2, diffuse=0.5, specular=0.8, specular_power=200.0
    )


def create_glass_material(color: Tuple[float, float, float]) -> Material:
    """Create glass-like material"""
    return Material(
        color=color,
        opacity=0.3,
        ambient=0.1,
        diffuse=0.3,
        specular=0.9,
        specular_power=300.0,
    )


@lru_cache(maxsize=100)
def get_cached_colormap(
    name: str, n_colors: int = 256
) -> List[Tuple[float, float, float]]:
    """Get cached colormap"""
    if name == "viridis":
        return create_viridis_colormap(n_colors)
    elif name == "plasma":
        return create_plasma_colormap(n_colors)
    elif name == "rainbow":
        return create_rainbow_colormap(n_colors)
    else:
        return create_viridis_colormap(n_colors)  # Default


def apply_gamma_correction(
    color: Tuple[float, float, float], gamma: float = 2.2
) -> Tuple[float, float, float]:
    """Apply gamma correction to color"""
    return tuple(pow(c, 1.0 / gamma) for c in color)


def convert_color_space(
    color: Tuple[float, float, float], from_space: str = "rgb", to_space: str = "hsv"
) -> Tuple[float, float, float]:
    """Convert color between color spaces"""
    if from_space == "rgb" and to_space == "hsv":
        return colorsys.rgb_to_hsv(*color)
    elif from_space == "hsv" and to_space == "rgb":
        return colorsys.hsv_to_rgb(*color)
    elif from_space == "rgb" and to_space == "hsl":
        return colorsys.rgb_to_hls(*color)
    elif from_space == "hsl" and to_space == "rgb":
        return colorsys.hls_to_rgb(*color)
    else:
        return color


def create_color_palette(
    elements: List[str], base_hue: float = 0.0
) -> Dict[str, Tuple[float, float, float]]:
    """Create color palette for elements"""
    palette = {}
    n_elements = len(elements)

    for i, element in enumerate(elements):
        hue = (base_hue + i / n_elements) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        palette[element] = rgb

    return palette


def get_colorblind_friendly_colors() -> List[Tuple[float, float, float]]:
    """Get colorblind-friendly color palette"""
    return [
        (0.0, 0.447, 0.698),
        (0.835, 0.369, 0.0),
        (0.8, 0.6, 0.0),
        (0.0, 0.6, 0.5),
        (0.902, 0.624, 0.0),
        (0.337, 0.706, 0.914),
        (0.941, 0.894, 0.259),
        (0.0, 0.447, 0.698),
    ]


def create_highlight_material(
    base_color: Tuple[float, float, float], highlight_factor: float = 1.5
) -> Material:
    """Create highlighted material"""
    highlighted_color = adjust_color_brightness(base_color, highlight_factor)
    return Material(
        color=highlighted_color,
        ambient=0.4,
        diffuse=0.8,
        specular=0.3,
        specular_power=150.0,
    )


def create_outline_material(
    color: Tuple[float, float, float] = (0.0, 0.0, 0.0), width: float = 2.0
) -> Dict[str, Any]:
    """Create outline material properties"""
    return {"color": color, "line_width": width, "style": "solid"}


def interpolate_materials(
    material1: Material, material2: Material, factor: float = 0.5
) -> Material:
    """Interpolate between two materials"""
    return Material(
        ambient=material1.ambient + factor * (material2.ambient - material1.ambient),
        diffuse=material1.diffuse + factor * (material2.diffuse - material1.diffuse),
        specular=material1.specular
        + factor * (material2.specular - material1.specular),
        specular_power=int(
            material1.specular_power
            + factor * (material2.specular_power - material1.specular_power)
        ),
        opacity=material1.opacity + factor * (material2.opacity - material1.opacity),
        color=blend_colors(material1.color, material2.color, factor),
    )


# Predefined materials
METAL_MATERIAL = create_metal_material((0.8, 0.8, 0.9))
GLASS_MATERIAL = create_glass_material((0.9, 0.9, 1.0))
PLASTIC_MATERIAL = Material(color=(0.5, 0.5, 0.5), specular=0.1, specular_power=50.0)

# Export functions
__all__ = [
    "Material",
    "ColorScheme",
    "Gradient",
    "create_viridis_colormap",
    "create_plasma_colormap",
    "create_rainbow_colormap",
    "blend_colors",
    "adjust_color_brightness",
    "create_material_from_properties",
    "get_element_material",
    "create_transparent_material",
    "create_metal_material",
    "create_glass_material",
    "get_cached_colormap",
    "apply_gamma_correction",
    "convert_color_space",
    "create_color_palette",
    "get_colorblind_friendly_colors",
    "create_highlight_material",
    "create_outline_material",
    "interpolate_materials",
    "METAL_MATERIAL",
    "GLASS_MATERIAL",
    "PLASTIC_MATERIAL",
]
