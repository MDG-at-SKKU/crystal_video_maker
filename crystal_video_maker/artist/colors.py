# __init__.py
from typing import TypeAlias

color_map_type = {
    "Jmol": {
        "Ac": "#70ABFA",
        "Ag": "#C0C0C0",
        "Al": "#BFA6A6",
        "Am": "#545CF2",
        "Ar": "#80D1E3",
        "As": "#BD80E3",
        "At": "#754F45",
        "Au": "#FFD123",
        "B": "#FFB5B5",
        "Ba": "#00C900",
        "Be": "#C2FF00",
        "Bh": "#E00038",
        "Bi": "#9E4FB5",
        "Bk": "#8A4FE3",
        "Br": "#A62929",
        "C": "#909090",
        "Ca": "#3DFF00",
        "Cd": "#FFD98F",
        "Ce": "#FFFFC7",
        "Cf": "#A136D4",
        "Cl": "#1FF01F",
        "Cm": "#785CE3",
        "Co": "#F090A0",
        "Cr": "#8A99C7",
        "Cs": "#57178F",
        "Cu": "#C88033",
        "Db": "#D1004F",
        "Dy": "#1FFFC7",
        "Er": "#00E675",
        "Es": "#B31FD4",
        "Eu": "#61FFC7",
        "F": "#90E050",
        "Fe": "#E06633",
        "Fm": "#B31FBA",
        "Fr": "#420066",
        "Ga": "#C28F8F",
        "Gd": "#45FFC7",
        "Ge": "#668F8F",
        "H": "#FFFFFF",
        "He": "#D9FFFF",
        "Hf": "#4DC2FF",
        "Hg": "#B8B8D0",
        "Ho": "#00FF9C",
        "Hs": "#E6002E",
        "I": "#940094",
        "In": "#A67573",
        "Ir": "#175487",
        "K": "#8F40D4",
        "Kr": "#5CB8D1",
        "La": "#70D4FF",
        "Li": "#CC80FF",
        "Lr": "#C70066",
        "Lu": "#00AB24",
        "Md": "#B30DA6",
        "Mg": "#8AFF00",
        "Mn": "#9C7AC7",
        "Mo": "#54B5B5",
        "Mt": "#EB0026",
        "N": "#3050F8",
        "Na": "#AB5CF2",
        "Nb": "#73C2C9",
        "Nd": "#C7FFC7",
        "Ne": "#B3E3F5",
        "Ni": "#50D050",
        "No": "#BD0D87",
        "Np": "#0080FF",
        "O": "#FF0D0D",
        "Os": "#266696",
        "P": "#FF8000",
        "Pa": "#00A1FF",
        "Pb": "#575961",
        "Pd": "#006985",
        "Pm": "#A3FFC7",
        "Po": "#AB5C00",
        "Pr": "#D9FFC7",
        "Pt": "#D0D0E0",
        "Pu": "#006BFF",
        "Ra": "#007D00",
        "Rb": "#702EB0",
        "Re": "#267DAB",
        "Rf": "#CC0059",
        "Rh": "#0A7D8C",
        "Rn": "#428296",
        "Ru": "#248F8F",
        "S": "#FFFF30",
        "Sb": "#9E63B5",
        "Sc": "#E6E6E6",
        "Se": "#FFA100",
        "Sg": "#D90045",
        "Si": "#F0C8A0",
        "Sm": "#8FFFC7",
        "Sn": "#668080",
        "Sr": "#00FF00",
        "Ta": "#4DA6FF",
        "Tb": "#30FFC7",
        "Tc": "#3B9E9E",
        "Te": "#D47A00",
        "Th": "#00BAFF",
        "Ti": "#BFC2C7",
        "Tl": "#A6544D",
        "Tm": "#00D452",
        "U": "#008FFF",
        "V": "#A6A6AB",
        "W": "#2194D6",
        "Xe": "#429EB0",
        "Y": "#94FFFF",
        "Yb": "#00BF38",
        "Zn": "#7D80B0",
        "Zr": "#94E0E0",
    },
    "VESTA": {
        "Ac": "#70ABFA",
        "Ag": "#C0C0C0",
        "Al": "#81B2D6",
        "Am": "#545CF2",
        "Ar": "#CFFEC4",
        "As": "#74D057",
        "At": "#754F45",
        "Au": "#FFD123",
        "B": "#1FA20F",
        "Ba": "#00C900",
        "Be": "#5ED77B",
        "Bh": "#E00038",
        "Bi": "#9E4FB5",
        "Bk": "#8A4FE3",
        "Br": "#7E3102",
        "C": "#4C4C4C",
        "Ca": "#5A96BD",
        "Cd": "#FFD98F",
        "Ce": "#FFFFC7",
        "Cf": "#A136D4",
        "Cl": "#31FC02",
        "Cm": "#785CE3",
        "Co": "#0000AF",
        "Cr": "#00009E",
        "Cs": "#57178F",
        "Cu": "#2247DC",
        "Db": "#D1004F",
        "Dy": "#1FFFC7",
        "Er": "#00E675",
        "Es": "#B31FD4",
        "Eu": "#61FFC7",
        "F": "#B0B9E6",
        "Fe": "#B57100",
        "Fm": "#B31FBA",
        "Fr": "#420066",
        "Ga": "#9EE373",
        "Gd": "#45FFC7",
        "Ge": "#7E6EA6",
        "H": "#FFCCCC",
        "He": "#FCE8CE",
        "Hf": "#4DC2FF",
        "Hg": "#B8B8D0",
        "Ho": "#00FF9C",
        "Hs": "#E6002E",
        "I": "#940094",
        "In": "#A67573",
        "Ir": "#175487",
        "K": "#A121F6",
        "Kr": "#FAC1F3",
        "La": "#5AC449",
        "Li": "#86DF73",
        "Lr": "#C70066",
        "Lu": "#00AB24",
        "Md": "#B30DA6",
        "Mg": "#FB7B15",
        "Mn": "#A7089D",
        "Mo": "#54B5B5",
        "Mt": "#EB0026",
        "N": "#B0B9E6",
        "Na": "#F9DC3C",
        "Nb": "#73C2C9",
        "Nd": "#C7FFC7",
        "Ne": "#FE37B5",
        "Ni": "#B7BBBD",
        "No": "#BD0D87",
        "Np": "#0080FF",
        "O": "#FE0300",
        "Os": "#266696",
        "P": "#C09CC2",
        "Pa": "#00A1FF",
        "Pb": "#575961",
        "Pd": "#006985",
        "Pm": "#A3FFC7",
        "Po": "#AB5C00",
        "Pr": "#D9FFC7",
        "Pt": "#D0D0E0",
        "Pu": "#006BFF",
        "Ra": "#007D00",
        "Rb": "#702EB0",
        "Re": "#267DAB",
        "Rf": "#CC0059",
        "Rh": "#0A7D8C",
        "Rn": "#428296",
        "Ru": "#248F8F",
        "S": "#FFFA00",
        "Sb": "#9E63B5",
        "Sc": "#B563AB",
        "Se": "#9AEF0F",
        "Sg": "#D90045",
        "Si": "#1B3BFA",
        "Sm": "#8FFFC7",
        "Sn": "#9A8EB9",
        "Sr": "#00FF00",
        "Ta": "#4DA6FF",
        "Tb": "#30FFC7",
        "Tc": "#3B9E9E",
        "Te": "#D47A00",
        "Th": "#00BAFF",
        "Ti": "#78CAFF",
        "Tl": "#A6544D",
        "Tm": "#00D452",
        "U": "#008FFF",
        "V": "#E51900",
        "W": "#2194D6",
        "Xe": "#429EB0",
        "Y": "#94FFFF",
        "Yb": "#00BF38",
        "Zn": "#8F8F81",
        "Zr": "#00FF00",
    },
    "Extras": {
        "Xbcp": "#0000FF",
        "Xrcp": "#FF0000",
        "Xccp": "#FFFF00",
        "Xncp": "#00FFFF",
    },
}

Rgb256ColorType: TypeAlias = tuple[int, int, int]  # 8-bit RGB

RgbColorType: TypeAlias = tuple[float, float, float] | str  # normalized to [0, 1]

RgbAColorType: TypeAlias = (  # normalized to [0, 1] with alpha
    str  # "none" or "#RRGGBBAA"/"#RGBA" hex strings
    | tuple[float, float, float, float]
    | tuple[RgbColorType, float]
    | tuple[tuple[float, float, float, float], float]
)
ColorType: TypeAlias = RgbColorType | RgbAColorType

def contrast_ratio(color1: ColorType, color2: ColorType) -> float:
    """Calculate the contrast ratio between two colors according to WCAG 2.0.

    Args:
        color1 (ColorType): First color (RGB tuple with values in [0, 1] or [0, 255],
            or a color string that can be converted to RGB).
        color2 (ColorType): Second color (RGB tuple with values in [0, 1] or [0, 255],
            or a color string that can be converted to RGB).

    Returns:
        float: Contrast ratio between the two colors, ranging from 1:1 to 21:1.
    """
    lum1 = luminance(color1)
    lum2 = luminance(color2)

    # Ensure lighter color is first for the formula
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)

    # Calculate contrast ratio: (L1 + 0.05) / (L2 + 0.05)
    return (lighter + 0.05) / (darker + 0.05)


def pick_max_contrast_color(
    bg_color: ColorType,
    colors: tuple[ColorType, ColorType] = ("white", "black"),
    min_contrast_ratio: float = 2.0,  # Lower threshold makes dark colors get white text
) -> ColorType:
    """Choose text color for a given background color based on WCAG 2.0 contrast ratio.

    This function calculates the contrast ratio between the background color and each
    of the provided text colors, then returns the color with the highest contrast ratio.
    If the contrast ratio with white is above the minimum contrast ratio, white will be
    chosen even if black has a slightly higher contrast ratio. This ensures that darker
    colors always get white text, which is often more readable in 3D visualizations.

    Args:
        bg_color (ColorType): Background color.
        colors (tuple[ColorType, ColorType], optional): Text colors to choose
            from. Defaults to ("white", "black").
        min_contrast_ratio (float, optional): Minimum contrast ratio to prefer white
            over black text. Defaults to 2.0 (lower than WCAG AA standard to ensure
            dark colors get white text).

    Returns:
        ColorType: item in `colors` that provides the best contrast with bg_color.
    """
    # Calculate contrast ratios for each potential text color
    contrast_ratios = [contrast_ratio(bg_color, color) for color in colors]

    # If the contrast ratio with white is above the minimum contrast ratio,
    # prefer white text even if black has a slightly higher contrast ratio
    if contrast_ratios[0] >= min_contrast_ratio:
        return colors[0]

    # Otherwise, return the color with the highest contrast ratio
    return colors[contrast_ratios.index(max(contrast_ratios))]


def luminance(color: ColorType) -> float:
    """Compute the relative luminance of a color using the WCAG 2.0 formula.

    Args:
        color (ColorType): RGB color tuple with values in [0, 1] or [0, 255], or a color
            string that can be converted to RGB.

    Returns:
        float: Relative luminance of the color in range [0, 1].
    """
    # Handle basic color strings
    color_map = {
        "black": (0, 0, 0),
        "white": (1, 1, 1),
        "red": (1, 0, 0),
        "green": (0, 1, 0),
        "blue": (0, 0, 1),
        "yellow": (1, 1, 0),
        "cyan": (0, 1, 1),
        "magenta": (1, 0, 1),
        "gray": (0.5, 0.5, 0.5),
        "grey": (0.5, 0.5, 0.5),
    }

    if isinstance(color, str):
        if color in color_map:
            r, g, b = color_map[color]
        elif color.startswith("#"):
            # Hex color
            color = color.lstrip("#")
            if len(color) == 3:
                r, g, b = tuple(int(color[i], 16) / 15 for i in range(3))
            elif len(color) == 6:
                r, g, b = tuple(int(color[i : i + 2], 16) / 255 for i in (0, 2, 4))
            else:
                raise ValueError(f"Invalid hex color: #{color}")
        elif color.startswith("rgb("):
            rgb_values = color.strip("rgb()").split(",")
            r, g, b = [float(x.strip()) for x in rgb_values[:3]]
            if r > 1 or g > 1 or b > 1:
                r, g, b = r / 255, g / 255, b / 255
        else:
            raise ValueError(f"Unsupported color format: {color}")
    elif isinstance(color, tuple) and len(color) >= 3:
        # Check if any value is > 1, indicating 0-255 range
        if any(c > 1 for c in color[:3]):
            r, g, b = color[0] / 255, color[1] / 255, color[2] / 255
        else:
            r, g, b = color[:3]
    else:
        raise ValueError(f"Unsupported color type: {type(color)}")

    def _convert_rgb_to_linear(rgb: float) -> float:
        """Convert an RGB value to linear RGB (remove gamma correction)."""
        return rgb / 12.92 if rgb <= 0.03928 else ((rgb + 0.055) / 1.055) ** 2.4

    # Convert RGB to linear RGB (remove gamma correction)
    r, g, b = map(_convert_rgb_to_linear, (r, g, b))

    # Calculate relative luminance using WCAG 2.0 coefficients
    return 0.2126 * r + 0.7152 * g + 0.0722 * b