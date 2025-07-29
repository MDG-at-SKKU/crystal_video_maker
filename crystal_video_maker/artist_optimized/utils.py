from typing import Sequence, Dict, Union, Literal
from enum import Enum
import numpy as np
from pymatgen.core import PeriodicSite

from __future__ import annotations
import re
from typing import Tuple
import plotly.colors as pcolors
from crystal_video_maker.artist_optimized.hands_optimized import get_spherical_wedge_mesh


LabelMode = Union[
    Literal["symbol", "species", "legend", False],
    Sequence[str],
    Dict[str, str],
]


_HEX_RE = re.compile(r"^#?([0-9A-Fa-f]{6})$")

def _hex_to_rgb_frac(hex_str: str) -> Tuple[float, float, float]:
    rgb = tuple(int(hex_str[i:i+2], 16)/255 for i in (0, 2, 4))
    return rgb  # (r,g,b) in 0‒1

def normalize_elem_color(raw_color: str | Tuple[float, float, float]) -> str:
    """
    Convert hex, named or tuple colours to Plotly-compatible 'rgba(r,g,b,a)' (0‒1 range).
    """
    if isinstance(raw_color, tuple):
        r, g, b = raw_color
    else:                                    # str input
        raw_color = raw_color.strip()
        # 1) hex (‘#FF00AA’ or ‘FF00AA’)
        m = _HEX_RE.match(raw_color)
        if m:
            r, g, b = _hex_to_rgb_frac(m.group(1))
        # 2) CSS/Plotly named colour
        elif raw_color in pcolors.PLOTLY_SCALES or raw_color in pcolors.COLORBREWER_SCALES:
            r, g, b = pcolors.hex_to_rgb(pcolors.get_named_colors_mapping()[raw_color])
            r, g, b = r/255, g/255, b/255
        else:  # fallback: let Plotly handle unknown strings
            return raw_color
    return f"rgba({r:.3f},{g:.3f},{b:.3f},1)"


class SiteCoords(str, Enum):
    cartesian = "cartesian"
    fractional = "fractional"
    cartesian_fractional = "cartesian_fractional"

def _coord_fmt(arr: np.ndarray, fmt: str | callable) -> str:
    if callable(fmt):         # 사용자가 함수로 직접 지정
        return ", ".join(fmt(v) for v in arr)
    if "." in fmt:            # 포맷 문자열 ('.4' 등)
        return ", ".join(f"{v:{fmt}f}" for v in arr)
    return ", ".join(str(v) for v in arr)


def get_site_hover_text(
    site: PeriodicSite,
    hover_text: SiteCoords | callable,
    majority_species,
    float_fmt: str | callable = ".4",
) -> str:
    """
    Build hover text for a site given display mode.
    """
    ct = ""
    if hover_text == SiteCoords.cartesian:
        ct = _coord_fmt(site.coords, float_fmt)
    elif hover_text == SiteCoords.fractional:
        ct = _coord_fmt(site.frac_coords, float_fmt)
    elif hover_text == SiteCoords.cartesian_fractional:
        ct = (
            f"Cartesian: {_coord_fmt(site.coords, float_fmt)}<br>"
            f"Fractional: {_coord_fmt(site.frac_coords, float_fmt)}"
        )
    elif callable(hover_text):         # user-provided
        ct = str(hover_text(site))
    else:
        ct = ""                        # hover OFF
    return f"{majority_species}: {ct}" if ct else str(majority_species)


def generate_site_label(
    site_labels: LabelMode,
    site_idx: int,
    site: PeriodicSite,
) -> str:
    """
    Determine what text (if any) appears on top of the atom marker.
    """
    # Case 1: labels disabled
    if site_labels is False:
        return ""

    # Case 2: explicit list/tuple
    if isinstance(site_labels, (list, tuple)):
        try:
            return str(site_labels[site_idx])
        except IndexError:
            return ""

    # Case 3: mapping {symbol -> label}
    if isinstance(site_labels, dict):
        return str(site_labels.get(site.specie.symbol, ""))

    # Case 4: preset modes
    if site_labels == "symbol":
        return site.specie.symbol
    if site_labels == "species":
        return str(site.specie)
    if site_labels == "legend":
        # leave legend handling to Plotly; return empty (avoids double text)
        return ""

    # Fallback: nothing
    return ""


def draw_disordered_site(
    fig,
    site: PeriodicSite,
    coords,
    site_idx: int,
    site_labels,
    elem_colors,
    atomic_radii,
    atom_size,
    scale,
    site_kwargs,
    *,
    is_image=False,
    is_3d=True,
    row=None,
    col=None,
    scene=None,
    hover_text=SiteCoords.cartesian_fractional,
    float_fmt=".4",
    legendgroup=None,
    showlegend=False,
    legend="legend",
    **kwargs,
):
    """
    Render a disordered (multi-species) site as a pie-style sphere.

    Pie wedge sizes reflect species occupancy fractions.
    Uses Mesh3d for 3-D or multiple Scatter traces for 2-D.
    """
    comps = site.species.as_dict()             # {"Fe":0.6,"Ni":0.4}
    total = sum(comps.values())
    # Pre-compute sphere radius once
    radius = atomic_radii[max(comps, key=comps.get)] * scale

    start_angle = 0.0
    for specie, occu in comps.items():
        if occu == 0:
            continue
        end_angle = start_angle + 2 * np.pi * (occu / total)
        wedge_color = normalize_elem_color(elem_colors.get(specie, "gray"))

        xs, ys, zs, i, j, k = get_spherical_wedge_mesh(
            np.asarray(coords), radius, start_angle, end_angle,
            n_theta=12, n_phi=18,
        )

        fig.add_mesh3d(
            x=xs, y=ys, z=zs,
            i=i, j=j, k=k,
            color=wedge_color,
            opacity=0.9 if is_image else 1,
            hoverinfo="skip",
            showlegend=False,
            scene=scene,
        )
        start_angle = end_angle

    # Optional text label (once, at sphere center)
    txt = generate_site_label(site_labels, site_idx, site)
    if txt:
        fig.add_scatter3d(
            x=[coords[0]], y=[coords[1]], z=[coords[2]],
            mode="text",
            text=[txt],
            textfont=dict(color="black", size=max(atom_size * radius, 10)),
            showlegend=False,
            hoverinfo="skip",
            scene=scene,
        )
