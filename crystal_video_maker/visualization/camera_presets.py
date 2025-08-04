# visualization/camera_presets.py

"""
Camera presets and crystal-view helpers
"""

import numpy as np
from pymatgen.core import Structure

def _preset_camera(preset: str) -> dict:
    presets = {
        "isometric": {"eye": {"x":1.25,"y":1.25,"z":1.25}},
        "top":       {"eye": {"x":0,"y":0,"z":2.5}},
        "front":     {"eye": {"x":0,"y":-2.5,"z":0}},
        "side":      {"eye": {"x":2.5,"y":0,"z":0}},
        "x_axis":    {"eye": {"x":2.5,"y":0,"z":0}},
        "y_axis":    {"eye": {"x":0,"y":2.5,"z":0}},
        "z_axis":    {"eye": {"x":0,"y":0,"z":2.5}},
    }
    return presets.get(preset, presets["isometric"])

def _layer_camera(structures: dict[str, Structure]) -> dict:
    # view along the largest lattice face normal
    struct = next(iter(structures.values()))
    a,b,c = struct.lattice.abc
    areas = {"ab":a*b, "bc":b*c, "ca":c*a}
    face = max(areas, key=areas.get)
    normals = {"ab": [0,0,1], "bc":[1,0,0], "ca":[0,1,0]}
    normal = np.array(normals[face])
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    span = max(struct.lattice.abc)
    eye = center + normal * span
    return {"center": center.tolist(), "eye": eye.tolist()}

def _slab_camera(structures: dict[str, Structure]) -> dict:
    # view along c-axis for layered 3D materials
    struct = next(iter(structures.values()))
    normal = np.array([0,0,1])
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    span = max(struct.lattice.abc) * 0.5
    eye = center + normal * span
    return {"center": center.tolist(), "eye": eye.tolist()}

def _wire_camera(structures: dict[str, Structure]) -> dict:
    # view along a-axis for chain-like 3D materials
    struct = next(iter(structures.values()))
    normal = np.array([1,0,0])
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    span = max(struct.lattice.abc) * 0.5
    eye = center + normal * span
    return {"center": center.tolist(), "eye": eye.tolist()}

def _crystal_view_camera(
    structures: dict[str, Structure],
    view_type: str
) -> dict:
    """
    Choose camera based on crystal view type.

    view_type:
      - "layer": view normal to largest face
      - "slab": view along c-axis for layered materials
      - "wire": view along a-axis for chain materials
      - any preset name for point-group views
    """
    if view_type == "layer":
        return _layer_camera(structures)
    if view_type == "slab":
        return _slab_camera(structures)
    if view_type == "wire":
        return _wire_camera(structures)
    # fallback to named preset
    return _preset_camera(view_type)
