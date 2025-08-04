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

    # Convert numpy arrays to dict format that Plotly expects
    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

def _slab_camera(structures: dict[str, Structure]) -> dict:
    # view along c-axis for layered 3D materials
    struct = next(iter(structures.values()))
    normal = np.array([0,0,1])
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    span = max(struct.lattice.abc) * 0.5
    eye = center + normal * span

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

def _wire_camera(structures: dict[str, Structure]) -> dict:
    # view along a-axis for chain-like 3D materials
    struct = next(iter(structures.values()))
    normal = np.array([1,0,0])
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    span = max(struct.lattice.abc) * 0.5
    eye = center + normal * span

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

def _perovskite_camera(structures: dict[str, Structure]) -> dict:
    # view for perovskite-like cubic structures
    struct = next(iter(structures.values()))
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    # View from corner for best visibility of octahedra
    eye = center + np.array([1, 1, 1]) * max(struct.lattice.abc) * 0.8

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

def _spinel_camera(structures: dict[str, Structure]) -> dict:
    # view for spinel structures (cubic with tetrahedral/octahedral sites)
    struct = next(iter(structures.values()))
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    # View along [110] direction for spinel
    eye = center + np.array([1, 1, 0]) * max(struct.lattice.abc) * 0.8

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

def _wurtzite_camera(structures: dict[str, Structure]) -> dict:
    # view for wurtzite structures (hexagonal)
    struct = next(iter(structures.values()))
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    # View perpendicular to c-axis to show hexagonal symmetry
    eye = center + np.array([1, 0, 0]) * max(struct.lattice.abc) * 0.8

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

def _zincblende_camera(structures: dict[str, Structure]) -> dict:
    # view for zincblende structures (cubic, tetrahedral coordination)
    struct = next(iter(structures.values()))
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    # View along [111] to show tetrahedral coordination
    eye = center + np.array([1, 1, 1]) * max(struct.lattice.abc) * 0.8

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

def _rocksalt_camera(structures: dict[str, Structure]) -> dict:
    # view for rocksalt structures (cubic, octahedral coordination)
    struct = next(iter(structures.values()))
    coords = np.array([s.coords for s in struct.sites])
    center = coords.mean(axis=0)
    # View along [100] to show octahedral coordination
    eye = center + np.array([1, 0, 0]) * max(struct.lattice.abc) * 0.8

    return {
        "center": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}, 
        "eye": {"x": float(eye[0]), "y": float(eye[1]), "z": float(eye[2])}
    }

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
      - "perovskite": view for perovskite structures
      - "spinel": view for spinel structures
      - "wurtzite": view for wurtzite structures
      - "zincblende": view for zincblende structures
      - "rocksalt": view for rocksalt structures
      - any preset name for point-group views
    """
    if view_type == "layer":
        return _layer_camera(structures)
    elif view_type == "slab":
        return _slab_camera(structures)
    elif view_type == "wire":
        return _wire_camera(structures)
    elif view_type == "perovskite":
        return _perovskite_camera(structures)
    elif view_type == "spinel":
        return _spinel_camera(structures)
    elif view_type == "wurtzite":
        return _wurtzite_camera(structures)
    elif view_type == "zincblende":
        return _zincblende_camera(structures)
    elif view_type == "rocksalt":
        return _rocksalt_camera(structures)
    else:
        # fallback to named preset
        return _preset_camera(view_type)
