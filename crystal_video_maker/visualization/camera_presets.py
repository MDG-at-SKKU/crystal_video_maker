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


def _compute_cell_center(struct: Structure) -> np.ndarray:
    """
    Compute Cartesian center of the unit cell.
    """
    return struct.lattice.get_cartesian_coords([0.5, 0.5, 0.5])


def _compute_eye_position(struct: Structure, direction: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """
    Compute eye position given a view direction and scale factor.
    """
    center = _compute_cell_center(struct)
    # Convert fractional direction to Cartesian
    cart_dir = struct.lattice.get_cartesian_coords(direction) - struct.lattice.get_cartesian_coords([0, 0, 0])
    cart_dir /= np.linalg.norm(cart_dir)
    diag = np.linalg.norm(struct.lattice.matrix.diagonal())
    return center + cart_dir * diag * factor


def _layer_camera(structures: dict[str, Structure]) -> dict:
    # view normal to largest face
    struct = next(iter(structures.values()))
    a, b, c = struct.lattice.abc
    areas = {"ab": a * b, "bc": b * c, "ca": c * a}
    face = max(areas, key=areas.get)
    normals = {"ab": [0, 0, 1], "bc": [1, 0, 0], "ca": [0, 1, 0]}
    direction = np.array(normals[face])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _slab_camera(structures: dict[str, Structure]) -> dict:
    # view along c-axis
    struct = next(iter(structures.values()))
    direction = np.array([0, 0, 1])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _wire_camera(structures: dict[str, Structure]) -> dict:
    # view along a-axis
    struct = next(iter(structures.values()))
    direction = np.array([1, 0, 0])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _perovskite_camera(structures: dict[str, Structure]) -> dict:
    # view from corner
    struct = next(iter(structures.values()))
    direction = np.array([1, 1, 1])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _spinel_camera(structures: dict[str, Structure]) -> dict:
    # view along [110]
    struct = next(iter(structures.values()))
    direction = np.array([1, 1, 0])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _wurtzite_camera(structures: dict[str, Structure]) -> dict:
    # view perpendicular to c-axis
    struct = next(iter(structures.values()))
    direction = np.array([1, 0, 0])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _zincblende_camera(structures: dict[str, Structure]) -> dict:
    # view along [111]
    struct = next(iter(structures.values()))
    direction = np.array([1, 1, 1])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _rocksalt_camera(structures: dict[str, Structure]) -> dict:
    # view along [100]
    struct = next(iter(structures.values()))
    direction = np.array([1, 0, 0])

    center = _compute_cell_center(struct)
    eye = _compute_eye_position(struct, direction)

    return {
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "eye":    {"x": eye[0],    "y": eye[1],    "z": eye[2]},
    }


def _crystal_view_camera(
    structures: dict[str, Structure],
    view_type: str
) -> dict:
    """
    Choose camera based on crystal view type.

    view_type:
      - "layer":      view normal to largest face
      - "slab":       view along c-axis
      - "wire":       view along a-axis
      - "perovskite": corner view
      - "spinel":     [110] view
      - "wurtzite":   hexagonal c-axis view
      - "zincblende": [111] view
      - "rocksalt":   [100] view
      - fallback:     preset camera
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
        return _preset_camera(view_type)
