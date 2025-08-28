"""
Rendering components for Rendering
Rendering functions for atoms, bonds, cells, and vectors
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import math
from functools import lru_cache
from abc import ABC, abstractmethod

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from .geometry import (
    Vector3D,
    calculate_distance,
    create_sphere_mesh,
    create_cylinder_mesh,
)
from .styling import Material, get_element_material, create_outline_material
from .config import ATOMIC_RADII, get_atomic_radius
from .utils import performance_monitor, timing_decorator


class BaseRenderer:
    """Base class for all renderer components"""

    def __init__(self, quality: str = "medium"):
        self.quality = quality
        self._cache = {}

    def render(self, *args, **kwargs) -> Dict[str, Any]:
        """Base render method - should be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement render method")

    def clear_cache(self):
        """Clear the internal cache"""
        self._cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached items"""
        return len(self._cache)


class AtomRenderer(BaseRenderer):
    """Atom rendering"""

    def __init__(self, quality: str = "medium"):
        self.quality = quality
        self._sphere_cache = {}

    def render(self, positions: List[Vector3D], elements: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Render atoms - implements base render method"""
        return self.render_atoms_batch(positions, elements, **kwargs)

    @lru_cache(maxsize=1000)
    def get_sphere_mesh(
        self, element: str, scale: float = 1.0, sphere_resolution: int = None
    ) -> Dict[str, Any]:
        """Get cached sphere mesh for element"""
        radius = get_atomic_radius(element, scale)
        if sphere_resolution is None:
            resolution = self._get_sphere_resolution(radius)
        else:
            resolution = sphere_resolution

        cache_key = f"{element}_{scale}_{resolution}"
        if cache_key not in self._sphere_cache:
            self._sphere_cache[cache_key] = create_sphere_mesh(radius, resolution)

        return self._sphere_cache[cache_key]

    def _get_sphere_resolution(self, radius: float) -> int:
        """Determine optimal sphere resolution based on radius"""
        if self.quality == "low":
            return 8
        elif self.quality == "medium":
            return 12
        elif self.quality == "high":
            return 16
        else:  # ultra
            return 24

    def render_atom(
        self,
        position: Vector3D,
        element: str,
        scale: float = 1.0,
        color_scheme: str = "vesta",
        sphere_resolution: int = None,
    ) -> Dict[str, Any]:
        """Render single atom"""
        mesh_data = self.get_sphere_mesh(element, scale, sphere_resolution)
        material = get_element_material(element, color_scheme)

        return {
            "type": "sphere",
            "position": position.to_list(),
            "mesh": mesh_data,
            "material": material.to_dict(),
            "element": element,
        }

    def render_atoms_batch(
        self,
        positions: List[Vector3D],
        elements: List[str],
        scale: float = 1.0,
        color_scheme: str = "vesta",
        sphere_resolution: int = None,
    ) -> List[Dict[str, Any]]:
        """Render multiple atoms efficiently"""
        atoms = []
        for pos, elem in zip(positions, elements):
            atom = self.render_atom(pos, elem, scale, color_scheme, sphere_resolution)
            atoms.append(atom)
        return atoms


class BondRenderer(BaseRenderer):
    """Bond rendering"""

    def __init__(self, quality: str = "medium"):
        self.quality = quality
        self._cylinder_cache = {}

    def render(self, bonds: List[Tuple[Vector3D, Vector3D, str]], **kwargs) -> List[Dict[str, Any]]:
        """Render bonds - implements base render method"""
        return self.render_bonds_batch(bonds, **kwargs)

    @lru_cache(maxsize=500)
    def get_bond_mesh(self, distance: float, radius: float = 0.1) -> Dict[str, Any]:
        """Get cached cylinder mesh for bond"""
        resolution = self._get_cylinder_resolution()

        cache_key = f"{distance}_{radius}_{resolution}"
        if cache_key not in self._cylinder_cache:
            self._cylinder_cache[cache_key] = create_cylinder_mesh(
                radius, distance, resolution
            )

        return self._cylinder_cache[cache_key]

    def _get_cylinder_resolution(self) -> int:
        """Determine optimal cylinder resolution"""
        if self.quality == "low":
            return 6
        elif self.quality == "medium":
            return 8
        elif self.quality == "high":
            return 12
        else:  # ultra
            return 16

    def render_bond(
        self,
        start_pos: Vector3D,
        end_pos: Vector3D,
        bond_type: str = "single",
        radius: float = 0.1,
    ) -> Dict[str, Any]:
        """Render single bond"""
        distance = calculate_distance(start_pos, end_pos)
        mesh_data = self.get_bond_mesh(distance, radius)

        # Calculate rotation to align cylinder with bond direction
        direction = end_pos - start_pos
        rotation = self._calculate_bond_rotation(direction)

        material = self._get_bond_material(bond_type)

        return {
            "type": "cylinder",
            "start_position": start_pos.to_list(),
            "end_position": end_pos.to_list(),
            "mesh": mesh_data,
            "rotation": rotation,
            "material": material.to_dict(),
            "bond_type": bond_type,
        }

    def _calculate_bond_rotation(self, direction: Vector3D) -> List[float]:
        """Calculate rotation angles for bond alignment"""
        length = direction.magnitude()
        if length == 0:
            return [0, 0, 0]

        dir_norm = direction.normalize()

        pitch = math.asin(dir_norm.z)
        yaw = math.atan2(dir_norm.y, dir_norm.x)

        return [pitch, yaw, 0]

    def _get_bond_material(self, bond_type: str) -> Material:
        """Get material for bond type"""
        if bond_type == "single":
            return Material(color=(0.5, 0.5, 0.5), opacity=1.0)
        elif bond_type == "double":
            return Material(color=(0.3, 0.3, 0.3), opacity=1.0)
        elif bond_type == "triple":
            return Material(color=(0.2, 0.2, 0.2), opacity=1.0)
        else:
            return Material(color=(0.4, 0.4, 0.4), opacity=1.0)

    def render_bonds_batch(
        self, bonds: List[Tuple[Vector3D, Vector3D, str]], radius: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Render multiple bonds efficiently"""
        bond_objects = []
        for start, end, bond_type in bonds:
            bond = self.render_bond(start, end, bond_type, radius)
            bond_objects.append(bond)
        return bond_objects


class CellRenderer(BaseRenderer):
    """Unit cell rendering"""

    def __init__(self, quality: str = "medium"):
        self.quality = quality

    def render(self, lattice_vectors: List[Vector3D], **kwargs) -> Dict[str, Any]:
        """Render unit cell - implements base render method"""
        return self.render_unit_cell(lattice_vectors, **kwargs)

    def render_unit_cell(
        self,
        lattice_vectors: List[Vector3D],
        show_faces: bool = True,
        show_edges: bool = True,
        face_opacity: float = 0.1,
    ) -> Dict[str, Any]:
        """Render unit cell"""
        # Calculate cell vertices
        vertices = self._calculate_cell_vertices(lattice_vectors)

        cell_data = {
            "type": "cell",
            "vertices": [v.to_list() for v in vertices],
            "show_faces": show_faces,
            "show_edges": show_edges,
            "face_opacity": face_opacity,
            "edge_material": create_outline_material((0.0, 0.0, 0.0), 2.0),
            "face_material": Material(color=(0.8, 0.8, 0.8), opacity=face_opacity),
        }

        return cell_data

    def _calculate_cell_vertices(
        self, lattice_vectors: List[Vector3D]
    ) -> List[Vector3D]:
        """Calculate unit cell vertices from lattice vectors"""
        if len(lattice_vectors) != 3:
            raise ValueError("Need exactly 3 lattice vectors")

        a, b, c = lattice_vectors

        vertices = [
            Vector3D(0, 0, 0),  # Origin
            a,  # a
            b,  # b
            a + b,  # a + b
            c,  # c
            a + c,  # a + c
            b + c,  # b + c
            a + b + c,  # a + b + c
        ]

        return vertices

    def render_supercell(
        self, lattice_vectors: List[Vector3D], nx: int = 2, ny: int = 2, nz: int = 2
    ) -> Dict[str, Any]:
        """Render supercell"""
        cells = []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    translation = (
                        lattice_vectors[0] * i
                        + lattice_vectors[1] * j
                        + lattice_vectors[2] * k
                    )

                    translated_vectors = [v + translation for v in lattice_vectors]

                    cell = self.render_unit_cell(translated_vectors)
                    cells.append(cell)

        return {"type": "supercell", "cells": cells, "dimensions": (nx, ny, nz)}


class VectorRenderer(BaseRenderer):
    """Vector/arrow rendering"""

    def __init__(self, quality: str = "medium"):
        self.quality = quality

    def render(self, vectors: List[Tuple[Vector3D, Vector3D]], **kwargs) -> List[Dict[str, Any]]:
        """Render vectors - implements base render method"""
        return self.render_vectors_batch(vectors, **kwargs)

    def render_vector(
        self,
        start_pos: Vector3D,
        direction: Vector3D,
        color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Render vector as arrow"""
        end_pos = start_pos + direction * scale

        return {
            "type": "arrow",
            "start_position": start_pos.to_list(),
            "end_position": end_pos.to_list(),
            "direction": direction.to_list(),
            "color": color,
            "scale": scale,
            "material": Material(color=color).to_dict(),
        }

    def render_vectors_batch(
        self,
        vectors: List[Tuple[Vector3D, Vector3D]],
        colors: Optional[List[Tuple[float, float, float]]] = None,
        scale: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Render multiple vectors"""
        if colors is None:
            colors = [(1.0, 0.0, 0.0)] * len(vectors)

        vector_objects = []
        for (start, direction), color in zip(vectors, colors):
            vector = self.render_vector(start, direction, color, scale)
            vector_objects.append(vector)

        return vector_objects


class LabelRenderer(BaseRenderer):
    """Text label rendering"""

    def __init__(self):
        self.labels = []

    def render(self) -> List[Dict[str, Any]]:
        """Render labels - implements base render method"""
        return self.render_labels()

    def add_label(
        self,
        position: Vector3D,
        text: str,
        color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        size: float = 12.0,
    ) -> None:
        """Add text label"""
        self.labels.append(
            {"position": position.to_list(), "text": text, "color": color, "size": size}
        )

    def render_labels(self) -> List[Dict[str, Any]]:
        """Get all labels for rendering"""
        return self.labels.copy()

    def clear_labels(self) -> None:
        """Clear all labels"""
        self.labels.clear()


# Global renderer instances
atom_renderer = AtomRenderer()
bond_renderer = BondRenderer()
cell_renderer = CellRenderer()
vector_renderer = VectorRenderer()
label_renderer = LabelRenderer()


# Convenience functions
def render_atom(
    position: Vector3D, element: str, scale: float = 1.0, color_scheme: str = "vesta"
) -> Dict[str, Any]:
    """Render single atom"""
    return atom_renderer.render_atom(position, element, scale, color_scheme)


def render_bond(
    start_pos: Vector3D,
    end_pos: Vector3D,
    bond_type: str = "single",
    radius: float = 0.1,
) -> Dict[str, Any]:
    """Render single bond"""
    return bond_renderer.render_bond(start_pos, end_pos, bond_type, radius)


def render_cell(lattice_vectors: List[Vector3D], **kwargs) -> Dict[str, Any]:
    """Render unit cell"""
    return cell_renderer.render_unit_cell(lattice_vectors, **kwargs)


def render_vector(start_pos: Vector3D, direction: Vector3D, **kwargs) -> Dict[str, Any]:
    """Render vector"""
    return vector_renderer.render_vector(start_pos, direction, **kwargs)


# Export functions
__all__ = [
    "BaseRenderer",
    "AtomRenderer",
    "BondRenderer",
    "CellRenderer",
    "VectorRenderer",
    "LabelRenderer",
    "render_atom",
    "render_bond",
    "render_cell",
    "render_vector",
    "atom_renderer",
    "bond_renderer",
    "cell_renderer",
    "vector_renderer",
    "label_renderer",
]
