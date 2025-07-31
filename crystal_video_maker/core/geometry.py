"""
Geometric calculations with massive performance improvements through vectorization and caching
"""

import itertools
from functools import lru_cache
from typing import Dict, List
import numpy as np
from pymatgen.core import PeriodicSite, Lattice

from ..constants import ELEMENT_RADIUS, get_scaled_radii_dict
from ..common_types import Xyz

# Cache for expensive geometric calculations
_MIN_DIST_DEDUP = 0.1
_CELL_EDGE_OFFSETS = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
_CELL_EDGE_OFFSETS = _CELL_EDGE_OFFSETS[~np.all(_CELL_EDGE_OFFSETS == 0, axis=1)]

@lru_cache(maxsize=1000)
def get_atomic_radii(atomic_radii: float | dict[str, float] | None) -> dict[str, float]:
    """
    Get atomic radii with caching for massive performance improvement

    Args:
        atomic_radii: Scaling factor or custom radii mapping

    Returns:
        Dictionary mapping element symbols to atomic radii
    """
    if atomic_radii is None or isinstance(atomic_radii, float):
        scale = atomic_radii or 1.0
        return get_scaled_radii_dict(scale)
    return atomic_radii

@lru_cache(maxsize=2000)
def _get_cached_image_sites(
    site_coords_tuple: tuple,
    lattice_matrix_tuple: tuple, 
    cell_boundary_tol: float,
    min_dist_dedup: float
) -> tuple:
    """
    Cached image site calculation for massive performance improvement

    Args:
        site_coords_tuple: Site fractional coordinates as tuple
        lattice_matrix_tuple: Lattice matrix as flattened tuple
        cell_boundary_tol: Cell boundary tolerance
        min_dist_dedup: Minimum distance for deduplication

    Returns:
        Tuple of image site coordinates
    """
    # Reconstruct arrays from tuples
    site_frac_coords = np.array(site_coords_tuple)
    lattice_matrix = np.array(lattice_matrix_tuple).reshape(3, 3)

    # Vectorized calculation of all potential image sites
    offsets = _CELL_EDGE_OFFSETS
    new_frac_coords = site_frac_coords + offsets

    # Vectorized boundary check
    within_bounds = np.all(
        (new_frac_coords >= -cell_boundary_tol) & 
        (new_frac_coords <= 1 + cell_boundary_tol), 
        axis=1
    )

    valid_frac_coords = new_frac_coords[within_bounds]

    if len(valid_frac_coords) == 0:
        return tuple()

    # Convert to cartesian coordinates efficiently
    valid_cart_coords = np.dot(valid_frac_coords, lattice_matrix.T)

    # Vectorized distance calculation for deduplication
    original_cart = np.dot(site_frac_coords, lattice_matrix.T)
    distances = np.linalg.norm(valid_cart_coords - original_cart, axis=1)

    # Filter by minimum distance
    far_enough = distances > min_dist_dedup
    final_coords = valid_cart_coords[far_enough]

    # Return as tuple for caching
    return tuple(tuple(coord) for coord in final_coords)

def get_image_sites(
    site: PeriodicSite,
    lattice: Lattice,
    cell_boundary_tol: float = 0.0,
    min_dist_dedup: float = 0.1,
) -> np.ndarray:
    """
    Get image sites with massive performance improvements through caching and vectorization

    Args:
        site: Periodic site to get images for
        lattice: Crystal lattice
        cell_boundary_tol: Distance beyond unit cell to include atoms
        min_dist_dedup: Minimum distance to avoid duplicates

    Returns:
        Array of image site cartesian coordinates
    """
    # Convert to hashable types for caching
    site_coords_tuple = tuple(site.frac_coords)
    lattice_matrix_tuple = tuple(lattice.matrix.flatten())

    # Use cached calculation
    cached_result = _get_cached_image_sites(
        site_coords_tuple,
        lattice_matrix_tuple,
        cell_boundary_tol,
        min_dist_dedup
    )

    # Convert back to numpy array
    return np.array([list(coord) for coord in cached_result]) if cached_result else np.array([]).reshape(0, 3)

@lru_cache(maxsize=500)
def calculate_distances_vectorized(
    coords1_tuple: tuple,
    coords2_tuple: tuple
) -> float:
    """
    Calculate distance between two coordinate sets with caching

    Args:
        coords1_tuple: First coordinate set as tuple
        coords2_tuple: Second coordinate set as tuple

    Returns:
        Euclidean distance
    """
    coords1 = np.array(coords1_tuple)
    coords2 = np.array(coords2_tuple)
    return float(np.linalg.norm(coords1 - coords2))

def clear_geometry_cache():
    """
    Clear geometry calculation cache for memory management
    """
    _get_cached_image_sites.cache_clear()
    get_atomic_radii.cache_clear()
    calculate_distances_vectorized.cache_clear()
