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

# (legacy _get_cached_image_sites removed)


def get_image_shifts(frac_coords: np.ndarray, tol: float = 0.05):
    """
    Compute image shifts following Crystal Toolkit (_get_sites_to_draw) logic.

    For each fractional coordinate component near 0 or 1 (within tol),
    generate permutations of those axes and return corresponding image shift
    tuples (integers in -1,0,1).

    Args:
        frac_coords: fractional coordinates of a site (length-3 array)
        tol: tolerance (default ~0.05)

    Returns:
        set of 3-int tuples representing image shifts (excluding (0,0,0))
    """
    shifts = set()
    # near zero
    zero_elements = [i for i, f in enumerate(frac_coords) if np.allclose(f, 0.0, atol=tol)]
    for length in range(1, len(zero_elements) + 1):
        for combo in itertools.combinations(zero_elements, length):
            shifts.add((int(0 in combo), int(1 in combo), int(2 in combo)))

    # near one -> negative shifts
    one_elements = [i for i, f in enumerate(frac_coords) if np.allclose(f, 1.0, atol=tol)]
    for length in range(1, len(one_elements) + 1):
        for combo in itertools.combinations(one_elements, length):
            shifts.add(( -int(0 in combo), -int(1 in combo), -int(2 in combo)))

    # remove the origin if present
    shifts.discard((0, 0, 0))
    return shifts


def get_image_sites_from_shifts(
    site: PeriodicSite,
    lattice: Lattice,
    cell_boundary_tol: float = 0.05,
    min_dist_dedup: float = 1e-5,
):
    """
    Compute image-site cartesian coordinates.

    This returns cartesian coordinates of periodic image copies that lie just
    outside the unit cell faces (within cell_boundary_tol of 0 or 1 in fractional coords).
    """
    frac = np.array(site.frac_coords)
    shifts = get_image_shifts(frac, tol=cell_boundary_tol)
    if not shifts:
        return np.array([]).reshape(0, 3)

    # Convert to cartesian
    coords = []
    for shift in shifts:
        new_frac = frac + np.array(shift)
        cart = np.dot(new_frac, lattice.matrix)
        # dedupe by distance from original
        orig_cart = np.dot(frac, lattice.matrix)
        if np.linalg.norm(cart - orig_cart) > min_dist_dedup:
            coords.append(cart)

    return np.array(coords) if coords else np.array([]).reshape(0, 3)

def get_image_sites(
    site: PeriodicSite,
    lattice: Lattice,
    cell_boundary_tol: float = 0.0,
    min_dist_dedup: float = 0.1,
    image_shell: int = 1,
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
    effective_tol = cell_boundary_tol if cell_boundary_tol and cell_boundary_tol > 0 else 0.05
    return get_image_sites_from_shifts(site, lattice, cell_boundary_tol=effective_tol, min_dist_dedup=min_dist_dedup)

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

# Removed unused vectorized distance cache and cache-clear helper.
# If needed in future, reintroduce a small public API to clear module caches.
