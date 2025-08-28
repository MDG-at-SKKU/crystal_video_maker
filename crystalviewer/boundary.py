"""
Boundary and image site utilities for crystal structures
Handles periodic boundary conditions and image site generation
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Set
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def get_image_shifts(frac_coords: np.ndarray, tol: float = 0.05) -> Set[Tuple[int, int, int]]:
    """
    Get lattice shifts needed to create image sites for sites near cell boundaries

    Args:
        frac_coords: Fractional coordinates
        tol: Tolerance for boundary detection

    Returns:
        Set of (i,j,k) shift tuples
    """
    shifts = set()

    # Check each dimension for boundary proximity
    for i, coord in enumerate(frac_coords):
        if coord < tol:
            # Near lower boundary
            shift = [0, 0, 0]
            shift[i] = -1
            shifts.add(tuple(shift))
        elif coord > 1 - tol:
            # Near upper boundary
            shift = [0, 0, 0]
            shift[i] = 1
            shifts.add(tuple(shift))

    # Add corner/edge cases for 2D boundaries
    if len([c for c in frac_coords if c < tol or c > 1 - tol]) >= 2:
        for i in range(3):
            for j in range(i + 1, 3):
                if (frac_coords[i] < tol or frac_coords[i] > 1 - tol) and (
                    frac_coords[j] < tol or frac_coords[j] > 1 - tol
                ):
                    # Add diagonal shifts
                    for si in [-1, 1]:
                        for sj in [-1, 1]:
                            shift = [0, 0, 0]
                            shift[i] = si
                            shift[j] = sj
                            shifts.add(tuple(shift))

    # Remove the origin if present
    shifts.discard((0, 0, 0))
    return shifts


def get_image_sites_from_shifts(
    position: List[float],
    lattice_vectors: List[List[float]],
    cell_boundary_tol: float = 0.05,
    min_dist_dedup: float = 1e-5,
) -> np.ndarray:
    """
    Compute image-site cartesian coordinates for sites near cell boundaries

    Args:
        position: Cartesian position of the site
        lattice_vectors: Lattice vectors defining the unit cell
        cell_boundary_tol: Tolerance for boundary detection
        min_dist_dedup: Minimum distance to avoid duplicates

    Returns:
        Array of image site cartesian coordinates
    """
    if not HAS_NUMPY:
        return np.array([]).reshape(0, 3)

    # Convert cartesian to fractional coordinates
    lattice_matrix = np.array(lattice_vectors)
    frac_coords = np.dot(np.array(position), np.linalg.inv(lattice_matrix))

    # Get required shifts
    shifts = get_image_shifts(frac_coords, tol=cell_boundary_tol)
    if not shifts:
        return np.array([]).reshape(0, 3)

    # Convert to cartesian coordinates
    coords = []
    for shift in shifts:
        new_frac = frac_coords + np.array(shift)
        cart = np.dot(new_frac, lattice_matrix)

        # Dedupe by distance from original
        if np.linalg.norm(cart - np.array(position)) > min_dist_dedup:
            coords.append(cart)

    return np.array(coords) if coords else np.array([]).reshape(0, 3)


def add_image_sites_static(
    positions: List[List[float]],
    elements: List[str],
    lattice_vectors: List[List[float]],
    **kwargs
) -> Tuple[Optional[List[List[float]]], Optional[List[str]]]:
    """
    Add image sites for atoms near cell boundaries (static version)

    Args:
        positions: Atomic positions
        elements: Element symbols
        lattice_vectors: Lattice vectors
        **kwargs: Additional options

    Returns:
        Tuple of (image_positions, image_elements) or (None, None)
    """
    if not HAS_NUMPY:
        return None, None

    cell_boundary_tol = kwargs.get("cell_boundary_tol", 0.05)
    image_positions = []
    image_elements = []

    for pos, elem in zip(positions, elements):
        # Get image sites for this position
        image_sites = get_image_sites_from_shifts(
            pos, lattice_vectors, cell_boundary_tol
        )

        if len(image_sites) > 0:
            image_positions.extend(image_sites.tolist())
            image_elements.extend([elem] * len(image_sites))

    if image_positions:
        return image_positions, image_elements
    else:
        return None, None


def is_near_boundary(
    position: List[float],
    lattice_vectors: List[List[float]],
    tolerance: float = 0.05
) -> bool:
    """
    Check if a position is near any cell boundary

    Args:
        position: Cartesian position
        lattice_vectors: Lattice vectors
        tolerance: Boundary tolerance

    Returns:
        True if near boundary
    """
    if not HAS_NUMPY:
        return False

    # Convert to fractional coordinates
    lattice_matrix = np.array(lattice_vectors)
    frac_coords = np.dot(np.array(position), np.linalg.inv(lattice_matrix))

    # Check if any coordinate is near boundary
    for coord in frac_coords:
        if coord < tolerance or coord > 1 - tolerance:
            return True

    return False


def apply_periodic_boundary_conditions(
    positions: List[List[float]],
    lattice_vectors: List[List[float]]
) -> List[List[float]]:
    """
    Apply periodic boundary conditions to positions

    Args:
        positions: Atomic positions
        lattice_vectors: Lattice vectors

    Returns:
        Positions wrapped within unit cell
    """
    if not HAS_NUMPY:
        return positions

    lattice_matrix = np.array(lattice_vectors)
    wrapped_positions = []

    for pos in positions:
        # Convert to fractional coordinates
        frac_coords = np.dot(np.array(pos), np.linalg.inv(lattice_matrix))

        # Wrap fractional coordinates to [0, 1)
        wrapped_frac = frac_coords - np.floor(frac_coords)

        # Convert back to cartesian
        cart_pos = np.dot(wrapped_frac, lattice_matrix)
        wrapped_positions.append(cart_pos.tolist())

    return wrapped_positions


def get_minimum_image_distance(
    pos1: List[float],
    pos2: List[float],
    lattice_vectors: List[List[float]]
) -> float:
    """
    Calculate minimum image distance considering periodic boundary conditions

    Args:
        pos1: First position
        pos2: Second position
        lattice_vectors: Lattice vectors

    Returns:
        Minimum distance
    """
    if not HAS_NUMPY:
        # Fallback to simple distance
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    # Convert to fractional coordinates
    lattice_matrix = np.array(lattice_vectors)
    frac1 = np.dot(np.array(pos1), np.linalg.inv(lattice_matrix))
    frac2 = np.dot(np.array(pos2), np.linalg.inv(lattice_matrix))

    # Find minimum image
    diff = frac1 - frac2
    diff = diff - np.round(diff)  # Apply minimum image convention

    # Convert back to cartesian distance
    cart_diff = np.dot(diff, lattice_matrix)
    return np.linalg.norm(cart_diff)
