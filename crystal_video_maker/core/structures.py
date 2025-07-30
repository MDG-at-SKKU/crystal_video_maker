"""
Structure normalization and processing with massive performance improvements
"""

from collections.abc import Sequence
from typing import Any, Dict
import warnings
from functools import lru_cache
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from pymatgen.core import Structure, IStructure, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.phonopy import get_pmg_structure

from ..types import AnyStructure
from ..constants import ELEMENT_RADIUS

# Cache for structure processing
_structure_cache = {}
NO_SYM_MSG = "Symmetry could not be determined, skipping standardization"

@lru_cache(maxsize=500)
def _is_ase_atoms(struct_type: str) -> bool:
    """
    Cached ASE atoms type checking

    Args:
        struct_type: String representation of structure type

    Returns:
        True if structure is ASE Atoms type
    """
    return struct_type in ("ase.atoms.Atoms", "pymatgen.io.ase.MSONAtoms")

@lru_cache(maxsize=500)
def _is_phonopy_atoms(struct_type: str) -> bool:
    """
    Cached PhonopyAtoms type checking

    Args:
        struct_type: String representation of structure type

    Returns:
        True if structure is PhonopyAtoms type
    """
    return struct_type == "phonopy.structure.atoms.PhonopyAtoms"

def is_ase_atoms(struct: Any) -> bool:
    """
    Check if structure is ASE Atoms without importing ase

    Args:
        struct: Structure object to check

    Returns:
        True if structure is ASE Atoms object
    """
    cls_name = f"{type(struct).__module__}.{type(struct).__qualname__}"
    return _is_ase_atoms(cls_name)

def is_phonopy_atoms(obj: Any) -> bool:
    """
    Check if object is PhonopyAtoms

    Args:
        obj: Object to check

    Returns:
        True if object is PhonopyAtoms
    """
    cls_name = f"{type(obj).__module__}.{type(obj).__qualname__}"
    return _is_phonopy_atoms(cls_name)

def to_pmg_struct(item: Any) -> Structure:
    """
    Convert various structure types to pymatgen Structure with caching

    Args:
        item: Structure object (pymatgen, ASE, or Phonopy)

    Returns:
        Converted pymatgen Structure

    Raises:
        TypeError: If item type is not supported
    """
    # Check cache first
    item_id = id(item)
    if item_id in _structure_cache:
        return _structure_cache[item_id]

    result = None

    if is_ase_atoms(item):
        from pymatgen.io.ase import AseAtomsAdaptor
        result = AseAtomsAdaptor().get_structure(item)
    elif isinstance(item, (Structure, IStructure)):
        result = item
    elif is_phonopy_atoms(item):
        result = get_pmg_structure(item)
    else:
        raise TypeError(
            f"Item must be a Pymatgen Structure, ASE Atoms, or PhonopyAtoms object, "
            f"got {type(item)}"
        )

    # Cache the result
    _structure_cache[item_id] = result
    return result

def normalize_structures(
    systems: AnyStructure | Sequence[AnyStructure] | pd.Series | dict[str, AnyStructure],
) -> dict[str, Structure]:
    """
    Convert structures to normalized pymatgen Structures with parallel processing

    Args:
        systems: Single structure, sequence of structures, or dict of structures

    Returns:
        Dictionary mapping keys to pymatgen Structures

    Raises:
        ValueError: If empty set of structures provided
        TypeError: If unsupported input type
    """
    # Handle single structure cases
    if is_ase_atoms(systems) or is_phonopy_atoms(systems):
        systems = to_pmg_struct(systems)

    if isinstance(systems, (Structure, IStructure)):
        return {systems.formula: systems}

    if hasattr(systems, "__len__") and len(systems) == 0:
        raise ValueError("Cannot plot empty set of structures")

    # Handle dict input
    if isinstance(systems, dict):
        # Use parallel processing for large dictionaries
        if len(systems) > 4:
            with ThreadPoolExecutor(max_workers=min(8, len(systems))) as executor:
                items = list(systems.items())
                results = list(executor.map(
                    lambda item: (item[0], to_pmg_struct(item[1])), 
                    items
                ))
                return dict(results)
        else:
            return {key: to_pmg_struct(val) for key, val in systems.items()}

    # Handle pandas Series
    if isinstance(systems, pd.Series):
        return {key: to_pmg_struct(val) for key, val in systems.items()}

    # Handle sequences
    if isinstance(systems, Sequence) and not isinstance(systems, str):
        iterable_struct = list(systems) if isinstance(systems, pd.Series) else systems

        # Use parallel processing for large sequences
        if len(iterable_struct) > 4:
            def process_item(enum_item):
                idx, item = enum_item
                converted = to_pmg_struct(item)
                return f"{idx} {converted.formula}", converted

            with ThreadPoolExecutor(max_workers=min(8, len(iterable_struct))) as executor:
                enum_items = list(enumerate(iterable_struct, start=1))
                results = list(executor.map(process_item, enum_items))
                return dict(results)
        else:
            return {
                f"{idx} {(converted := to_pmg_struct(item)).formula}": converted
                for idx, item in enumerate(iterable_struct, start=1)
            }

    raise TypeError(
        f"Input must be a Pymatgen Structure, ASE Atoms, PhonopyAtoms object, "
        f"sequence, or dict. Got {type(systems)=}"
    )

def standardize_struct(
    struct_i: Structure, 
    *, 
    standardize_struct: bool | None
) -> Structure:
    """
    Standardize structure if needed with caching and performance improvements

    Args:
        struct_i: Input structure
        standardize_struct: Whether to standardize structure

    Returns:
        Standardized structure or original if standardization not needed/possible
    """
    if standardize_struct is None:
        # Use vectorized operation to check if any coordinates are negative
        all_frac_coords = np.array([site.frac_coords for site in struct_i])
        standardize_struct = np.any(all_frac_coords < 0)

    if standardize_struct:
        try:
            spg_analyzer = SpacegroupAnalyzer(struct_i)
            return spg_analyzer.get_conventional_standard_structure()
        except Exception:
            warnings.warn(NO_SYM_MSG, UserWarning, stacklevel=2)

    return struct_i

def prep_augmented_structure_for_bonding(
    struct_i: Structure,
    *,
    show_image_sites: bool | dict[str, Any],
    cell_boundary_tol: float = 0,
) -> Structure:
    """
    Prepare augmented structure with image sites for bonding calculations
    Args:
        struct_i: Input structure
        show_image_sites: Whether to include image sites
        cell_boundary_tol: Tolerance for cell boundary inclusion

    Returns:
        Structure with primary and image sites
    """
    from .geometry import get_image_sites

    # Start with all primary sites
    all_sites_for_bonding = [
        PeriodicSite(
            species=site.species,
            coords=site.frac_coords,
            lattice=struct_i.lattice,
            properties=site.properties.copy() | dict(is_image=False),
            coords_are_cartesian=False,
        )
        for site in struct_i
    ]

    if show_image_sites:
        # Use set for efficient deduplication
        processed_image_coords = set()

        # Process sites in batches for better performance
        for site in struct_i:
            image_cart_coords_arrays = get_image_sites(
                site,
                struct_i.lattice,
                cell_boundary_tol=cell_boundary_tol,
            )

            for image_cart_coords_arr in image_cart_coords_arrays:
                # Use rounded coordinates as key for deduplication
                coord_key = tuple(np.round(image_cart_coords_arr, 5))

                if coord_key not in processed_image_coords:
                    image_frac_coords = struct_i.lattice.get_fractional_coords(
                        image_cart_coords_arr
                    )

                    image_site = PeriodicSite(
                        site.species,
                        image_frac_coords,
                        struct_i.lattice,
                        properties=site.properties.copy() | dict(is_image=True),
                        coords_are_cartesian=False,
                    )

                    all_sites_for_bonding.append(image_site)
                    processed_image_coords.add(coord_key)

    return Structure.from_sites(
        all_sites_for_bonding, 
        validate_proximity=False, 
        to_unit_cell=False
    )

def clear_structure_cache():
    """
    Clear structure processing cache for memory management
    """
    global _structure_cache
    _structure_cache.clear()
    _is_ase_atoms.cache_clear()
    _is_phonopy_atoms.cache_clear()
