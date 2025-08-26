"""
Structure processing and normalization functions
"""

from typing import Dict, List, Optional, Union, Sequence, Any
from concurrent.futures import ThreadPoolExecutor
import warnings

from pymatgen.core import Structure, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ..common_types import AnyStructure
import numpy as np


def normalize_structures(
    struct: Union[AnyStructure, Dict[str, AnyStructure], Sequence[AnyStructure]],
    *,
    show_image_sites: bool = False,
    cell_boundary_tol: float = 0.0,
    image_shell: int = 1,
    standardize_struct: bool | None = None,
) -> Dict[str, Structure]:
    """
    Normalize input structures to consistent format and optionally compute image sites

    Args:
        struct: Single structure, dict of structures, or sequence of structures
        show_image_sites: Whether to compute and include image sites
        cell_boundary_tol: Tolerance for cell boundary inclusion
        standardize_struct: Whether to apply structure standardization

    Returns:
        Dictionary mapping keys to Structure objects (with image sites if requested)
    """
    # 1. Basic structure normalization
    if isinstance(struct, dict):
        normalized = {str(k): to_pmg_struct(v) for k, v in struct.items()}
    elif isinstance(struct, (list, tuple)):
        normalized = {str(i): to_pmg_struct(s) for i, s in enumerate(struct)}
    else:
        normalized = {"0": to_pmg_struct(struct)}

    # 2. Apply structure standardization if requested
    if standardize_struct:
        for key, structure in normalized.items():
            try:
                normalized[key] = standardize_struct(structure)
            except Exception as e:
                warnings.warn(f"Structure standardization failed for {key}: {e}")

    # 3. Compute and integrate image sites if requested
    if show_image_sites:
        normalized = _compute_image_sites_batch(
            normalized, cell_boundary_tol=cell_boundary_tol, image_shell=image_shell
        )

    return normalized


def _compute_image_sites_batch(
    structures: Dict[str, Structure], cell_boundary_tol: float = 0.0, image_shell: int = 1
) -> Dict[str, Structure]:
    """
    Batch compute image sites and integrate into structures
    Integrates logic from prep_augmented_structure_for_bonding
    """
    # Prefer boundary-shift selection to avoid exploding image-site counts.
    from ..core.geometry import get_image_sites_from_shifts as get_image_sites

    result = {}

    for key, struct in structures.items():
        # Base sites (is_image=False)
        all_sites = [
            PeriodicSite(
                species=site.species,
                coords=site.frac_coords,
                lattice=struct.lattice,
                properties=site.properties.copy() | {"is_image": False},
                coords_are_cartesian=False,
            )
            for site in struct
        ]

        # Compute and add image sites
        processed_image_coords = set()
        for site in struct:
            # CTK uses an implicit tolerance of ~0.05; if caller passed 0.0,
            # use the CTK default so boundary-adjacent sites are detected.
            effective_tol = cell_boundary_tol if cell_boundary_tol and cell_boundary_tol > 0 else 0.05
            image_cart_coords_arrays = get_image_sites(
                site,
                struct.lattice,
                cell_boundary_tol=effective_tol,
            )

            for image_cart_coords_arr in image_cart_coords_arrays:
                coord_key = tuple(np.round(image_cart_coords_arr, 5))

                if coord_key not in processed_image_coords:
                    image_frac_coords = struct.lattice.get_fractional_coords(
                        image_cart_coords_arr
                    )

                    image_site = PeriodicSite(
                        site.species,
                        image_frac_coords,
                        struct.lattice,
                        properties=site.properties.copy() | {"is_image": True},
                        coords_are_cartesian=False,
                    )

                    all_sites.append(image_site)
                    processed_image_coords.add(coord_key)

        # Create augmented structure
        result[key] = Structure.from_sites(
            all_sites, validate_proximity=False, to_unit_cell=False
        )

    return result


def to_pmg_struct(struct: AnyStructure) -> Structure:
    """
    Convert any structure type to pymatgen Structure

    Args:
        struct: Structure object of any supported type

    Returns:
        Pymatgen Structure object
    """
    if isinstance(struct, Structure):
        return struct

    try:
        from pymatgen.core import IStructure

        if isinstance(struct, IStructure):
            return Structure.from_sites(struct.sites)
    except ImportError:
        pass

    try:
        from pymatgen.io.ase import AseAtomsAdaptor

        if hasattr(struct, "get_positions"):
            adaptor = AseAtomsAdaptor()
            return adaptor.get_structure(struct)
    except (ImportError, AttributeError):
        pass

    try:
        from pymatgen.io.ase import MSONAtoms

        if isinstance(struct, MSONAtoms):
            adaptor = AseAtomsAdaptor()
            return adaptor.get_structure(struct)
    except ImportError:
        pass

    if hasattr(struct, "to_pymatgen"):
        return struct.to_pymatgen()

    if hasattr(struct, "get_structure"):
        return struct.get_structure()

    raise TypeError(f"Cannot convert {type(struct)} to pymatgen Structure")


def standardize_struct(struct: Structure) -> Structure:
    """
    Standardize crystal structure using spacegroup analysis

    Args:
        struct: Input structure

    Returns:
        Standardized structure
    """
    try:
        analyzer = SpacegroupAnalyzer(struct)
        return analyzer.get_conventional_standard_structure()
    except Exception as e:
        warnings.warn(f"Structure standardization failed: {e}")
        return struct


def batch_structure_standardization(
    structures: List[Structure],
    use_multiprocessing: bool = True,
    max_workers: Optional[int] = None,
) -> List[Structure]:
    """
    Apply standardization to multiple structures

    Args:
        structures: List of input structures
        use_multiprocessing: Whether to use parallel processing
        max_workers: Maximum number of worker processes

    Returns:
        List of standardized structures
    """
    if not use_multiprocessing or len(structures) == 1:
        return [standardize_struct(struct) for struct in structures]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(standardize_struct, structures))


def filter_structures_by_formula(
    structures: List[Structure], target_formula: str
) -> List[Structure]:
    """
    Filter structures by chemical formula

    Args:
        structures: List of input structures
        target_formula: Target chemical formula

    Returns:
        Filtered list of structures
    """
    return [s for s in structures if s.formula == target_formula]


def sort_structures_by_energy(
    structures: List[Structure], energy_key: str = "energy"
) -> List[Structure]:
    """
    Sort structures by energy property

    Args:
        structures: List of input structures
        energy_key: Property key for energy values

    Returns:
        Sorted list of structures (lowest energy first)
    """

    def get_energy(struct):
        try:
            return struct.site_properties.get(energy_key, float("inf"))
        except (AttributeError, KeyError):
            return float("inf")

    return sorted(structures, key=get_energy)


def merge_structures(structures: List[Structure]) -> Structure:
    """
    Merge multiple structures into a single structure

    Args:
        structures: List of structures to merge

    Returns:
        Merged structure
    """
    if not structures:
        raise ValueError("Cannot merge empty list of structures")

    if len(structures) == 1:
        return structures[0]

    # Use first structure as base
    base_struct = structures[0]
    all_sites = list(base_struct.sites)

    # Add sites from other structures
    for struct in structures[1:]:
        all_sites.extend(struct.sites)

    return Structure.from_sites(all_sites)


def create_supercell(
    struct: Structure, scaling_matrix: Union[int, List[int], np.ndarray]
) -> Structure:
    """
    Create supercell from input structure

    Args:
        struct: Input structure
        scaling_matrix: Scaling factors for supercell creation

    Returns:
        Supercell structure
    """
    if isinstance(scaling_matrix, int):
        scaling_matrix = [scaling_matrix] * 3

    return struct.make_supercell(scaling_matrix)


def remove_duplicate_structures(
    structures: List[Structure], tolerance: float = 1e-3
) -> List[Structure]:
    """
    Remove duplicate structures based on structural similarity

    Args:
        structures: List of input structures
        tolerance: Tolerance for similarity comparison

    Returns:
        List with duplicates removed
    """
    unique_structures = []

    for struct in structures:
        is_duplicate = False
        for unique_struct in unique_structures:
            try:
                # Simple lattice parameter comparison
                if (
                    abs(struct.lattice.a - unique_struct.lattice.a) < tolerance
                    and abs(struct.lattice.b - unique_struct.lattice.b) < tolerance
                    and abs(struct.lattice.c - unique_struct.lattice.c) < tolerance
                ):
                    is_duplicate = True
                    break
            except Exception:
                continue

        if not is_duplicate:
            unique_structures.append(struct)

    return unique_structures


def apply_strain_to_structure(
    struct: Structure, strain_matrix: np.ndarray
) -> Structure:
    """
    Apply strain to crystal structure

    Args:
        struct: Input structure
        strain_matrix: 3x3 strain matrix

    Returns:
        Strained structure
    """
    new_lattice = struct.lattice.matrix @ strain_matrix
    return Structure(
        lattice=new_lattice,
        species=[site.species for site in struct],
        coords=[site.frac_coords for site in struct],
        coords_are_cartesian=False,
    )


def get_structure_fingerprint(structure: Structure) -> Dict[str, Any]:
    """
    Generate fingerprint for structure identification

    Args:
        structure: Input structure

    Returns:
        Dictionary containing structure fingerprint
    """
    try:
        analyzer = SpacegroupAnalyzer(structure)
        spacegroup = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system()
    except Exception:
        spacegroup = None
        crystal_system = "unknown"

    lattice = structure.lattice

    return {
        "formula": structure.formula,
        "num_sites": len(structure.sites),
        "volume": lattice.volume,
        "density": structure.density,
        "lattice_abc": [lattice.a, lattice.b, lattice.c],
        "lattice_angles": [lattice.alpha, lattice.beta, lattice.gamma],
        "spacegroup": spacegroup,
        "crystal_system": crystal_system,
    }


__all__ = [
    "normalize_structures",
    "to_pmg_struct",
    "standardize_struct",
    "batch_structure_standardization",
    "filter_structures_by_formula",
    "sort_structures_by_energy",
    "merge_structures",
    "create_supercell",
    "remove_duplicate_structures",
    "apply_strain_to_structure",
    "get_structure_fingerprint",
]
