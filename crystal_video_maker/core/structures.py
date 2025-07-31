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

def normalize_structures(struct: Union[AnyStructure, Dict[str, AnyStructure], Sequence[AnyStructure]]) -> Dict[str, Structure]:
    """
    Normalize input structures to consistent format
    
    Args:
        struct: Single structure, dict of structures, or sequence of structures
        
    Returns:
        Dictionary mapping keys to Structure objects
    """
    if isinstance(struct, dict):
        return {str(k): to_pmg_struct(v) for k, v in struct.items()}
    elif isinstance(struct, (list, tuple)):
        return {str(i): to_pmg_struct(s) for i, s in enumerate(struct)}
    else:
        return {"0": to_pmg_struct(struct)}

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
        if hasattr(struct, 'get_positions'):
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
    
    if hasattr(struct, 'to_pymatgen'):
        return struct.to_pymatgen()
    
    raise ValueError(f"Cannot convert {type(struct)} to pymatgen Structure")

def standardize_struct(struct: Structure, standardize_struct: Optional[bool] = None) -> Structure:
    """
    Standardize crystal structure using spacegroup symmetry
    
    Args:
        struct: Input structure
        standardize_struct: Whether to standardize (None = auto-detect)
        
    Returns:
        Standardized structure
    """
    if standardize_struct is False:
        return struct
    
    if standardize_struct is None:
        if len(struct.sites) > 100:
            return struct
        standardize_struct = True
    
    if not standardize_struct:
        return struct
    
    try:
        analyzer = SpacegroupAnalyzer(struct, symprec=0.1)
        return analyzer.get_conventional_standard_structure()
    except Exception as e:
        warnings.warn(f"Structure standardization failed: {e}")
        return struct

def prep_augmented_structure_for_bonding(
    struct_i: Structure,
    *,
    show_image_sites: bool,
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
    from ..core.geometry import get_image_sites
    
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
        processed_image_coords = set()
        
        for site in struct_i:
            image_cart_coords_arrays = get_image_sites(
                site,
                struct_i.lattice,
                cell_boundary_tol=cell_boundary_tol,
            )
            
            for image_cart_coords_arr in image_cart_coords_arrays:
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

def batch_structure_standardization(
    structures: List[Structure],
    standardize_struct: bool = True,
    max_workers: Optional[int] = None
) -> List[Structure]:
    """
    Standardize multiple structures in parallel
    
    Args:
        structures: List of structures to standardize
        standardize_struct: Whether to apply standardization
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of standardized structures
    """
    if not standardize_struct:
        return structures
    
    if max_workers is None:
        max_workers = min(8, len(structures))
    
    def standardize_single(struct):
        return standardize_struct(struct, standardize_struct=True)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(standardize_single, structures))

def filter_structures_by_formula(structures: List[Structure], formulas: List[str]) -> List[Structure]:
    """
    Filter structures by chemical formula
    
    Args:
        structures: List of structures to filter
        formulas: List of allowed formulas
        
    Returns:
        Filtered list of structures
    """
    return [struct for struct in structures if struct.formula in formulas]

def sort_structures_by_energy(structures: List[Structure], energy_key: str = "energy") -> List[Structure]:
    """
    Sort structures by energy property
    
    Args:
        structures: List of structures to sort
        energy_key: Property key for energy values
        
    Returns:
        Sorted list of structures (lowest energy first)
    """
    def get_energy(struct):
        if hasattr(struct, 'properties') and energy_key in struct.properties:
            return struct.properties[energy_key]
        return float('inf')
    
    return sorted(structures, key=get_energy)

def merge_structures(structures: List[Structure], lattice_tolerance: float = 0.1) -> Structure:
    """
    Merge multiple structures into a single supercell
    
    Args:
        structures: List of structures to merge
        lattice_tolerance: Tolerance for lattice matching
        
    Returns:
        Merged structure
    """
    if not structures:
        raise ValueError("No structures provided")
    
    if len(structures) == 1:
        return structures[0]
    
    base_struct = structures[0]
    base_lattice = base_struct.lattice
    
    all_sites = list(base_struct.sites)
    
    for struct in structures[1:]:
        if not np.allclose(struct.lattice.matrix, base_lattice.matrix, atol=lattice_tolerance):
            warnings.warn("Lattice parameters differ between structures")
        
        for site in struct.sites:
            all_sites.append(site)
    
    return Structure.from_sites(all_sites, validate_proximity=False)

def create_supercell(structure: Structure, scaling_matrix: Union[int, List[int], np.ndarray]) -> Structure:
    """
    Create supercell from structure
    
    Args:
        structure: Input structure
        scaling_matrix: Scaling factors for supercell
        
    Returns:
        Supercell structure
    """
    if isinstance(scaling_matrix, int):
        scaling_matrix = [scaling_matrix] * 3
    elif len(scaling_matrix) == 1:
        scaling_matrix = scaling_matrix * 3
    
    scaling_matrix = np.array(scaling_matrix)
    
    if len(scaling_matrix) == 3:
        scaling_matrix = np.diag(scaling_matrix)
    
    supercell = structure.copy()
    supercell.make_supercell(scaling_matrix)
    
    return supercell

def remove_duplicate_structures(structures: List[Structure], tolerance: float = 0.1) -> List[Structure]:
    """
    Remove duplicate structures based on structure matching
    
    Args:
        structures: List of structures
        tolerance: Tolerance for structure matching
        
    Returns:
        List of unique structures
    """
    unique_structures = []
    
    for struct in structures:
        is_duplicate = False
        
        for unique_struct in unique_structures:
            try:
                if struct.matches(unique_struct, ltol=tolerance, stol=tolerance, angle_tol=5):
                    is_duplicate = True
                    break
            except Exception:
                continue
        
        if not is_duplicate:
            unique_structures.append(struct)
    
    return unique_structures

def apply_strain_to_structure(structure: Structure, strain_matrix: np.ndarray) -> Structure:
    """
    Apply strain to crystal structure
    
    Args:
        structure: Input structure
        strain_matrix: 3x3 strain tensor
        
    Returns:
        Strained structure
    """
    strained_struct = structure.copy()
    
    new_lattice_matrix = np.dot(strain_matrix, structure.lattice.matrix)
    
    from pymatgen.core import Lattice
    new_lattice = Lattice(new_lattice_matrix)
    
    strained_struct.lattice = new_lattice
    
    return strained_struct

def get_structure_fingerprint(structure: Structure) -> Dict[str, Any]:
    """
    Generate fingerprint for structure comparison
    
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
        'formula': structure.formula,
        'num_sites': len(structure.sites),
        'volume': lattice.volume,
        'density': structure.density,
        'lattice_abc': [lattice.a, lattice.b, lattice.c],
        'lattice_angles': [lattice.alpha, lattice.beta, lattice.gamma],
        'spacegroup': spacegroup,
        'crystal_system': crystal_system
    }

__all__ = [
    "normalize_structures",
    "to_pmg_struct",
    "standardize_struct",
    "prep_augmented_structure_for_bonding",
    "batch_structure_standardization",
    "filter_structures_by_formula",
    "sort_structures_by_energy",
    "merge_structures",
    "create_supercell",
    "remove_duplicate_structures",
    "apply_strain_to_structure",
    "get_structure_fingerprint"
]
