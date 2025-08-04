"""
Helper utility functions for crystal structure processing
"""

import os
import warnings
from typing import Any, List, Sequence, Callable, Optional, Dict, Set, Tuple
from functools import lru_cache
import numpy as np

from pymatgen.core import Structure, PeriodicSite
from ..common_types import AnyStructure

def make_dir(target_path: str) -> None:
    """
    Create directory if it doesn't exist
    
    Args:
        target_path: Directory path to create
    """
    try:
        os.makedirs(target_path, exist_ok=True)
    except OSError as e:
        warnings.warn(f"Failed to create directory: {target_path}\n{e}")

@lru_cache(maxsize=1000)
def _get_dir_files_cached(directory: str) -> set:
    """
    Get cached set of files in directory for fast lookup
    
    Args:
        directory: Directory path
        
    Returns:
        Set of filenames in directory
    """
    try:
        return set(os.listdir(directory)) if os.path.exists(directory) else set()
    except (OSError, PermissionError):
        return set()

def check_name_available(input_name: str, result_dir: str = "result") -> bool:
    """
    Check if filename is available in directory
    
    Args:
        input_name: Filename to check
        result_dir: Directory to check in
        
    Returns:
        True if name is available
    """
    existing_files = _get_dir_files_cached(result_dir)
    return input_name not in existing_files

def batch_check_names_available(names: List[str], result_dir: str = "result") -> List[bool]:
    """
    Check multiple filenames availability efficiently
    
    Args:
        names: List of filenames to check
        result_dir: Directory to check in
        
    Returns:
        List of boolean availability results
    """
    existing_files = _get_dir_files_cached(result_dir)
    return [name not in existing_files for name in names]

def reserve_available_names(
    base_name: str, 
    count: int, 
    result_dir: str = "result", 
    fmt: str = "png", 
    numbering_rule: str = "000"
) -> List[str]:
    """
    Reserve multiple available filenames efficiently
    
    Args:
        base_name: Base filename
        count: Number of names to reserve
        result_dir: Target directory
        fmt: File extension
        numbering_rule: Numbering format
        
    Returns:
        List of available filenames
    """
    width = numbering_rule.count("0")
    existing_files = _get_dir_files_cached(result_dir)
    
    available_names = []
    i = 0
    while len(available_names) < count:
        name = f"{base_name}-{i:0{width}d}.{fmt}"
        if name not in existing_files:
            available_names.append(name)
        i += 1
    
    return available_names

def get_first_matching_site_prop(
    structures: Sequence[Structure],
    prop_keys: Sequence[str],
    *,
    warn_if_none: bool = True,
    filter_callback: Optional[Callable[[str, Any], bool]] = None,
) -> Optional[str]:
    """
    Find the first property key that exists in any structure's site properties
    
    Args:
        structures: Sequence of pymatgen Structures to check
        prop_keys: Property keys to look for
        warn_if_none: Whether to warn if no matching property found
        filter_callback: Function to filter property values
        
    Returns:
        First matching property key or None
    """
    for prop in prop_keys:
        for struct in structures:
            if hasattr(struct, 'site_properties') and prop in struct.site_properties:
                for site in struct:
                    if prop in site.properties:
                        value = site.properties[prop]
                        if filter_callback is None or filter_callback(prop, value):
                            return prop
            
            elif hasattr(struct, 'properties') and prop in struct.properties:
                value = struct.properties[prop]
                if filter_callback is None or filter_callback(prop, value):
                    return prop
    
    if prop_keys and warn_if_none:
        warnings.warn(
            f"None of {prop_keys=} found in any site or structure properties",
            UserWarning, stacklevel=2
        )
    
    return None

def get_struct_prop(
    struct: AnyStructure,
    struct_key: str,
    prop_name: str,
    default_value: Any
) -> Any:
    """
    Get property from structure with fallback to default
    
    Args:
        struct: Structure object
        struct_key: Structure identifier key
        prop_name: Property name to retrieve
        default_value: Default value if property not found
        
    Returns:
        Property value or default
    """
    if hasattr(struct, 'properties') and struct.properties:
        if prop_name in struct.properties:
            return struct.properties[prop_name]
    
    if hasattr(struct, 'info') and struct.info:
        if prop_name in struct.info:
            return struct.info[prop_name]
    
    if isinstance(default_value, dict) and struct_key in default_value:
        return default_value[struct_key]
    
    return default_value

@lru_cache(maxsize=500)
def get_site_symbol(site: PeriodicSite) -> str:
    """
    Get element symbol from a site, handling disordered sites
    
    Args:
        site: PeriodicSite object
        
    Returns:
        Element symbol string
    """
    if hasattr(site.species, 'elements'):
        el_amt_dict = site.species.get_el_amt_dict()
        if el_amt_dict:
            return max(el_amt_dict, key=el_amt_dict.get)
        if site.species.elements:
            return site.species.elements[0].symbol
        return "X"
    
    if hasattr(site.species, 'symbol'):
        return site.species.symbol
    
    if hasattr(site.species, 'element'):
        return site.species.element.symbol
    
    try:
        return site.species_string.split()[0]
    except (AttributeError, IndexError):
        return "X"

def get_unique_elements(structures: Sequence[Structure]) -> List[str]:
    """
    Get unique elements across all structures
    
    Args:
        structures: Sequence of structures
        
    Returns:
        List of unique element symbols
    """
    elements = set()
    for struct in structures:
        for site in struct:
            elements.add(get_site_symbol(site))
    return sorted(list(elements))

def calculate_structure_center(structure: Structure) -> np.ndarray:
    """
    Calculate the geometric center of a structure
    
    Args:
        structure: Crystal structure
        
    Returns:
        Center coordinates as numpy array
    """
    coords = np.array([site.coords for site in structure.sites])
    return np.mean(coords, axis=0)

def get_structure_bounds(structure: Structure) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box of structure coordinates
    
    Args:
        structure: Crystal structure
        
    Returns:
        Tuple of (min_coords, max_coords)
    """
    coords = np.array([site.coords for site in structure.sites])
    return np.min(coords, axis=0), np.max(coords, axis=0)

def filter_sites_by_element(structure: Structure, elements: List[str]) -> List[PeriodicSite]:
    """
    Filter sites by element symbols
    
    Args:
        structure: Crystal structure
        elements: List of element symbols to keep
        
    Returns:
        List of filtered sites
    """
    return [site for site in structure.sites if get_site_symbol(site) in elements]

def get_coordination_info(structure: Structure, site_index: int) -> Dict[str, Any]:
    """
    Get coordination information for a specific site
    
    Args:
        structure: Crystal structure
        site_index: Index of the site
        
    Returns:
        Dictionary with coordination information
    """
    try:
        from pymatgen.analysis.local_env import CrystalNN
        nn = CrystalNN()
        nn_info = nn.get_nn_info(structure, site_index)
        
        return {
            'coordination_number': len(nn_info),
            'neighbors': [
                {
                    'element': get_site_symbol(info['site']),
                    'distance': info.get('weight', 0),
                    'coords': info['site'].coords
                }
                for info in nn_info
            ]
        }
    except ImportError:
        return {'coordination_number': 0, 'neighbors': []}

def calculate_bond_distances(structure: Structure) -> List[Dict[str, Any]]:
    """
    Calculate all bond distances in structure
    
    Args:
        structure: Crystal structure
        
    Returns:
        List of bond information dictionaries
    """
    bonds = []
    try:
        from pymatgen.analysis.local_env import CrystalNN
        nn = CrystalNN()
        
        for i, site in enumerate(structure.sites):
            nn_info = nn.get_nn_info(structure, i)
            for neighbor in nn_info:
                bonds.append({
                    'site1_index': i,
                    'site1_element': get_site_symbol(site),
                    'site2_element': get_site_symbol(neighbor['site']),
                    'distance': neighbor.get('weight', 0),
                    'site1_coords': site.coords,
                    'site2_coords': neighbor['site'].coords
                })
    except ImportError:
        pass
    
    return bonds

def validate_structure(structure: Structure) -> List[str]:
    """
    Validate structure and return list of issues
    
    Args:
        structure: Crystal structure to validate
        
    Returns:
        List of validation warning/error messages
    """
    issues = []
    
    if len(structure.sites) == 0:
        issues.append("Structure has no sites")
    
    if structure.lattice.volume <= 0:
        issues.append("Structure has invalid lattice volume")
    
    coords = [site.coords for site in structure.sites]
    for i, coord1 in enumerate(coords):
        for j, coord2 in enumerate(coords[i+1:], i+1):
            if np.linalg.norm(np.array(coord1) - np.array(coord2)) < 0.1:
                issues.append(f"Sites {i} and {j} are very close (possible overlap)")
    
    return issues

def get_structure_statistics(structure: Structure) -> Dict[str, Any]:
    """
    Get statistical information about the structure
    
    Args:
        structure: Crystal structure
        
    Returns:
        Dictionary with structure statistics
    """
    elements = [get_site_symbol(site) for site in structure.sites]
    unique_elements = list(set(elements))
    
    return {
        'formula': structure.formula,
        'num_sites': len(structure.sites),
        'num_unique_elements': len(unique_elements),
        'elements': unique_elements,
        'element_counts': {elem: elements.count(elem) for elem in unique_elements},
        'lattice_volume': structure.lattice.volume,
        'lattice_parameters': {
            'a': structure.lattice.a,
            'b': structure.lattice.b, 
            'c': structure.lattice.c,
            'alpha': structure.lattice.alpha,
            'beta': structure.lattice.beta,
            'gamma': structure.lattice.gamma
        },
        'density': structure.density if hasattr(structure, 'density') else None
    }

def batch_process_structures(
    structures: List[Structure], 
    process_func: Callable[[Structure], Any],
    max_workers: Optional[int] = None
) -> List[Any]:
    """
    Process multiple structures in parallel
    
    Args:
        structures: List of structures to process
        process_func: Function to apply to each structure
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of processed results
    """
    from concurrent.futures import ThreadPoolExecutor
    
    if max_workers is None:
        max_workers = min(8, len(structures))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(process_func, structures))

def optimize_structure_list(structures: List[Structure]) -> List[Structure]:
    """
    Optimize structure list by removing duplicates and invalid structures
    
    Args:
        structures: List of structures
        
    Returns:
        Optimized list of unique valid structures
    """
    unique_structures = []
    seen_formulas = set()
    
    for struct in structures:
        try:
            formula = struct.formula
            if formula not in seen_formulas:
                issues = validate_structure(struct)
                if not issues:
                    unique_structures.append(struct)
                    seen_formulas.add(formula)
        except Exception:
            continue
    
    return unique_structures

def optimize_atom_size(
    structures: Dict[str, Structure],
    elem_radii: Dict[str, float],
    base_atom_size: float = 10,
    max_overlap: float = 0.2
) -> float:
    """
    Automatically adjust atom_size so that marker overlaps
    do not exceed max_overlap fraction.

    Args:
        structures: Dict of key->Structure to consider
        elem_radii: Mapping element->radius
        base_atom_size: Initial atom_size
        max_overlap: Maximum allowed overlap fraction
    Returns:
        Scaled atom_size
    """
    all_coords = []
    all_rs = []
    for struct in structures.values():
        for site in struct.sites:
            coord = np.array(site.coords)
            symbol = site.species_string
            r = elem_radii.get(symbol, 0.8)
            all_coords.append(coord)
            all_rs.append(r * base_atom_size)
    coords = np.array(all_coords)
    rs = np.array(all_rs)
    if len(rs) < 2:
        return base_atom_size
    diffs = coords[None, ...] - coords[:, None, ...]
    dists = np.linalg.norm(diffs, axis=-1)
    sum_r = rs[None, :] + rs[:, None]
    mask = ~np.eye(len(rs), dtype=bool)
    ratios = (dists[mask] / sum_r[mask])
    min_ratio = ratios.min() if ratios.size else 1.0
    scale = (min_ratio) / (1 + max_overlap)
    return base_atom_size * scale

__all__ = [
    "make_dir",
    "check_name_available", 
    "batch_check_names_available",
    "reserve_available_names",
    "get_first_matching_site_prop",
    "get_struct_prop",
    "get_site_symbol",
    "get_unique_elements",
    "calculate_structure_center",
    "get_structure_bounds",
    "filter_sites_by_element",
    "get_coordination_info",
    "calculate_bond_distances",
    "validate_structure",
    "get_structure_statistics",
    "batch_process_structures",
    "optimize_structure_list",
    "optimize_atom_size"
]
