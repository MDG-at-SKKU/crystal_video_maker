"""
Structure parsing and conversion utilities
Handles conversion between different structure formats (pymatgen, ASE, arrays)
"""

from typing import List, Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    try:
        from pymatgen.core import Structure as PymatgenStructure
    except ImportError:
        PymatgenStructure = None

    try:
        from ase import Atoms
    except ImportError:
        Atoms = None

try:
    from pymatgen.core import Structure as PymatgenStructure
    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    PymatgenStructure = None

try:
    import ase
    from ase import Atoms
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Atoms = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def convert_structure_to_arrays(structure: Any) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """
    Convert various structure formats to position/element arrays

    Args:
        structure: Structure object (pymatgen Structure, ASE Atoms, or tuple)

    Returns:
        Tuple of (positions, elements, lattice_vectors)
    """
    if isinstance(structure, tuple) and len(structure) >= 2:
        # Handle positions, elements arrays
        positions = structure[0]
        elements = structure[1]
        lattice_vectors = structure[2] if len(structure) > 2 else None
        return positions, elements, lattice_vectors

    if HAS_PYMATGEN and isinstance(structure, PymatgenStructure):
        return _convert_pymatgen_to_arrays(structure)

    if HAS_ASE and isinstance(structure, Atoms):
        return _convert_ase_to_arrays(structure)

    # Try to handle as generic object with positions/elements attributes
    if hasattr(structure, 'positions') and hasattr(structure, 'elements'):
        positions = structure.positions
        elements = structure.elements
        lattice_vectors = getattr(structure, 'lattice_vectors', None)
        return positions, elements, lattice_vectors

    raise ValueError(f"Unsupported structure format: {type(structure)}")


def _convert_pymatgen_to_arrays(structure: Any) -> Tuple[List[List[float]], List[str], List[List[float]]]:
    """Convert pymatgen Structure to arrays"""
    positions = [site.coords.tolist() for site in structure.sites]
    elements = [site.species.elements[0].symbol for site in structure.sites]
    lattice_vectors = structure.lattice.matrix.tolist()
    return positions, elements, lattice_vectors


def _convert_ase_to_arrays(atoms: Any) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """Convert ASE Atoms to arrays"""
    positions = atoms.positions.tolist()
    elements = [atom.symbol for atom in atoms]
    lattice_vectors = atoms.cell.tolist() if atoms.cell is not None else None
    return positions, elements, lattice_vectors


def validate_structure_data(positions: List[List[float]], elements: List[str]) -> bool:
    """
    Validate structure data for consistency

    Args:
        positions: Atomic positions
        elements: Element symbols

    Returns:
        True if valid
    """
    if len(positions) != len(elements):
        return False

    if len(positions) == 0:
        return False

    # Check position dimensions
    for pos in positions:
        if len(pos) != 3:
            return False
        if not all(isinstance(coord, (int, float)) for coord in pos):
            return False

    # Check elements
    for elem in elements:
        if not isinstance(elem, str) or len(elem) == 0:
            return False

    return True


def get_structure_info(structure: Any) -> Dict[str, Any]:
    """
    Extract basic information from structure

    Args:
        structure: Structure object

    Returns:
        Dictionary with structure information
    """
    positions, elements, lattice_vectors = convert_structure_to_arrays(structure)

    info = {
        'num_atoms': len(positions),
        'elements': list(set(elements)),
        'element_counts': {elem: elements.count(elem) for elem in set(elements)},
        'has_lattice': lattice_vectors is not None
    }

    if lattice_vectors:
        info['lattice_vectors'] = lattice_vectors

    return info
