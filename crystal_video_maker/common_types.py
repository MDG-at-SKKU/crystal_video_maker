"""
Type aliases and basic type definitions for the crystal_video_maker package
"""

from typing import TypeAlias, Union

try:
    from pymatgen.core import Structure, IStructure
    from pymatgen.io.ase import MSONAtoms
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False

try:
    from ase import Atoms as AseAtoms
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False

# Basic type aliases
Xyz: TypeAlias = tuple[float, float, float]

# Structure type alias
if PYMATGEN_AVAILABLE:
    if ASE_AVAILABLE:
        AnyStructure: TypeAlias = Union[Structure, IStructure, MSONAtoms, "AseAtoms"]
    else:
        AnyStructure: TypeAlias = Union[Structure, IStructure, MSONAtoms]
else:
    if ASE_AVAILABLE:
        AnyStructure: TypeAlias = "AseAtoms"
    else:
        AnyStructure: TypeAlias = None

__all__ = ["Xyz", "AnyStructure"]
