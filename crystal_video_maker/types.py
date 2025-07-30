"""
Type definitions for crystal video maker package
"""

from typing import Any, Self, TypeAlias, ParamSpec, TypeVar, Union, get_args
from pymatgen.core import IStructure, Structure
from pymatgen.io.ase import MSONAtoms
from ase import Atoms as AseAtoms

# Type aliases for coordinate and structure types
Xyz: TypeAlias = tuple[float, float, float]
AnyStructure: TypeAlias = Union[Structure, IStructure, MSONAtoms, "AseAtoms"]

# Type variables for generic functions
T = TypeVar('T')
P = ParamSpec('P')

# Color type for element coloring
ColorType: TypeAlias = Union[str, tuple[float, float, float]]
