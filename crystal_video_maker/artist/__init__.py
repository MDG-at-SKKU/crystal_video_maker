# __init__.py

from typing import Any, Self, TypeAlias, ParamSpec, TypeVar, Union, get_args
from pymatgen.core import IStructure, Structure
from pymatgen.io.ase import MSONAtoms
from ase.atoms import Atoms as AseAtoms

Xyz: TypeAlias = tuple[float, float, float]
AnyStructure: TypeAlias = Union[Structure, IStructure, MSONAtoms, "AseAtoms"]