"""
Structure type handling and conversion functions
"""

from typing import Any, Dict, Optional, Union
from functools import lru_cache
import warnings

from ..common_types import AnyStructure

@lru_cache(maxsize=200)
def get_structure_type(struct: AnyStructure) -> str:
    """
    Get the type name of a structure object
    
    Args:
        struct: Structure object
        
    Returns:
        String identifier for structure type
    """
    return type(struct).__name__

def validate_structure_type(struct: AnyStructure) -> bool:
    """
    Validate if object is a supported structure type
    
    Args:
        struct: Object to validate
        
    Returns:
        True if valid structure type
    """
    try:
        from pymatgen.core import Structure, IStructure
        valid_types = (Structure, IStructure)
    except ImportError:
        valid_types = ()
    
    # Check for ASE atoms
    try:
        from ase import Atoms
        valid_types = valid_types + (Atoms,)
    except ImportError:
        pass
    
    # Check for MSONAtoms
    try:
        from pymatgen.io.ase import MSONAtoms
        valid_types = valid_types + (MSONAtoms,)
    except ImportError:
        pass
    
    return isinstance(struct, valid_types) if valid_types else False

def convert_structure_format(
    struct: AnyStructure,
    target_format: str = "pymatgen"
) -> Union[Any, AnyStructure]:
    """
    Convert structure to target format
    
    Args:
        struct: Input structure
        target_format: Target format (pymatgen, ase, etc.)
        
    Returns:
        Converted structure
    """
    if target_format.lower() == "pymatgen":
        try:
            from pymatgen.core import Structure, IStructure
            if isinstance(struct, (Structure, IStructure)):
                return struct
        except ImportError:
            pass
        
        # Try to convert from ASE
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            if hasattr(struct, 'get_positions'):  # ASE-like interface
                adaptor = AseAtomsAdaptor()
                return adaptor.get_structure(struct)
        except (ImportError, AttributeError):
            pass
        
        # Try other conversion methods
        if hasattr(struct, 'to_pymatgen'):
            return struct.to_pymatgen()
    
    elif target_format.lower() == "ase":
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            from pymatgen.core import Structure, IStructure
            if isinstance(struct, (Structure, IStructure)):
                adaptor = AseAtomsAdaptor()
                return adaptor.get_atoms(struct)
        except ImportError:
            warnings.warn("ASE not available for conversion")
    
    # Return original if conversion fails
    return struct

def get_structure_properties(struct: AnyStructure) -> Dict[str, Any]:
    """
    Extract properties from structure object
    
    Args:
        struct: Structure object
        
    Returns:
        Dictionary of structure properties
    """
    properties = {}
    
    # Common properties
    if hasattr(struct, 'formula'):
        properties['formula'] = struct.formula
    
    if hasattr(struct, 'lattice'):
        properties['lattice_volume'] = struct.lattice.volume
        properties['lattice_parameters'] = {
            'a': struct.lattice.a,
            'b': struct.lattice.b,
            'c': struct.lattice.c,
            'alpha': struct.lattice.alpha,
            'beta': struct.lattice.beta,
            'gamma': struct.lattice.gamma
        }
    
    if hasattr(struct, 'sites'):
        properties['num_sites'] = len(struct.sites)
    
    # Structure-specific properties
    if hasattr(struct, 'properties') and struct.properties:
        properties.update(struct.properties)
    
    # ASE-specific properties
    if hasattr(struct, 'info') and struct.info:
        properties.update(struct.info)
    
    return properties

__all__ = [
    "get_structure_type",
    "validate_structure_type", 
    "convert_structure_format",
    "get_structure_properties"
]
