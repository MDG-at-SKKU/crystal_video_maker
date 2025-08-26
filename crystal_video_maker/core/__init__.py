"""
Core functionality for crystal structure processing
"""

from .structures import (
    normalize_structures,
    standardize_struct,
    #prep_augmented_structure_for_bonding,
    batch_structure_standardization,
    to_pmg_struct,
    filter_structures_by_formula,
    sort_structures_by_energy,
    merge_structures,
    create_supercell,
    remove_duplicate_structures,
    apply_strain_to_structure,
    get_structure_fingerprint
)

from .geometry import (
    get_atomic_radii,
    get_image_sites,
    get_scaled_radii_dict
)

try:
    from .structure_types import (
        get_structure_type,
        validate_structure_type,
        convert_structure_format,
        get_structure_properties
    )
    _STRUCTURE_TYPES_AVAILABLE = True
except ImportError:
    _STRUCTURE_TYPES_AVAILABLE = False

__all__ = [
    "normalize_structures",
    "standardize_struct", 
    "batch_structure_standardization",
    "to_pmg_struct",
    "filter_structures_by_formula",
    "sort_structures_by_energy",
    "merge_structures",
    "create_supercell",
    "remove_duplicate_structures",
    "apply_strain_to_structure",
    "get_structure_fingerprint",
    "get_atomic_radii",
    "get_image_sites",
    "get_scaled_radii_dict"
]

if _STRUCTURE_TYPES_AVAILABLE:
    __all__.extend([
        "get_structure_type",
        "validate_structure_type",
        "convert_structure_format",
        "get_structure_properties"
    ])
