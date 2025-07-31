"""
Utility functions for crystal structure visualization
"""

from .colors import (
    get_elem_colors,
    get_cached_color_scheme,
    get_default_colors,
    normalize_color,
    generate_color_palette,
    interpolate_colors,
    get_contrasting_text_color,
    color_map_type,
    get_color_scheme_names,
    validate_color_scheme,
    get_element_colors_for_structure,
    VESTA_COLORS,
    JMOL_COLORS,
    CPK_COLORS
)

from .helpers import (
    make_dir,
    check_name_available,
    batch_check_names_available,
    reserve_available_names,
    get_first_matching_site_prop,
    get_struct_prop,
    get_site_symbol,
    get_unique_elements,
    calculate_structure_center,
    get_structure_bounds,
    filter_sites_by_element,
    get_coordination_info,
    calculate_bond_distances,
    validate_structure,
    get_structure_statistics,
    batch_process_structures,
    optimize_structure_list
)

from .labels import (
    generate_site_label,
    get_site_hover_text,
    format_coordinate_text,
    create_element_legend,
    generate_structure_title
)

__all__ = [
    "get_elem_colors",
    "get_cached_color_scheme",
    "get_default_colors",
    "normalize_color",
    "generate_color_palette",
    "interpolate_colors",
    "get_contrasting_text_color",
    "color_map_type",
    "get_color_scheme_names",
    "validate_color_scheme",
    "get_element_colors_for_structure",
    "VESTA_COLORS",
    "JMOL_COLORS",
    "CPK_COLORS",
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
    "generate_site_label",
    "get_site_hover_text",
    "format_coordinate_text",
    "create_element_legend",
    "generate_structure_title"
]
