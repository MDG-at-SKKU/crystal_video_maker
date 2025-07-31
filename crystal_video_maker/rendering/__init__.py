"""
Rendering functions for crystal structure visualization components
"""

from .sites import (
    draw_site,
    draw_disordered_site,
    get_site_symbol,
    batch_draw_sites,
    get_site_marker_properties,
    format_site_hover_text,
    get_site_coordinates,
    get_site_properties_summary,
    filter_sites_by_criteria
)

from .bonds import (
    draw_bonds,
    calculate_bond_distances,
    get_bond_midpoint,
    get_bond_color,
    filter_bonds_by_distance,
    batch_draw_bonds,
    analyze_bond_statistics,
    create_bond_distance_histogram
)

from .cells import (
    draw_cell,
    draw_cell_faces,
    get_cell_vertices,
    draw_cell_edges,
    get_cell_face_centers,
    calculate_cell_volume,
    get_cell_parameters,
    draw_multiple_cells
)

from .vectors import (
    draw_vector,
    draw_arrow_3d,
    get_vector_components,
    batch_draw_vectors,
    normalize_vector_length,
    create_vector_field_visualization,
    analyze_vector_statistics,
    create_vector_magnitude_colormap
)

try:
    from .styling import (
        blend_colors,
        adjust_color_brightness,
        create_custom_colormap
    )
    _STYLING_AVAILABLE = True
except ImportError:
    _STYLING_AVAILABLE = False

__all__ = [
    "draw_site",
    "draw_disordered_site", 
    "get_site_symbol",
    "batch_draw_sites",
    "get_site_marker_properties",
    "format_site_hover_text",
    "get_site_coordinates",
    "get_site_properties_summary",
    "filter_sites_by_criteria",
    "draw_bonds",
    "calculate_bond_distances",
    "get_bond_midpoint",
    "get_bond_color",
    "filter_bonds_by_distance",
    "batch_draw_bonds",
    "analyze_bond_statistics", 
    "create_bond_distance_histogram",
    "draw_cell",
    "draw_cell_faces",
    "get_cell_vertices",
    "draw_cell_edges",
    "get_cell_face_centers",
    "calculate_cell_volume",
    "get_cell_parameters",
    "draw_multiple_cells",
    "draw_vector",
    "draw_arrow_3d",
    "get_vector_components",
    "batch_draw_vectors",
    "normalize_vector_length",
    "create_vector_field_visualization",
    "analyze_vector_statistics",
    "create_vector_magnitude_colormap"
]

if _STYLING_AVAILABLE:
    __all__.extend([
        "blend_colors", 
        "adjust_color_brightness", 
        "create_custom_colormap"
    ])
