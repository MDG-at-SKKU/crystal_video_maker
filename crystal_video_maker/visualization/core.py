from typing import Literal, Any, cast, Sequence, Callable, Union, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import warnings

from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from pymatgen.core import PeriodicSite, Structure
from pymatgen.core.periodic_table import Element
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from ..common_types import AnyStructure
from ..core.structures import normalize_structures, standardize_struct as struct_standardizer  # 이름 변경
from ..core.geometry import get_atomic_radii
from ..utils.colors import get_elem_colors, get_default_colors
from ..common_enum import SiteCoords
from ..rendering.sites import get_site_symbol, format_site_hover_text
from ..rendering.bonds import draw_bonds
from ..rendering.cells import draw_cell
from ..rendering.vectors import draw_vector
from ..utils.helpers import get_first_matching_site_prop, get_struct_prop

@lru_cache(maxsize=32)
def _get_layout_config(n_structs: int, n_cols: int) -> dict:
    """
    Cached layout configuration for subplots
    
    Args:
        n_structs: Number of structures to plot
        n_cols: Number of columns for subplot grid
        
    Returns:
        Dictionary containing rows, cols, height, and width for subplot layout
    """
    n_rows = (n_structs - 1) // n_cols + 1
    return {
        "rows": n_rows,
        "cols": min(n_cols, n_structs),
        "height": 400 * n_rows,
        "width": 400 * min(n_cols, n_structs)
    }

def _structure_3d_single(
    struct: AnyStructure,
    *,
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 20,
    elem_colors: dict[str, str] = None,
    scale: float = 1,
    show_cell: bool | dict[str, Any] = True,
    show_cell_faces: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool | dict[str, Any] = True,
    cell_boundary_tol: float | dict[str, float] = 0.0,
    show_bonds: bool | NearNeighbors | dict[str, bool | NearNeighbors] = False,
    site_labels: Literal["symbol", "species", "legend", False] | dict[str, str] | Sequence[str] = "legend",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None | Literal[False] = None,
    show_subplot_titles: bool = False,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: SiteCoords | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    hover_float_fmt: str | Callable[[float], str] = ".4",
    bond_kwargs: dict[str, Any] | None = None,
    use_internal_threads: bool = True
) -> go.Figure:
    """
    Plot single or multiple pymatgen structures in 3D with Plotly
    
    Args:
        struct: Structure(s) to plot - can be single structure, dict, or sequence
        atomic_radii: Scaling factor for default radii or custom radii mapping
        atom_size: Scaling factor for atom marker sizes
        elem_colors: Element color scheme or custom color mapping dictionary
        scale: Overall scaling factor for plotted atoms and lines
        show_cell: Whether to render unit cell outline
        show_cell_faces: Whether to show transparent cell faces
        show_sites: Whether to plot atomic sites as markers
        show_image_sites: Whether to show image sites on cell boundaries
        cell_boundary_tol: Buffer distance beyond unit cell for including image atoms
        show_bonds: Whether to draw bonds between sites using nearest neighbor algorithm
        site_labels: How to annotate lattice sites (symbol/species/legend/False)
        standardize_struct: Whether to standardize the crystal structure
        n_cols: Number of columns for subplot grid layout
        subplot_title: Function to generate subplot titles or False to disable
        show_subplot_titles: Whether to show default subplot titles
        show_site_vectors: Vector properties to display as arrows (force/magmom)
        vector_kwargs: Customization options for vector arrow appearance
        hover_text: Controls hover tooltip template format
        hover_float_fmt: Float formatting string for hover coordinates
        bond_kwargs: Customization options for bond line appearance
        use_internal_threads: Whether to use internal threading for processing
        
    Returns:
        Plotly Figure object containing complete 3D structure visualization
    """
    structures = normalize_structures(struct)
    n_structs = len(structures)
    
    layout_config = _get_layout_config(n_structs, n_cols)
    n_cols = layout_config["cols"]
    n_rows = layout_config["rows"]
    
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=[" " for _ in range(n_structs)],
    )
    
    _atomic_radii = get_atomic_radii(atomic_radii)
    _elem_colors = get_elem_colors(elem_colors) if elem_colors else get_default_colors()
    
    if isinstance(show_site_vectors, str):
        show_site_vectors = [show_site_vectors]
    
    vector_prop = get_first_matching_site_prop(
        list(structures.values()),
        show_site_vectors,
        warn_if_none=show_site_vectors != ("force", "magmom"),
        filter_callback=lambda _prop, value: (np.array(value).shape or [None])[-1] == 3,
    )
    
    seen_elements_per_subplot = {}
    
    def process_structure(args):
        idx, (struct_key, raw_struct_i) = args
        return _process_single_structure(
            idx, struct_key, raw_struct_i, fig, _atomic_radii, _elem_colors,
            atom_size, scale, show_sites, show_image_sites, show_bonds,
            show_cell, show_cell_faces, site_labels, cell_boundary_tol,
            standardize_struct, vector_prop, vector_kwargs,
            hover_text, hover_float_fmt, bond_kwargs, subplot_title,
            show_subplot_titles,
            seen_elements_per_subplot
        )
    
    structure_items = list(enumerate(structures.items(), start=1))
    
    if use_internal_threads:
        with ThreadPoolExecutor(max_workers=min(8, n_structs)) as executor:
            list(executor.map(process_structure, structure_items))
    else:
        for item in structure_items:
            process_structure(item)
    
    _configure_3d_scenes(fig, n_structs, n_cols, n_rows, site_labels)
    
    fig.layout.height = layout_config["height"]
    fig.layout.width = layout_config["width"]
    fig.layout.showlegend = site_labels == "legend"
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.margin = dict(l=0, r=0, t=30, b=0)
    
    return fig

def structure_3d(
    struct: AnyStructure | dict[str, AnyStructure] | Sequence[AnyStructure],
    *,
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 20,
    elem_colors: dict[str, str] = None,
    scale: float = 1,
    show_cell: bool | dict[str, Any] = True,
    show_cell_faces: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool | dict[str, Any] = True,
    cell_boundary_tol: float | dict[str, float] = 0.0,
    show_bonds: bool | NearNeighbors | dict[str, bool | NearNeighbors] = False,
    site_labels: Literal["symbol", "species", "legend", False] | dict[str, str] | Sequence[str] = "legend",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None | Literal[False] = None,
    show_subplot_titles: bool = False,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: SiteCoords | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    hover_float_fmt: str | Callable[[float], str] = ".4",
    bond_kwargs: dict[str, Any] | None = None,
    use_internal_threads: bool = True,
    return_subplots_as_list: bool = True
) -> Union[go.Figure, List[go.Figure]]:
    """
    Plot pymatgen structures in 3D with Plotly
    
    Args:
        struct: Structure(s) to plot - can be single structure, dict, or sequence
        atomic_radii: Scaling factor for default radii or custom radii mapping
        atom_size: Scaling factor for atom marker sizes
        elem_colors: Element color scheme or custom color mapping dictionary
        scale: Overall scaling factor for plotted atoms and lines
        show_cell: Whether to render unit cell outline
        show_cell_faces: Whether to show transparent cell faces
        show_sites: Whether to plot atomic sites as markers
        show_image_sites: Whether to show image sites on cell boundaries
        cell_boundary_tol: Buffer distance beyond unit cell for including image atoms
        show_bonds: Whether to draw bonds between sites using nearest neighbor algorithm
        site_labels: How to annotate lattice sites (symbol/species/legend/False)
        standardize_struct: Whether to standardize the crystal structure
        n_cols: Number of columns for subplot grid layout
        subplot_title: Function to generate subplot titles or False to disable
        show_subplot_titles: Whether to show default subplot titles
        show_site_vectors: Vector properties to display as arrows (force/magmom)
        vector_kwargs: Customization options for vector arrow appearance
        hover_text: Controls hover tooltip template format
        hover_float_fmt: Float formatting string for hover coordinates
        bond_kwargs: Customization options for bond line appearance
        use_internal_threads: Whether to use internal threading for processing
        return_subplots_as_list: Whether to return list of individual figures
        
    Returns:
        Single Figure for single structure or when return_subplots_as_list=False
        List of Figures when return_subplots_as_list=True or auto-detected for multiple structures
    """
    
    # Auto-detect behavior if not explicitly set
    if return_subplots_as_list is None:
        if isinstance(struct, Sequence) and not isinstance(struct, dict) and not isinstance(struct, str):
            # Return list for sequences with multiple items, single Figure for single item
            return_subplots_as_list = len(struct) > 1
        else:
            # Single structure or dict - return single Figure
            return_subplots_as_list = False
    
    # Return list of individual figures if requested and input is a sequence
    if return_subplots_as_list and isinstance(struct, Sequence) and not isinstance(struct, dict) and not isinstance(struct, str):
        figs = []
        for s in struct:
            fig = _structure_3d_single(
                s,
                atomic_radii=atomic_radii,
                atom_size=atom_size,
                elem_colors=elem_colors,
                scale=scale,
                show_cell=show_cell,
                show_cell_faces=show_cell_faces,
                show_sites=show_sites,
                show_image_sites=show_image_sites,
                cell_boundary_tol=cell_boundary_tol,
                show_bonds=show_bonds,
                site_labels=site_labels,
                standardize_struct=standardize_struct,
                n_cols=1,
                subplot_title=subplot_title,
                show_subplot_titles=show_subplot_titles,
                show_site_vectors=show_site_vectors,
                vector_kwargs=vector_kwargs,
                hover_text=hover_text,
                hover_float_fmt=hover_float_fmt,
                bond_kwargs=bond_kwargs,
                use_internal_threads=use_internal_threads
            )
            figs.append(fig)
        return figs
    
    # Return single figure with subplots or single structure
    return _structure_3d_single(
        struct,
        atomic_radii=atomic_radii,
        atom_size=atom_size,
        elem_colors=elem_colors,
        scale=scale,
        show_cell=show_cell,
        show_cell_faces=show_cell_faces,
        show_sites=show_sites,
        show_image_sites=show_image_sites,
        cell_boundary_tol=cell_boundary_tol,
        show_bonds=show_bonds,
        site_labels=site_labels,
        standardize_struct=standardize_struct,
        n_cols=n_cols,
        subplot_title=subplot_title,
        show_subplot_titles=show_subplot_titles,
        show_site_vectors=show_site_vectors,
        vector_kwargs=vector_kwargs,
        hover_text=hover_text,
        hover_float_fmt=hover_float_fmt,
        bond_kwargs=bond_kwargs,
        use_internal_threads=use_internal_threads
    )

def _process_single_structure(
    idx, struct_key, raw_struct_i, fig, atomic_radii, elem_colors,
    atom_size, scale, show_sites, show_image_sites, show_bonds,
    show_cell, show_cell_faces, site_labels, cell_boundary_tol,
    standardize_struct_param, vector_prop, vector_kwargs,  # 매개변수 이름 변경
    hover_text, hover_float_fmt, bond_kwargs, subplot_title,
    show_subplot_titles,
    seen_elements_per_subplot
):
    """
    Process and render a single crystal structure with all visualization options
    
    Args:
        idx: Structure index for subplot positioning
        struct_key: Structure identifier key
        raw_struct_i: Raw structure object to process
        fig: Plotly figure object to add traces to
        atomic_radii: Dictionary of atomic radii for elements
        elem_colors: Dictionary of element colors
        atom_size: Scaling factor for atom marker sizes
        scale: Overall scaling factor
        show_sites: Whether to display atomic sites
        show_image_sites: Whether to display image sites
        show_bonds: Whether to display chemical bonds
        show_cell: Whether to display unit cell
        show_cell_faces: Whether to display cell faces
        site_labels: Site labeling configuration
        cell_boundary_tol: Cell boundary tolerance for image sites
        standardize_struct_param: Whether to standardize structure
        vector_prop: Vector property to display
        vector_kwargs: Vector display customization
        hover_text: Hover text configuration
        hover_float_fmt: Float formatting for hover text
        bond_kwargs: Bond display customization
        subplot_title: Subplot title configuration
        show_subplot_titles: Whether to show default subplot titles
        seen_elements_per_subplot: Dictionary tracking seen elements for legend
        
    Returns:
        None (modifies fig object in-place)
    """
    # Standardize structure using imported function
    struct_i = struct_standardizer(raw_struct_i, standardize_struct=standardize_struct_param)
    
    # Manage legend elements
    seen_elements_per_subplot[idx] = set()
    
    # Handle cell boundary tolerance
    cell_boundary_tol_i = get_struct_prop(
        raw_struct_i, struct_key, "cell_boundary_tol", cell_boundary_tol
    ) or 0.0
    
    # Prepare augmented structure including image sites
    from ..core.structures import prep_augmented_structure_for_bonding
    augmented_structure = prep_augmented_structure_for_bonding(
        struct_i,
        show_image_sites=show_image_sites and show_sites,
        cell_boundary_tol=cell_boundary_tol_i,
    )
    
    # Draw sites
    if show_sites:
        _plot_sites_optimized(
            fig, augmented_structure, struct_i, idx, atomic_radii,
            elem_colors, atom_size, scale, site_labels, 
            seen_elements_per_subplot[idx], hover_text, hover_float_fmt
        )
    
    # Draw vectors
    if vector_prop:
        _plot_vectors(
            fig, struct_i, vector_prop, vector_kwargs, idx
        )
    
    # Draw bonds
    if show_bonds:
        _plot_bonds(
            fig, augmented_structure, show_bonds, struct_key, idx,
            bond_kwargs, elem_colors, show_sites
        )
    
    # Draw cell
    if show_cell:
        draw_cell(
            fig, struct_i,
            cell_kwargs={} if show_cell is True else show_cell,
            is_3d=True, scene=f"scene{idx}",
            show_faces=show_cell_faces
        )
    
    # Set subplot title
    # If subplot_title is provided and not False, show it regardless of show_subplot_titles
    if subplot_title is not None and subplot_title is not False:
        # If subplot_title is None but show_subplot_titles=True, show default title
        _set_subplot_title(fig, struct_i, struct_key, idx, subplot_title)
    elif show_subplot_titles and subplot_title is not False:
        # Use default title generation when show_subplot_titles=True
        _set_subplot_title(fig, struct_i, struct_key, idx, None)


def _plot_sites_optimized(
    fig, augmented_structure, struct_i, idx, atomic_radii,
    elem_colors, atom_size, scale, site_labels, seen_elements,
    hover_text, hover_float_fmt
):
    """
    Render atomic sites with batch processing
    
    Args:
        fig: Plotly figure object to add site traces to
        augmented_structure: Structure with primary and image sites
        struct_i: Original structure for reference
        idx: Subplot index for scene identification
        atomic_radii: Dictionary of atomic radii
        elem_colors: Dictionary of element colors
        atom_size: Scaling factor for atom sizes
        scale: Overall scaling factor
        site_labels: Site labeling configuration
        seen_elements: Set tracking elements already added to legend
        hover_text: Hover text configuration
        hover_float_fmt: Float formatting for coordinates
        
    Returns:
        None (modifies fig object in-place)
    """
    sites_by_element = {}
    # Group sites by element, separating primary and image sites
    for site_idx, site in enumerate(augmented_structure.sites):
        is_image = site_idx >= len(struct_i)
        symbol = get_site_symbol(site)
        sites_by_element.setdefault(symbol, {"primary": [], "image": []})
        coords = site.coords
        sites_by_element[symbol]["image" if is_image else "primary"].append((site, coords, site_idx))
    
    for symbol, groups in sites_by_element.items():
        for group_type, site_list in groups.items():
            if not site_list:
                continue
            coords = np.array([coords for _, coords, _ in site_list])
            # Determine marker properties
            radius = atomic_radii.get(symbol, 0.8)
            size = radius * scale * atom_size
            color = elem_colors.get(symbol, "gray")
            opacity = 0.6 if group_type == "image" else 1.0
            # Generate hover texts
            hover_texts = []
            for site, _, site_idx in site_list:
                if callable(hover_text):
                    hover_txt = hover_text(site)
                else:
                    coords_str = _format_coordinates(site, hover_text, hover_float_fmt)
                    hover_txt = f"<b>{symbol}</b><br>{coords_str}"
                hover_texts.append(hover_txt)
            # Legend grouping: only show legend for primary group once
            showlegend = symbol not in seen_elements and group_type == "primary"
            if showlegend:
                seen_elements.add(symbol)
            # Text labels if requested
            text_labels = None
            if site_labels == "symbol":
                text_labels = [symbol] * len(site_list)
            elif site_labels == "species":
                text_labels = [str(site.species) for site, _, _ in site_list]
            elif isinstance(site_labels, dict):
                text_labels = [site_labels.get(idx, "") for _, _, idx in site_list]
            elif isinstance(site_labels, Sequence):
                text_labels = [site_labels[idx] if idx < len(site_labels) else "" for _, _, idx in site_list]
            # Add trace
            fig.add_scatter3d(
                x=coords[:,0], y=coords[:,1], z=coords[:,2],
                mode="markers+text" if text_labels else "markers",
                marker=dict(size=size, color=color, opacity=opacity, line=dict(width=1, color="rgba(0,0,0,0.4)")),
                text=text_labels,
                textposition="middle center",
                hovertext=hover_texts,
                hoverinfo="text",
                name=symbol,
                legendgroup=symbol,
                showlegend=showlegend,
                scene=f"scene{idx}"
            )


def _plot_vectors(fig, struct_i, vector_prop, vector_kwargs, idx):
    """
    Render vector properties (forces, magnetic moments) as 3D arrows
    """
    for site_idx, site in enumerate(struct_i):
        vector = None
        if vector_prop in site.properties:
            vector = np.array(site.properties[vector_prop])
        elif vector_prop in struct_i.properties and site_idx < len(struct_i.properties[vector_prop]):
            vector = struct_i.properties[vector_prop][site_idx]
        
        if vector is not None and np.any(vector):
            draw_vector(
                fig, site.coords, vector,
                is_3d=True,
                arrow_kwargs=(vector_kwargs or {}).get(vector_prop, {}),
                scene=f"scene{idx}",
                name=f"vector{site_idx}"
            )

def _plot_bonds(fig, augmented_structure, show_bonds, struct_key, idx, 
               bond_kwargs, elem_colors, show_sites):
    """
    Render chemical bonds between atoms using nearest neighbor algorithms
    """
    struct_show_bonds = show_bonds
    if isinstance(show_bonds, dict):
        struct_show_bonds = show_bonds.get(struct_key, False)
    
    if struct_show_bonds:
        plotted_sites_coords = None
        if show_sites:
            plotted_sites_coords = {
                tuple(np.round(site.coords, 5))
                for site in augmented_structure.sites
            }
        else:
            plotted_sites_coords = set()
        
        draw_bonds(
            fig=fig,
            structure=augmented_structure,
            nn=CrystalNN() if struct_show_bonds is True else struct_show_bonds,
            is_3d=True,
            bond_kwargs=bond_kwargs,
            scene=f"scene{idx}",
            elem_colors=elem_colors,
            plotted_sites_coords=plotted_sites_coords,
        )

def _format_coordinates(site, hover_text, hover_float_fmt):
    """
    Format atomic site coordinates for hover text display
    """
    def format_coord(coord_val):
        if callable(hover_float_fmt):
            return hover_float_fmt(coord_val)
        return f"{float(coord_val):{hover_float_fmt}}"
    
    cart_text = f"({', '.join(format_coord(c) for c in site.coords)})"
    frac_text = f"[{', '.join(format_coord(c) for c in site.frac_coords)}]"
    
    if hover_text == SiteCoords.cartesian:
        return cart_text
    elif hover_text == SiteCoords.fractional:
        return frac_text
    elif hover_text == SiteCoords.cartesian_fractional:
        return f"{cart_text} {frac_text}"
    return cart_text

def _set_subplot_title(fig, struct_i, struct_key, idx, subplot_title):
    """
    Set title for individual subplot based on structure information
    """
    if callable(subplot_title):
        title_info = subplot_title(struct_i, struct_key)
        if isinstance(title_info, str):
            title_text = title_info
        elif isinstance(title_info, dict):
            title_text = title_info.get("text", f"{idx}. {struct_i.formula}")
        else:
            title_text = str(title_info)
    else:
        try:
            spg_num = struct_i.get_space_group_info()[1]
            title_text = f"{idx}. {struct_i.formula} (spg={spg_num})"
        except:
            title_text = f"{idx}. {struct_i.formula}"
    
    if idx <= len(fig.layout.annotations):
        fig.layout.annotations[idx - 1].update(text=title_text)

@lru_cache(maxsize=50)
def _get_scene_config():
    """
    Cached 3D scene configuration for consistent appearance
    """
    return dict(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False
    )

def _configure_3d_scenes(fig, n_structs, n_cols, n_rows, site_labels):
    """
    Configure 3D scene properties and subplot domains
    """
    no_axes_kwargs = _get_scene_config()
    
    fig.update_scenes(
        xaxis=no_axes_kwargs,
        yaxis=no_axes_kwargs,  
        zaxis=no_axes_kwargs,
        aspectmode="data",
        bgcolor="rgba(80,80,80,0.01)",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    )
    
    gap = 0.01
    for idx in range(1, n_structs + 1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1
        
        x_start = (col - 1) / n_cols + gap / 2
        x_end = col / n_cols - gap / 2
        y_start = 1 - row / n_rows + gap / 2
        y_end = 1 - (row - 1) / n_rows - gap / 2
        
        domain = dict(x=[x_start, x_end], y=[y_start, y_end])
        fig.update_layout({f"scene{idx}": dict(domain=domain, aspectmode="data")})
