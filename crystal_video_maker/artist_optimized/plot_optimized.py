"""
Optimized plot.py with parallel processing and improved performance.
"""
from typing import Literal, Any, cast
from collections.abc import Sequence, Callable
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import numpy as np

from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from pymatgen.core import PeriodicSite, Structure
from pymatgen.core.periodic_table import Element

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from crystal_video_maker.artist import AnyStructure
from crystal_video_maker.artist.colors import color_map_type
from crystal_video_maker.artist.enum import SiteCoords
from crystal_video_maker.artist_optimized import (
    normalize_structures,
    get_atomic_radii,
    get_elem_colors,
    get_first_matching_site_prop,
    get_struct_prop,
    _prep_augmented_structure_for_bonding,
    get_site_symbol,
)
from crystal_video_maker.artist_optimized import (
    draw_cell,
    draw_site,
    draw_vector,
    draw_bonds,
    _standardize_struct,
)
from crystal_video_maker.artist.add_details import (
    configure_subplot_legends,
    get_subplot_title,
)


def process_single_structure(
    struct_data: tuple,
    common_params: dict,
) -> dict:
    """Process a single structure for parallel execution."""
    idx, struct_key, raw_struct_i = struct_data
    
    struct_i = _standardize_struct(
        raw_struct_i, standardize_struct=common_params['standardize_struct']
    )
    
    # Handle per-structure parameters
    cell_boundary_tol_i = (
        get_struct_prop(
            raw_struct_i, struct_key, "cell_boundary_tol", 
            common_params['cell_boundary_tol']
        ) or 0.0
    )
    
    atomic_radii_i = get_struct_prop(
        raw_struct_i, struct_key, "atomic_radii", common_params['atomic_radii']
    )
    if atomic_radii_i is None:
        atomic_radii_i = common_params['atomic_radii']
        
    atom_size_i = (
        get_struct_prop(raw_struct_i, struct_key, "atom_size", common_params['atom_size'])
        or common_params['atom_size']
    )
    
    scale_i = (
        get_struct_prop(raw_struct_i, struct_key, "scale", common_params['scale']) 
        or common_params['scale']
    )
    
    # Process atomic radii with per-structure precedence
    _atomic_radii = get_atomic_radii(atomic_radii_i)
    
    # Handle per-structure elem_colors settings
    struct_elem_colors = common_params['elem_colors']
    if isinstance(common_params['elem_colors'], dict):
        is_struct_key_mapping = any(
            key not in Element.__members__ for key in common_params['elem_colors']
        )
        if is_struct_key_mapping:
            struct_elem_colors = cast(
                "dict[str, str]",
                common_params['elem_colors'].get(struct_key, color_map_type["VESTA"]),
            )
    
    _elem_colors = get_elem_colors(struct_elem_colors)
    
    # Prepare augmented structure
    augmented_structure = _prep_augmented_structure_for_bonding(
        struct_i,
        show_image_sites=common_params['show_image_sites'] and common_params['show_sites'],
        cell_boundary_tol=cell_boundary_tol_i,
        parallel=True,  # Enable parallel processing for image sites
    )
    
    return {
        'idx': idx,
        'struct_key': struct_key,
        'struct_i': struct_i,
        'augmented_structure': augmented_structure,
        'atomic_radii': _atomic_radii,
        'elem_colors': _elem_colors,
        'atom_size': atom_size_i,
        'scale': scale_i,
        'cell_boundary_tol': cell_boundary_tol_i,
    }


def structure_3d(
    struct: AnyStructure | dict[str, AnyStructure] | Sequence[AnyStructure],
    *,
    atomic_radii: float | dict[str, float] | None = None,
    atom_size: float = 20,
    elem_colors: dict[str, str] = color_map_type["VESTA"],
    scale: float = 1,
    show_cell: bool | dict[str, Any] = True,
    show_cell_faces: bool | dict[str, Any] = True,
    show_sites: bool | dict[str, Any] = True,
    show_image_sites: bool | dict[str, Any] = True,
    cell_boundary_tol: float | dict[str, float] = 0.0,
    show_bonds: bool | NearNeighbors | dict[str, bool | NearNeighbors] = False,
    site_labels: (
        Literal["symbol", "species", "legend", False] | dict[str, str] | Sequence[str]
    ) = "legend",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: (
        Callable[[Structure, str | int], str | dict[str, Any]] | None | Literal[False]
    ) = None,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: (
        SiteCoords | Callable[[PeriodicSite], str]
    ) = SiteCoords.cartesian_fractional,
    hover_float_fmt: str | Callable[[float], str] = ".4",
    bond_kwargs: dict[str, Any] | None = None,
    parallel: bool = False,  # New parameter for parallel processing
) -> go.Figure:
    """
    Optimized 3D structure plotting with parallel processing support.
    
    Args:
        parallel (bool): Enable parallel processing for multiple structures.
            Recommended for more than 5 structures or structures with >1000 atoms.
        
    [All other args same as original]
    
    Returns:
        go.Figure: Plotly figure showing optimized 3D structure(s).
    """
    structures = normalize_structures(struct)
    n_structs = len(structures)
    
    # Auto-enable parallel processing for large datasets
    if not parallel and (n_structs > 5 or any(len(s) > 1000 for s in structures.values())):
        parallel = True
    
    n_cols = min(n_cols, n_structs)
    n_rows = (n_structs - 1) // n_cols + 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=[" " for _ in range(n_structs)],
    )

    _atomic_radii = get_atomic_radii(atomic_radii)

    if isinstance(show_site_vectors, str):
        show_site_vectors = [show_site_vectors]

    # Determine vector property to plot
    vector_prop = get_first_matching_site_prop(
        list(structures.values()),
        show_site_vectors,
        warn_if_none=show_site_vectors != ("force", "magmom"),
        filter_callback=lambda _prop, value: (np.array(value).shape or [None])[-1] == 3,
    )

    # Track seen elements per subplot for legend management
    seen_elements_per_subplot: dict[int, set[str]] = {}

    # Prepare common parameters for parallel processing
    common_params = {
        'atomic_radii': atomic_radii,
        'atom_size': atom_size,
        'elem_colors': elem_colors,
        'scale': scale,
        'show_sites': show_sites,
        'show_image_sites': show_image_sites,
        'cell_boundary_tol': cell_boundary_tol,
        'standardize_struct': standardize_struct,
    }

    # Process structures
    if parallel and n_structs > 3:
        # Parallel processing for multiple structures
        struct_data_list = [
            (idx, struct_key, raw_struct_i)
            for idx, (struct_key, raw_struct_i) in enumerate(structures.items(), start=1)
        ]
        
        with ThreadPoolExecutor(max_workers=min(mp.cpu_count(), n_structs)) as executor:
            processed_structures = list(executor.map(
                lambda data: process_single_structure(data, common_params),
                struct_data_list
            ))
    else:
        # Sequential processing for smaller datasets
        processed_structures = []
        for idx, (struct_key, raw_struct_i) in enumerate(structures.items(), start=1):
            struct_data = (idx, struct_key, raw_struct_i)
            processed_structures.append(
                process_single_structure(struct_data, common_params)
            )

    # Render processed structures
    for processed in processed_structures:
        idx = processed['idx']
        struct_key = processed['struct_key']
        struct_i = processed['struct_i']
        augmented_structure = processed['augmented_structure']
        _atomic_radii = processed['atomic_radii']
        _elem_colors = processed['elem_colors']
        atom_size_i = processed['atom_size']
        scale_i = processed['scale']

        # Initialize seen elements for this subplot
        seen_elements_per_subplot[idx] = set()

        # Plot atoms and vectors
        if show_sites:
            # Batch process sites for better performance
            sites_data = []
            
            for site_idx_loop, site in enumerate(augmented_structure.sites):
                is_image_site = site_idx_loop >= len(struct_i)
                original_site_idx = (
                    site_idx_loop if not is_image_site else site_idx_loop % len(struct_i)
                )
                
                symbol = get_site_symbol(site)
                
                # Determine legend parameters for primary sites only
                legendgroup = None
                showlegend = False
                if site_labels == "legend" and not is_image_site:
                    legendgroup = f"{idx}-{symbol}"
                    if symbol not in seen_elements_per_subplot[idx]:
                        showlegend = True
                        seen_elements_per_subplot[idx].add(symbol)
                
                sites_data.append({
                    'site': site,
                    'coords': site.coords,
                    'site_idx': original_site_idx,
                    'is_image': is_image_site,
                    'symbol': symbol,
                    'legendgroup': legendgroup,
                    'showlegend': showlegend,
                })
            
            # Process sites in batches for better performance
            batch_size = 100
            for i in range(0, len(sites_data), batch_size):
                batch = sites_data[i:i + batch_size]
                
                if parallel and len(batch) > 20:
                    # Parallel processing for large batches
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = []
                        for site_data in batch:
                            future = executor.submit(
                                draw_site,
                                fig=fig,
                                site=site_data['site'],
                                coords=site_data['coords'],
                                site_idx=site_data['site_idx'],
                                site_labels=site_labels,
                                elem_colors=_elem_colors,
                                atomic_radii=_atomic_radii,
                                atom_size=atom_size_i,
                                scale=scale_i,
                                site_kwargs={} if show_sites is True else show_sites,
                                is_image=site_data['is_image'],
                                is_3d=True,
                                scene=f"scene{idx}",
                                hover_text=hover_text,
                                float_fmt=hover_float_fmt,
                                legendgroup=site_data['legendgroup'],
                                showlegend=site_data['showlegend'],
                                legend=f"legend{idx}" if idx > 1 and n_structs > 1 else "legend",
                                name=f"Image of {site_data['symbol']}" if site_data['is_image'] else site_data['symbol'],
                            )
                            futures.append(future)
                        
                        # Wait for all futures to complete
                        for future in futures:
                            future.result()
                else:
                    # Sequential processing for smaller batches
                    for site_data in batch:
                        draw_site(
                            fig=fig,
                            site=site_data['site'],
                            coords=site_data['coords'],
                            site_idx=site_data['site_idx'],
                            site_labels=site_labels,
                            elem_colors=_elem_colors,
                            atomic_radii=_atomic_radii,
                            atom_size=atom_size_i,
                            scale=scale_i,
                            site_kwargs={} if show_sites is True else show_sites,
                            is_image=site_data['is_image'],
                            is_3d=True,
                            scene=f"scene{idx}",
                            hover_text=hover_text,
                            float_fmt=hover_float_fmt,
                            legendgroup=site_data['legendgroup'],
                            showlegend=site_data['showlegend'],
                            legend=f"legend{idx}" if idx > 1 and n_structs > 1 else "legend",
                            name=f"Image of {site_data['symbol']}" if site_data['is_image'] else site_data['symbol'],
                        )

        # Add vectors for primary sites only (optimized)
        if vector_prop:
            vector_sites = [(i, site) for i, site in enumerate(struct_i)]
            
            def add_vector_for_site(site_data):
                site_idx_loop, site_in_original_struct = site_data
                vector = None
                
                if vector_prop in site_in_original_struct.properties:
                    vector = np.array(site_in_original_struct.properties[vector_prop])
                elif (vector_prop in struct_i.properties and 
                      site_idx_loop < len(struct_i.properties[vector_prop])):
                    vector = struct_i.properties[vector_prop][site_idx_loop]

                if vector is not None and np.any(vector):
                    draw_vector(
                        fig,
                        site_in_original_struct.coords,
                        vector,
                        is_3d=True,
                        arrow_kwargs=(vector_kwargs or {}).get(vector_prop, {}),
                        scene=f"scene{idx}",
                        name=f"vector{site_idx_loop}",
                    )
            
            if parallel and len(vector_sites) > 50:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(add_vector_for_site, vector_sites))
            else:
                for site_data in vector_sites:
                    add_vector_for_site(site_data)

        # Draw bonds using the augmented structure
        if show_bonds:
            struct_show_bonds = show_bonds
            if isinstance(show_bonds, dict):
                struct_show_bonds = show_bonds.get(struct_key, False)

            if struct_show_bonds:
                plotted_sites_coords: set[tuple[float, float, float]] | None = None
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
                    elem_colors=_elem_colors,
                    plotted_sites_coords=plotted_sites_coords,
                )

        if show_cell:
            draw_cell(
                fig,
                struct_i,
                cell_kwargs={} if show_cell is True else show_cell,
                is_3d=True,
                scene=f"scene{idx}",
                show_faces=show_cell_faces,
            )

        # Set subplot titles
        if subplot_title is not False:
            anno = get_subplot_title(struct_i, struct_key, idx, subplot_title)
            if "y" not in anno:
                row = (idx - 1) // n_cols + 1
                subtitle_y_pos = 1 - (row - 1) / n_rows - 0.02
                anno["y"] = subtitle_y_pos
            if "yanchor" not in anno:
                anno["yanchor"] = "top"
            fig.layout.annotations[idx - 1].update(anno)

    # Optimized scene updates
    no_axes_kwargs = dict(
        showticklabels=False, showgrid=False, zeroline=False, visible=False
    )

    fig.update_scenes(
        xaxis=no_axes_kwargs,
        yaxis=no_axes_kwargs,
        zaxis=no_axes_kwargs,
        aspectmode="data",
        bgcolor="rgba(80,80,80,0.01)",
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
        ),
    )

    # Optimized subplot positioning
    gap = 0.01
    scene_updates = {}
    for idx in range(1, n_structs + 1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1

        x_start = (col - 1) / n_cols + gap / 2
        x_end = col / n_cols - gap / 2
        y_start = 1 - row / n_rows + gap / 2
        y_end = 1 - (row - 1) / n_rows - gap / 2

        domain = dict(x=[x_start, x_end], y=[y_start, y_end])
        scene_updates[f"scene{idx}"] = dict(domain=domain, aspectmode="data")
    
    fig.update_layout(**scene_updates)

    # Final layout optimization
    fig.layout.height = 400 * n_rows
    fig.layout.width = 400 * n_cols
    fig.layout.showlegend = site_labels == "legend"
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"
    fig.layout.margin = dict(l=0, r=0, t=30, b=0)

    # Configure legends for each subplot
    configure_subplot_legends(fig, site_labels, n_structs, n_cols, n_rows)

    return fig
