# plot.py

from typing import Literal, Any, cast
from collections.abc import Sequence, Callable

import numpy as np

from pymatgen.analysis.local_env import CrystalNN, NearNeighbors
from pymatgen.core import PeriodicSite, Structure
from pymatgen.core.periodic_table import Element

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from crystal_video_maker.artist import AnyStructure
from crystal_video_maker.artist.colors import color_map_type
from crystal_video_maker.artist.enum import SiteCoords
from crystal_video_maker.artist.hands import normalize_structures, get_atomic_radii, get_elem_colors, get_first_matching_site_prop, get_struct_prop, _prep_augmented_structure_for_bonding, get_site_symbol
from crystal_video_maker.artist.hands import draw_cell, draw_site, draw_vector, draw_bonds, _standardize_struct
from crystal_video_maker.artist.add_details import configure_subplot_legends, get_subplot_title

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
    site_labels: Literal["symbol", "species", "legend", False]
    | dict[str, str]
    | Sequence[str] = "legend",
    standardize_struct: bool | None = None,
    n_cols: int = 3,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]]
    | None
    | Literal[False] = None,
    show_site_vectors: str | Sequence[str] = ("force", "magmom"),
    vector_kwargs: dict[str, dict[str, Any]] | None = None,
    hover_text: SiteCoords
    | Callable[[PeriodicSite], str] = SiteCoords.cartesian_fractional,
    hover_float_fmt: str | Callable[[float], str] = ".4",
    bond_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Plot pymatgen structures in 3D with Plotly.

    Args:
        struct (AnyStructure | dict[str, AnyStructure] | Sequence[AnyStructure]):
            Structure(s) to plot.
        atomic_radii (float | dict[str, float], optional): Either a scaling factor for
            default radii or map from element symbol to atomic radii. Defaults to None.
        atom_size (float, optional): Scaling factor for atom sizes. Defaults to 20.
        elem_colors (ElemColorScheme | dict[str, str], optional): Element color scheme
            or custom color map. Defaults to ElemColorScheme.jmol.
        scale (float, optional): Scaling of the plotted atoms and lines. Defaults to 1.
        show_cell (bool | dict[str, Any], optional): Whether to render cell. If
            a dict, will be used to customize cell appearance. The dict should have
            a "node"/"edge" key to customize node/edge appearance. Defaults to True.
        show_cell_faces (bool | dict[str, Any], optional): Whether to show transparent
            cell faces. If a dict, will  customize surfaces style. Defaults to True.
        show_sites (bool | dict[str, Any], optional): Whether to plot atomic sites. If
            a dict, will be used to customize site marker appearance. Defaults to True.
        show_image_sites (bool | dict[str, Any], optional): Whether to show image sites
            on cell edges and surfaces. If a dict, will be used to customize how
            image sites are rendered. Defaults to True.
        cell_boundary_tol (float | dict[str, float], optional): Buffer distance (in
            fractional coordinates) beyond the unit cell boundaries to include image
            atoms. Defaults to 0.0 for strict cell boundaries (only atoms with
            0 <= coord <= 1). Higher values include atoms further outside the cell.
            Can also be set per-structure via Structure.properties["cell_boundary_tol"]
            or Atoms.info["cell_boundary_tol"] with highest precedence.
        show_bonds (bool | NearNeighbors | dict[str, bool | NearNeighbors], optional):
            Whether to draw bonds between sites. If True, uses CrystalNN with a cutoff
            of 10 Å to determine nearest neighbors, including in neighboring cells.
            If a NearNeighbors object, uses that to determine nearest neighbors. If a
            dict, allows per-structure settings. Defaults to False (since still
            experimental).
        site_labels ("symbol" | "species" | "legend" | False | dict | Sequence):
            How to annotate lattice sites. Defaults to "legend" (show an element
            color legend). Other options: "symbol", "species", False (no labels),
            a dict mapping site index to label, or a sequence of labels.
        standardize_struct (bool, optional): Whether to standardize the structure.
            Defaults to None.
        n_cols (int, optional): Number of columns for subplots. Defaults to 3.
        subplot_title (Callable[[Structure, str | int], str | dict] | False, optional):
            Function to generate subplot titles. Defaults to
            lambda struct_i, idx: f"{idx}. {struct_i.formula} (spg={spg_num})". Set to
            False to hide all subplot titles.
        show_site_vectors (str | Sequence[str], optional): Whether to show vector site
            quantities such as forces or magnetic moments as arrow heads originating
            from each site. Pass the key (or sequence of keys) to look for in site
            properties. Defaults to ("force", "magmom"). If not found as a site
            property, will look for it in the structure properties as well and assume
            the key points at a (N, 3) array with N the number of sites. If multiple
            keys are provided, it plots the first key found in site properties or
            structure properties in any of the passed structures (if a dict of
            structures was passed). But it will only plot one vector per site and it
            will use the same key for all sites and across all structures.
        vector_kwargs (dict[str, dict[str, Any]], optional): For customizing vector
            arrows. Keys are property names (e.g., "force", "magmom"), values are
            dictionaries of arrow customization options. Use key "scale" to adjust
            vector length.
        hover_text (SiteCoords | Callable, optional): Controls the hover tooltip
            template. Can be SiteCoords.cartesian, SiteCoords.fractional,
            SiteCoords.cartesian_fractional, or a callable that takes a site and
            returns a custom string. Defaults to SiteCoords.cartesian_fractional.
        hover_float_fmt (str | Callable[[float], str], optional): Float formatting for
            hover coordinates. Can be an f-string format like ".4" (default) or a
            callable that takes a float and returns a string. Defaults to ".4".
        bond_kwargs (dict[str, Any], optional): For customizing bond lines. Keys are
            line properties (e.g., "color", "width"), values are the corresponding
            values. Defaults to None.

    Returns:
        go.Figure: Plotly figure showing the 3D structure(s).
    """
    structures = normalize_structures(struct)

    n_structs = len(structures)
    n_cols = min(n_cols, n_structs)
    n_rows = (n_structs - 1) // n_cols + 1

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        # Avoid IndexError on fig.layout.annotations[idx - 1].update(anno)
        subplot_titles=[" " for _ in range(n_structs)],
    )

    # Don't call get_elem_colors here if elem_colors is a dict mapping structure
    # keys to color schemes. Instead, handle per-structure color schemes in the loop
    _atomic_radii = get_atomic_radii(atomic_radii)

    if isinstance(show_site_vectors, str):
        show_site_vectors = [show_site_vectors]

    # Determine which vector property to plot (calling outside loop ensures we plot
    # the same prop for all sites in all structures)
    vector_prop = get_first_matching_site_prop(
        list(structures.values()),
        show_site_vectors,
        warn_if_none=show_site_vectors != ("force", "magmom"),
        filter_callback=lambda _prop, value: (np.array(value).shape or [None])[-1] == 3,
    )

    # Track seen elements per subplot for legend management
    seen_elements_per_subplot: dict[int, set[str]] = {}

    for idx, (struct_key, raw_struct_i) in enumerate(structures.items(), start=1):
        struct_i = _standardize_struct(
            raw_struct_i, standardize_struct=standardize_struct
        )

        # Handle per-structure elem_colors settings if it's a dict
        struct_elem_colors:  dict[str, str] = elem_colors
        if isinstance(elem_colors, dict):
            # If any key is not a valid element symbol, treat as structure-key mapping
            is_struct_key_mapping = any(
                key not in Element.__members__ for key in elem_colors
            )

            if is_struct_key_mapping:  # is map of structure keys to color schemes
                struct_elem_colors = cast(
                    "dict[str, str]",
                    elem_colors.get(struct_key, color_map_type["VESTA"]),
                )

        _elem_colors = get_elem_colors(struct_elem_colors)

        # Initialize seen elements for this subplot
        seen_elements_per_subplot[idx] = set()

        # Handle cell_boundary_tol with precedence:
        # 1. structure.properties["cell_boundary_tol"] (highest precedence)
        # 2. function parameter cell_boundary_tol (dict or float)
        cell_boundary_tol_i = (
            get_struct_prop(
                raw_struct_i, struct_key, "cell_boundary_tol", cell_boundary_tol
            )
            or 0.0
        )

        # Handle other parameters with precedence
        atomic_radii_i = get_struct_prop(
            raw_struct_i, struct_key, "atomic_radii", atomic_radii
        )
        if atomic_radii_i is None:
            atomic_radii_i = atomic_radii

        atom_size_i = (
            get_struct_prop(raw_struct_i, struct_key, "atom_size", atom_size)
            or atom_size
        )

        scale_i = (
            get_struct_prop(raw_struct_i, struct_key, "scale", scale) or scale
        )

        # Process atomic_radii with per-structure precedence
        _atomic_radii = get_atomic_radii(atomic_radii_i)

        # Prepare augmented structure: original + image sites for consistent processing
        # This augmented_structure is used for collecting all site data for the single
        # 3D trace and for bond calculations.
        augmented_structure = _prep_augmented_structure_for_bonding(
            struct_i,
            show_image_sites=show_image_sites and show_sites,
            cell_boundary_tol=cell_boundary_tol_i,
        )

        # Plot atoms and vectors
        if show_sites:
            # For 3D, need to handle disordered sites properly by drawing each site
            # individually instead of grouping by element symbol
            for site_idx_loop, site in enumerate(augmented_structure.sites):
                # Determine if this is primary site (from orig structure) or image site
                is_image_site = site_idx_loop >= len(struct_i)

                # For primary sites, use original index; for image sites, use modulo
                original_site_idx = (
                    site_idx_loop
                    if not is_image_site
                    else site_idx_loop % len(struct_i)
                )

                symbol = get_site_symbol(site)

                # Determine legend parameters for primary sites only
                legendgroup = None
                showlegend = False
                if site_labels == "legend" and not is_image_site:
                    legendgroup = f"{idx}-{symbol}"  # Unique per subplot
                    if symbol not in seen_elements_per_subplot[idx]:
                        showlegend = True
                        seen_elements_per_subplot[idx].add(symbol)

                # Use the new draw_site helper which handles disordered sites
                draw_site(
                    fig=fig,
                    site=site,
                    coords=site.coords,  # Use 3D coordinates directly
                    site_idx=original_site_idx,
                    site_labels=site_labels,
                    elem_colors=_elem_colors,
                    atomic_radii=_atomic_radii,
                    atom_size=atom_size_i,
                    scale=scale_i,
                    site_kwargs={} if show_sites is True else show_sites,
                    is_image=is_image_site,
                    is_3d=True,
                    scene=f"scene{idx}",
                    hover_text=hover_text,
                    float_fmt=hover_float_fmt,
                    legendgroup=legendgroup,
                    showlegend=showlegend,
                    legend=f"legend{idx}" if idx > 1 and n_structs > 1 else "legend",
                    name=f"Image of {symbol}" if is_image_site else symbol,
                )

            # Add vectors for primary sites only
            for site_idx_loop, site_in_original_struct in enumerate(struct_i):
                if vector_prop:
                    vector = None
                    # Check properties on the original site object
                    if vector_prop in site_in_original_struct.properties:
                        vector = np.array(
                            site_in_original_struct.properties[vector_prop]
                        )
                    # Check structure-level properties, using original site_idx_loop
                    elif vector_prop in struct_i.properties and site_idx_loop < len(
                        struct_i.properties[vector_prop]
                    ):
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

        # Draw bonds using the augmented structure after sites are processed
        if show_bonds:
            # Handle per-structure show_bonds settings if it's a dict
            struct_show_bonds = show_bonds
            if isinstance(show_bonds, dict):
                struct_show_bonds = show_bonds.get(struct_key, False)

            if struct_show_bonds:
                plotted_sites_coords: set[tuple[float, float, float]] | None = None
                if show_sites:  # Only consider plotted sites for bonds
                    # Get all plotted site coordinates from the augmented structure
                    plotted_sites_coords = {
                        tuple(np.round(site.coords, 5))
                        for site in augmented_structure.sites
                    }
                else:
                    # If no sites are rendered, set empty set to filter out all bonds
                    plotted_sites_coords = set()

                draw_bonds(
                    fig=fig,
                    structure=augmented_structure,  # Pass the augmented structure
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

        # Update 3D scene properties
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
                eye=dict(x=1.5, y=1.5, z=1.5),  # Camera position for better 3D view
            ),
        )

    # Calculate subplot positions with small gap
    gap = 0.01
    for idx in range(1, n_structs + 1):
        row = (idx - 1) // n_cols + 1
        col = (idx - 1) % n_cols + 1

        x_start = (col - 1) / n_cols + gap / 2
        x_end = col / n_cols - gap / 2
        y_start = 1 - row / n_rows + gap / 2  # Invert y-axis to match row order
        y_end = 1 - (row - 1) / n_rows - gap / 2

        domain = dict(x=[x_start, x_end], y=[y_start, y_end])
        fig.update_layout({f"scene{idx}": dict(domain=domain, aspectmode="data")})

    # Update overall layout
    fig.layout.height = 400 * n_rows
    fig.layout.width = 400 * n_cols
    fig.layout.showlegend = site_labels == "legend"  # Show legend if using legend mode
    fig.layout.paper_bgcolor = "rgba(0,0,0,0)"  # Transparent background
    fig.layout.plot_bgcolor = "rgba(0,0,0,0)"  # Transparent background
    fig.layout.margin = dict(l=0, r=0, t=30, b=0)  # Minimize margins

    # Configure legends for each subplot
    configure_subplot_legends(fig, site_labels, n_structs, n_cols, n_rows)

    return fig