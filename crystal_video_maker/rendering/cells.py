"""
Unit cell rendering functions with performance improvements
"""

import plotly.graph_objects as go
from typing import Any, Dict, List, Optional
import numpy as np
from pymatgen.core import Structure, Lattice
from functools import lru_cache

@lru_cache(maxsize=200)
def _get_cached_cell_line_style(
    color: str = "black",
    width: float = 1.0,
    dash: str = "solid"
) -> Dict[str, Any]:
    """
    Get cached unit cell line styling

    Args:
        color: Line color
        width: Line width
        dash: Line dash style

    Returns:
        Dictionary with line styling
    """
    return dict(color=color, width=width, dash=dash)

def draw_cell(
    fig: go.Figure,
    structure: Structure,
    cell_kwargs: Optional[Dict[str, Any]] = None,
    *,
    is_3d: bool = True,
    scene: str = "scene",
    show_faces: bool = False,
    row: Optional[int] = None,
    col: Optional[int] = None,
) -> None:
    """
    Draw unit cell edges and optionally faces with performance improvements

    Args:
        fig: Plotly figure to add cell to
        structure: Crystal structure
        cell_kwargs: Cell styling options
        is_3d: Whether this is a 3D plot
        scene: Scene identifier for 3D plots  
        show_faces: Whether to draw transparent cell faces
        row: Subplot row for 2D plots
        col: Subplot column for 2D plots
    """
    cell_kwargs = cell_kwargs or {}

    # Get lattice vectors
    lattice = structure.lattice
    a, b, c = lattice.matrix

    # Define unit cell vertices in fractional coordinates
    vertices_frac = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
    ])

    # Convert to cartesian coordinates
    vertices_cart = np.array([
        lattice.get_cartesian_coords(frac_coord) 
        for frac_coord in vertices_frac
    ])

    # Define edges of the unit cell (pairs of vertex indices)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face  
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    # Get line styling
    line_color = cell_kwargs.get('color', 'black')
    line_width = cell_kwargs.get('width', 1.0)
    line_style = _get_cached_cell_line_style(line_color, line_width)

    # Collect edge coordinates for batch rendering
    edge_coords = {'x': [], 'y': [], 'z': []}

    for start_idx, end_idx in edges:
        start_coord = vertices_cart[start_idx]
        end_coord = vertices_cart[end_idx]

        edge_coords['x'].extend([start_coord[0], end_coord[0], None])
        edge_coords['y'].extend([start_coord[1], end_coord[1], None])
        edge_coords['z'].extend([start_coord[2], end_coord[2], None])

    # Add cell edges
    scatter_kwargs = dict(
        x=edge_coords['x'],
        y=edge_coords['y'], 
        mode="lines",
        line=line_style,
        showlegend=False,
        name="unit_cell",
        hoverinfo="skip"
    )

    if is_3d:
        scatter_kwargs['z'] = edge_coords['z']
        scatter_kwargs['scene'] = scene
        fig.add_scatter3d(**scatter_kwargs)
    else:
        if row is not None and col is not None:
            scatter_kwargs.update(row=row, col=col)
        fig.add_scatter(**scatter_kwargs)

    # Add cell faces if requested
    if show_faces and is_3d:
        _draw_cell_faces(fig, vertices_cart, cell_kwargs, scene)

def _draw_cell_faces(
    fig: go.Figure,
    vertices: np.ndarray,
    cell_kwargs: Dict[str, Any],
    scene: str
) -> None:
    """
    Draw transparent unit cell faces

    Args:
        fig: Plotly figure to add faces to
        vertices: Unit cell vertices in cartesian coordinates
        cell_kwargs: Cell styling options
        scene: Scene identifier
    """
    face_color = cell_kwargs.get('face_color', 'rgba(255,255,255,0.1)')
    face_opacity = cell_kwargs.get('face_opacity', 0.1)

    # Define faces (sets of 4 vertices each)
    faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5]   # right
    ]

    # Convert each face to triangular mesh
    for face_vertices in faces:
        # Split quadrilateral into two triangles
        v0, v1, v2, v3 = [vertices[i] for i in face_vertices]

        # Triangle 1: v0, v1, v2
        # Triangle 2: v0, v2, v3
        face_mesh_vertices = np.array([v0, v1, v2, v3])

        fig.add_mesh3d(
            x=face_mesh_vertices[:, 0],
            y=face_mesh_vertices[:, 1],
            z=face_mesh_vertices[:, 2],
            i=[0, 0],  # First vertex of each triangle
            j=[1, 2],  # Second vertex of each triangle  
            k=[2, 3],  # Third vertex of each triangle
            color=face_color,
            opacity=face_opacity,
            showscale=False,
            showlegend=False,
            scene=scene,
            hoverinfo="skip"
        )

def draw_supercell(
    fig: go.Figure,
    structure: Structure,
    supercell_matrix: List[List[int]],
    **kwargs
) -> None:
    """
    Draw a supercell outline

    Args:
        fig: Plotly figure to add supercell to
        structure: Base crystal structure
        supercell_matrix: 3x3 supercell transformation matrix
        **kwargs: Additional arguments passed to draw_cell
    """
    # Create supercell structure
    supercell_structure = structure * supercell_matrix

    # Draw the supercell
    draw_cell(fig, supercell_structure, **kwargs)

def batch_draw_cells(
    fig: go.Figure,
    structures: List[Structure],
    cell_kwargs_list: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> None:
    """
    Draw multiple unit cells efficiently

    Args:
        fig: Plotly figure to add cells to
        structures: List of crystal structures
        cell_kwargs_list: List of cell styling options for each structure
        **kwargs: Common arguments for all cells
    """
    if cell_kwargs_list is None:
        cell_kwargs_list = [{}] * len(structures)

    for structure, cell_kwargs in zip(structures, cell_kwargs_list):
        draw_cell(fig, structure, cell_kwargs, **kwargs)
