"""
Unit cell rendering functions with advanced cell visualization
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import plotly.graph_objects as go

from pymatgen.core import Structure, Lattice

def get_cell_vertices(lattice: Lattice) -> np.ndarray:
    """
    Get vertices of the unit cell
    
    Args:
        lattice: Crystal lattice
        
    Returns:
        Array of vertex coordinates (8x3)
    """
    # Unit cell vertices in fractional coordinates
    frac_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    
    # Convert to cartesian coordinates
    cart_vertices = np.array([
        lattice.get_cartesian_coords(frac_coord) for frac_coord in frac_vertices
    ])
    
    return cart_vertices

def get_cell_edges() -> List[Tuple[int, int]]:
    """
    Get edge connectivity for unit cell
    
    Returns:
        List of vertex index pairs defining edges
    """
    return [
        # Bottom face edges
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face edges  
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

def get_cell_faces() -> List[List[int]]:
    """
    Get face connectivity for unit cell
    
    Returns:
        List of vertex index lists defining faces
    """
    return [
        [0, 1, 2, 3],  # Bottom face (z=0)
        [4, 7, 6, 5],  # Top face (z=1)
        [0, 4, 5, 1],  # Front face (y=0)
        [2, 6, 7, 3],  # Back face (y=1)
        [0, 3, 7, 4],  # Left face (x=0)
        [1, 5, 6, 2]   # Right face (x=1)
    ]

def get_cell_face_centers(lattice: Lattice) -> Dict[str, np.ndarray]:
    """
    Get centers of unit cell faces
    
    Args:
        lattice: Crystal lattice
        
    Returns:
        Dictionary mapping face names to center coordinates
    """
    vertices = get_cell_vertices(lattice)
    faces = get_cell_faces()
    
    face_names = ['bottom', 'top', 'front', 'back', 'left', 'right']
    face_centers = {}
    
    for name, face_indices in zip(face_names, faces):
        face_vertices = vertices[face_indices]
        center = np.mean(face_vertices, axis=0)
        face_centers[name] = center
    
    return face_centers

def draw_cell_edges(
    fig: go.Figure,
    vertices: np.ndarray,
    *,
    is_3d: bool = True,
    edge_kwargs: Optional[Dict[str, Any]] = None,
    scene: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None
) -> None:
    """
    Draw unit cell edges
    
    Args:
        fig: Plotly figure object
        vertices: Cell vertex coordinates
        is_3d: Whether this is a 3D plot
        edge_kwargs: Edge line styling options
        scene: Scene identifier for 3D plots
        row: Subplot row for 2D plots
        col: Subplot column for 2D plots
    """
    default_edge_kwargs = {
        "color": "black",
        "width": 2,
        "dash": "solid"
    }
    if edge_kwargs:
        default_edge_kwargs.update(edge_kwargs)
    
    edges = get_cell_edges()
    
    # Collect all edge coordinates
    x_coords, y_coords, z_coords = [], [], []
    
    for start_idx, end_idx in edges:
        start_vertex = vertices[start_idx]
        end_vertex = vertices[end_idx]
        
        x_coords.extend([start_vertex[0], end_vertex[0], None])
        y_coords.extend([start_vertex[1], end_vertex[1], None])
        if is_3d:
            z_coords.extend([start_vertex[2], end_vertex[2], None])
    
    trace_kwargs = {
        "mode": "lines",
        "line": default_edge_kwargs,
        "showlegend": False,
        "hoverinfo": "skip",
        "name": "cell_edges"
    }
    
    if is_3d:
        trace_kwargs.update({
            "x": x_coords, "y": y_coords, "z": z_coords
        })
        if scene:
            trace_kwargs["scene"] = scene
        fig.add_scatter3d(**trace_kwargs)
    else:
        trace_kwargs.update({
            "x": x_coords, "y": y_coords
        })
        if row and col:
            fig.add_scatter(row=row, col=col, **trace_kwargs)
        else:
            fig.add_scatter(**trace_kwargs)

def draw_cell_faces(
    fig: go.Figure,
    vertices: np.ndarray,
    *,
    face_kwargs: Optional[Dict[str, Any]] = None,
    scene: Optional[str] = None,
    opacity: float = 0.1,
    color: str = "lightblue"
) -> None:
    """
    Draw unit cell faces as transparent surfaces
    
    Args:
        fig: Plotly figure object
        vertices: Cell vertex coordinates
        face_kwargs: Face styling options
        scene: Scene identifier
        opacity: Face transparency
        color: Face color
    """
    default_face_kwargs = {
        "opacity": opacity,
        "color": color,
        "showscale": False
    }
    if face_kwargs:
        default_face_kwargs.update(face_kwargs)
    
    faces = get_cell_faces()
    
    # Create triangulated faces for mesh3d
    triangulated_faces = []
    for face in faces:
        # Split quadrilateral face into two triangles
        triangulated_faces.extend([
            [face[0], face[1], face[2]],
            [face[0], face[2], face[3]]
        ])
    
    # Flatten triangulated faces
    i_indices = [tri[0] for tri in triangulated_faces]
    j_indices = [tri[1] for tri in triangulated_faces]
    k_indices = [tri[2] for tri in triangulated_faces]
    
    trace_kwargs = {
        "x": vertices[:, 0],
        "y": vertices[:, 1],
        "z": vertices[:, 2],
        "i": i_indices,
        "j": j_indices,
        "k": k_indices,
        "name": "cell_faces",
        "showlegend": False,
        "hoverinfo": "skip",
        **default_face_kwargs
    }
    
    if scene:
        trace_kwargs["scene"] = scene
    
    fig.add_mesh3d(**trace_kwargs)

def draw_cell(
    fig: go.Figure,
    structure: Structure,
    *,
    cell_kwargs: Dict[str, Any] = None,
    is_3d: bool = True,
    scene: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    show_faces: bool = True,
    show_edges: bool = True,
    show_axes: bool = False,
    show_face_labels: bool = False,
    **kwargs
) -> None:
    """
    Draw complete unit cell with edges and faces
    
    Args:
        fig: Plotly figure object
        structure: Crystal structure
        cell_kwargs: Cell styling options
        is_3d: Whether this is a 3D plot
        scene: Scene identifier
        row: Subplot row
        col: Subplot column
        show_faces: Whether to show cell faces
        show_edges: Whether to show cell edges
        show_axes: Whether to show lattice axes
        show_face_labels: Whether to label faces
        **kwargs: Additional arguments
    """
    if cell_kwargs is None:
        cell_kwargs = {}
    
    # Get cell vertices
    vertices = get_cell_vertices(structure.lattice)
    
    # Draw cell edges
    if show_edges:
        edge_style = cell_kwargs.get('edge', {})
        draw_cell_edges(
            fig, vertices, is_3d=is_3d, edge_kwargs=edge_style,
            scene=scene, row=row, col=col
        )
    
    # Draw cell faces (3D only)
    if show_faces and is_3d:
        face_style = cell_kwargs.get('face', {})
        draw_cell_faces(
            fig, vertices, face_kwargs=face_style, scene=scene
        )
    
    # Draw lattice axes
    if show_axes:
        _draw_cell_axes(
            fig, structure.lattice, is_3d=is_3d, 
            scene=scene, row=row, col=col, **cell_kwargs.get('axes', {})
        )
    
    # Add face labels
    if show_face_labels and is_3d:
        _add_face_labels(fig, structure.lattice, scene=scene)

def _draw_cell_axes(
    fig: go.Figure,
    lattice: Lattice,
    *,
    is_3d: bool = True,
    scene: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    axis_colors: List[str] = None,
    axis_width: float = 3,
    axis_length_scale: float = 1.0
) -> None:
    """
    Draw lattice axes (a, b, c vectors)
    
    Args:
        fig: Plotly figure object
        lattice: Crystal lattice
        is_3d: Whether this is a 3D plot
        scene: Scene identifier
        row: Subplot row
        col: Subplot column
        axis_colors: Colors for a, b, c axes
        axis_width: Width of axis lines
        axis_length_scale: Scaling factor for axis length
    """
    if axis_colors is None:
        axis_colors = ["red", "green", "blue"]
    
    origin = np.array([0, 0, 0])
    axes = lattice.matrix * axis_length_scale
    axis_labels = ['a', 'b', 'c']
    
    for i, (axis, color, label) in enumerate(zip(axes, axis_colors, axis_labels)):
        end_point = origin + axis
        
        trace_kwargs = {
            "mode": "lines+text",
            "line": dict(color=color, width=axis_width),
            "showlegend": False,
            "name": f"axis_{label}",
            "text": [None, label],
            "textposition": "top center",
            "textfont": dict(size=12, color=color)
        }
        
        if is_3d:
            trace_kwargs.update({
                "x": [origin[0], end_point[0]],
                "y": [origin[1], end_point[1]],
                "z": [origin[2], end_point[2]]
            })
            if scene:
                trace_kwargs["scene"] = scene
            fig.add_scatter3d(**trace_kwargs)
        else:
            trace_kwargs.update({
                "x": [origin[0], end_point[0]],
                "y": [origin[1], end_point[1]]
            })
            if row and col:
                fig.add_scatter(row=row, col=col, **trace_kwargs)
            else:
                fig.add_scatter(**trace_kwargs)

def _add_face_labels(
    fig: go.Figure,
    lattice: Lattice,
    scene: Optional[str] = None
) -> None:
    """
    Add labels to unit cell faces
    
    Args:
        fig: Plotly figure object
        lattice: Crystal lattice
        scene: Scene identifier
    """
    face_centers = get_cell_face_centers(lattice)
    face_labels = {
        'bottom': '(001) bottom',
        'top': '(001) top', 
        'front': '(010) front',
        'back': '(010) back',
        'left': '(100) left',
        'right': '(100) right'
    }
    
    for face_name, center in face_centers.items():
        label = face_labels.get(face_name, face_name)
        
        trace_kwargs = {
            "x": [center[0]],
            "y": [center[1]], 
            "z": [center[2]],
            "mode": "text",
            "text": [label],
            "textfont": dict(size=8, color="black"),
            "textposition": "middle center",
            "showlegend": False,
            "hoverinfo": "skip",
            "name": f"face_label_{face_name}"
        }
        
        if scene:
            trace_kwargs["scene"] = scene
        
        fig.add_scatter3d(**trace_kwargs)

def calculate_cell_volume(lattice: Lattice) -> float:
    """
    Calculate unit cell volume
    
    Args:
        lattice: Crystal lattice
        
    Returns:
        Unit cell volume in Ų
    """
    return lattice.volume

def get_cell_parameters(lattice: Lattice) -> Dict[str, float]:
    """
    Get unit cell parameters
    
    Args:
        lattice: Crystal lattice
        
    Returns:
        Dictionary with lattice parameters
    """
    return {
        'a': lattice.a,
        'b': lattice.b,
        'c': lattice.c,
        'alpha': lattice.alpha,
        'beta': lattice.beta,
        'gamma': lattice.gamma,
        'volume': lattice.volume
    }

def draw_multiple_cells(
    fig: go.Figure,
    structure: Structure,
    *,
    nx: int = 1,
    ny: int = 1, 
    nz: int = 1,
    **kwargs
) -> None:
    """
    Draw multiple unit cells (supercell visualization)
    
    Args:
        fig: Plotly figure object
        structure: Crystal structure
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        nz: Number of cells in z direction
        **kwargs: Additional arguments for draw_cell
    """
    lattice = structure.lattice
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate translation vector
                translation = i * lattice.matrix[0] + j * lattice.matrix[1] + k * lattice.matrix[2]
                
                # Create translated vertices
                base_vertices = get_cell_vertices(lattice)
                translated_vertices = base_vertices + translation
                
                # Draw translated cell
                edge_style = kwargs.get('cell_kwargs', {}).get('edge', {})
                draw_cell_edges(
                    fig, translated_vertices,
                    is_3d=kwargs.get('is_3d', True),
                    edge_kwargs=edge_style,
                    scene=kwargs.get('scene')
                )
