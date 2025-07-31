"""
Vector rendering functions for forces, magnetic moments, and other vector properties
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go

def normalize_vector_length(
    vectors: np.ndarray,
    target_length: float = 1.0,
    preserve_direction: bool = True
) -> np.ndarray:
    """
    Normalize vector lengths while preserving direction
    
    Args:
        vectors: Array of vectors (Nx3)
        target_length: Target length for normalized vectors
        preserve_direction: Whether to preserve original direction
        
    Returns:
        Normalized vectors array
    """
    if not preserve_direction:
        return np.full_like(vectors, target_length)
    
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    magnitudes[magnitudes == 0] = 1  # Avoid division by zero
    
    normalized = vectors / magnitudes * target_length
    return normalized

def get_vector_components(vector: np.ndarray) -> Dict[str, float]:
    """
    Get vector components and magnitude
    
    Args:
        vector: 3D vector array
        
    Returns:
        Dictionary with vector components and properties
    """
    return {
        'x': float(vector[0]),
        'y': float(vector[1]),
        'z': float(vector[2]),
        'magnitude': float(np.linalg.norm(vector)),
        'unit_vector': vector / np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else vector
    }

def create_arrow_coordinates(
    start: np.ndarray,
    end: np.ndarray,
    head_size: float = 0.1,
    head_angle: float = 20.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create arrow coordinates including arrowhead
    
    Args:
        start: Starting point of arrow
        end: Ending point of arrow
        head_size: Size of arrowhead relative to arrow length
        head_angle: Angle of arrowhead in degrees
        
    Returns:
        Tuple of (shaft_coords, head_coords)
    """
    vector = end - start
    length = np.linalg.norm(vector)
    
    if length == 0:
        return np.array([start, end]), np.array([])
    
    unit_vector = vector / length
    head_length = length * head_size
    
    # Calculate arrowhead points
    angle_rad = np.radians(head_angle)
    
    # Create perpendicular vectors for arrowhead
    if abs(unit_vector[2]) < 0.9:
        perp1 = np.cross(unit_vector, [0, 0, 1])
    else:
        perp1 = np.cross(unit_vector, [1, 0, 0])
    
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(unit_vector, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    # Arrowhead base point
    head_base = end - unit_vector * head_length
    
    # Arrowhead wing points
    wing_offset = head_length * np.tan(angle_rad)
    wing1 = head_base + perp1 * wing_offset
    wing2 = head_base + perp2 * wing_offset
    wing3 = head_base - perp1 * wing_offset
    wing4 = head_base - perp2 * wing_offset
    
    shaft_coords = np.array([start, head_base])
    head_coords = np.array([
        head_base, wing1, end, wing2, head_base,
        head_base, wing3, end, wing4, head_base
    ])
    
    return shaft_coords, head_coords

def draw_arrow_3d(
    fig: go.Figure,
    start: np.ndarray,
    vector: np.ndarray,
    *,
    color: str = "red",
    width: float = 3,
    head_size: float = 0.15,
    opacity: float = 1.0,
    scene: Optional[str] = None,
    name: str = "arrow",
    show_in_legend: bool = False,
    scale_factor: float = 1.0
) -> None:
    """
    Draw a 3D arrow with proper arrowhead
    
    Args:
        fig: Plotly figure object
        start: Starting point of arrow
        vector: Vector direction and magnitude
        color: Arrow color
        width: Arrow line width
        head_size: Relative size of arrowhead
        opacity: Arrow opacity
        scene: Scene identifier
        name: Arrow name
        show_in_legend: Whether to show in legend
        scale_factor: Scaling factor for arrow size
    """
    if np.linalg.norm(vector) == 0:
        return
    
    scaled_vector = vector * scale_factor
    end = start + scaled_vector
    
    # Create arrow coordinates
    shaft_coords, head_coords = create_arrow_coordinates(start, end, head_size)
    
    # Draw arrow shaft
    shaft_trace = go.Scatter3d(
        x=[shaft_coords[0, 0], shaft_coords[1, 0]],
        y=[shaft_coords[0, 1], shaft_coords[1, 1]],
        z=[shaft_coords[0, 2], shaft_coords[1, 2]],
        mode="lines",
        line=dict(color=color, width=width),
        opacity=opacity,
        showlegend=show_in_legend,
        name=name,
        hoverinfo="skip"
    )
    
    if scene:
        shaft_trace.scene = scene
    
    fig.add_trace(shaft_trace)
    
    # Draw arrowhead
    if len(head_coords) > 0:
        head_x = head_coords[:, 0].tolist() + [None] * (len(head_coords) // 5)
        head_y = head_coords[:, 1].tolist() + [None] * (len(head_coords) // 5) 
        head_z = head_coords[:, 2].tolist() + [None] * (len(head_coords) // 5)
        
        # Interleave coordinates with None for separate line segments
        final_x, final_y, final_z = [], [], []
        for i in range(0, len(head_coords), 5):
            final_x.extend(head_coords[i:i+5, 0].tolist() + [None])
            final_y.extend(head_coords[i:i+5, 1].tolist() + [None])
            final_z.extend(head_coords[i:i+5, 2].tolist() + [None])
        
        head_trace = go.Scatter3d(
            x=final_x,
            y=final_y,
            z=final_z,
            mode="lines",
            line=dict(color=color, width=width),
            opacity=opacity,
            showlegend=False,
            name=f"{name}_head",
            hoverinfo="skip"
        )
        
        if scene:
            head_trace.scene = scene
        
        fig.add_trace(head_trace)

def draw_vector(
    fig: go.Figure,
    origin: np.ndarray,
    vector: np.ndarray,
    *,
    is_3d: bool = True,
    arrow_kwargs: Dict[str, Any] = None,
    scene: Optional[str] = None,
    row: Optional[int] = None,
    col: Optional[int] = None,
    name: str = "vector",
    vector_type: str = "force",
    auto_scale: bool = True,
    **kwargs
) -> None:
    """
    Draw a vector as an arrow
    
    Args:
        fig: Plotly figure object
        origin: Vector origin point
        vector: Vector direction and magnitude
        is_3d: Whether this is a 3D plot
        arrow_kwargs: Arrow styling options
        scene: Scene identifier for 3D plots
        row: Subplot row for 2D plots
        col: Subplot column for 2D plots
        name: Vector name
        vector_type: Type of vector (affects default styling)
        auto_scale: Whether to auto-scale vector length
        **kwargs: Additional arguments
    """
    if arrow_kwargs is None:
        arrow_kwargs = {}
    
    # Default styling based on vector type
    default_styles = {
        "force": {"color": "red", "width": 3, "scale_factor": 2.0},
        "magmom": {"color": "blue", "width": 3, "scale_factor": 1.5},
        "velocity": {"color": "green", "width": 2, "scale_factor": 1.0},
        "displacement": {"color": "orange", "width": 2, "scale_factor": 1.0}
    }
    
    style = default_styles.get(vector_type, default_styles["force"])
    style.update(arrow_kwargs)
    
    # Auto-scale vector if requested
    if auto_scale:
        magnitude = np.linalg.norm(vector)
        if magnitude > 0:
            # Scale based on typical atomic distances
            typical_scale = 2.0  # Angstroms
            if magnitude > typical_scale:
                vector = vector * (typical_scale / magnitude)
    
    if is_3d:
        draw_arrow_3d(
            fig, origin, vector,
            scene=scene, name=name, **style
        )
    else:
        # 2D arrow (simplified)
        end = origin + vector
        
        trace_kwargs = {
            "x": [origin[0], end[0]],
            "y": [origin[1], end[1]],
            "mode": "lines+markers",
            "line": dict(color=style.get("color", "red"), width=style.get("width", 3)),
            "marker": dict(
                symbol="arrow", size=10, angleref="previous",
                color=style.get("color", "red")
            ),
            "showlegend": style.get("show_in_legend", False),
            "name": name
        }
        
        if row and col:
            fig.add_scatter(row=row, col=col, **trace_kwargs)
        else:
            fig.add_scatter(**trace_kwargs)

def batch_draw_vectors(
    fig: go.Figure,
    origins: np.ndarray,
    vectors: np.ndarray,
    *,
    vector_types: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    is_3d: bool = True,
    **kwargs
) -> None:
    """
    Draw multiple vectors efficiently in batch
    
    Args:
        fig: Plotly figure object
        origins: Array of vector origins (Nx3)
        vectors: Array of vectors (Nx3)
        vector_types: List of vector types for styling
        colors: List of colors for each vector
        is_3d: Whether this is a 3D plot
        **kwargs: Additional arguments
    """
    n_vectors = len(origins)
    if len(vectors) != n_vectors:
        raise ValueError("Origins and vectors must have same length")
    
    # Default values
    if vector_types is None:
        vector_types = ["force"] * n_vectors
    if colors is None:
        colors = ["red"] * n_vectors
    
    # Group vectors by type and color for efficient rendering
    vector_groups = {}
    for i, (origin, vector, vec_type, color) in enumerate(
        zip(origins, vectors, vector_types, colors)
    ):
        key = (vec_type, color)
        if key not in vector_groups:
            vector_groups[key] = {"origins": [], "vectors": [], "indices": []}
        
        vector_groups[key]["origins"].append(origin)
        vector_groups[key]["vectors"].append(vector)
        vector_groups[key]["indices"].append(i)
    
    # Draw each group
    for (vec_type, color), group_data in vector_groups.items():
        group_origins = np.array(group_data["origins"])
        group_vectors = np.array(group_data["vectors"])
        
        # Draw vectors in this group
        for origin, vector in zip(group_origins, group_vectors):
            draw_vector(
                fig, origin, vector,
                is_3d=is_3d,
                arrow_kwargs={"color": color},
                vector_type=vec_type,
                name=f"{vec_type}_{color}",
                **kwargs
            )

def create_vector_field_visualization(
    fig: go.Figure,
    grid_points: np.ndarray,
    vectors: np.ndarray,
    *,
    subsample: int = 1,
    **kwargs
) -> None:
    """
    Create a vector field visualization
    
    Args:
        fig: Plotly figure object
        grid_points: Grid points for vector field (Nx3)
        vectors: Vectors at each grid point (Nx3)
        subsample: Subsampling factor to reduce density
        **kwargs: Additional arguments
    """
    # Subsample for better visualization
    indices = np.arange(0, len(grid_points), subsample)
    subsampled_points = grid_points[indices]
    subsampled_vectors = vectors[indices]
    
    # Normalize vectors for consistent visualization
    normalized_vectors = normalize_vector_length(
        subsampled_vectors, target_length=0.5
    )
    
    batch_draw_vectors(
        fig, subsampled_points, normalized_vectors,
        **kwargs
    )

def analyze_vector_statistics(vectors: np.ndarray) -> Dict[str, Any]:
    """
    Analyze statistical properties of vectors
    
    Args:
        vectors: Array of vectors (Nx3)
        
    Returns:
        Dictionary with vector statistics
    """
    magnitudes = np.linalg.norm(vectors, axis=1)
    
    return {
        'count': len(vectors),
        'magnitude_stats': {
            'min': float(np.min(magnitudes)),
            'max': float(np.max(magnitudes)),
            'mean': float(np.mean(magnitudes)),
            'std': float(np.std(magnitudes))
        },
        'component_stats': {
            'x': {'mean': float(np.mean(vectors[:, 0])), 'std': float(np.std(vectors[:, 0]))},
            'y': {'mean': float(np.mean(vectors[:, 1])), 'std': float(np.std(vectors[:, 1]))},
            'z': {'mean': float(np.mean(vectors[:, 2])), 'std': float(np.std(vectors[:, 2]))}
        },
        'net_vector': {
            'x': float(np.sum(vectors[:, 0])),
            'y': float(np.sum(vectors[:, 1])),
            'z': float(np.sum(vectors[:, 2])),
            'magnitude': float(np.linalg.norm(np.sum(vectors, axis=0)))
        }
    }

def create_vector_magnitude_colormap(
    vectors: np.ndarray,
    colorscale: str = "viridis"
) -> List[str]:
    """
    Create color mapping based on vector magnitudes
    
    Args:
        vectors: Array of vectors (Nx3)
        colorscale: Colorscale name
        
    Returns:
        List of colors corresponding to vector magnitudes
    """
    import plotly.colors as pcolors
    
    magnitudes = np.linalg.norm(vectors, axis=1)
    
    if len(magnitudes) == 0:
        return []
    
    # Normalize magnitudes to [0, 1]
    min_mag, max_mag = np.min(magnitudes), np.max(magnitudes)
    if max_mag > min_mag:
        normalized_mags = (magnitudes - min_mag) / (max_mag - min_mag)
    else:
        normalized_mags = np.zeros_like(magnitudes)
    
    # Get colorscale
    if hasattr(pcolors.sequential, colorscale.title()):
        scale = getattr(pcolors.sequential, colorscale.title())
    else:
        scale = pcolors.sequential.Viridis
    
    # Sample colors based on normalized magnitudes
    colors = pcolors.sample_colorscale(scale, normalized_mags)
    
    return colors
