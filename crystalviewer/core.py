from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import warnings
import os
from pathlib import Path

try:
    import pyvista as pv

    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    warnings.warn("PyVista not available - rendering disabled")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional dependencies for structure objects
try:
    from pymatgen.core import Structure as PymatgenStructure

    HAS_PYMATGEN = True
except ImportError:
    HAS_PYMATGEN = False
    PymatgenStructure = None

try:
    import ase
    from ase import Atoms

    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    Atoms = None

from .config import (
    get_quality_settings,
    get_atomic_radius,
    get_element_color,
    get_camera_preset,
    DEFAULT_RENDER_SETTINGS,
    PERFORMANCE_SETTINGS,
)
from .geometry import Vector3D, calculate_distance, calculate_center_of_mass
from .rendering import (
    atom_renderer,
    bond_renderer,
    cell_renderer,
    vector_renderer,
    label_renderer,
)
from .styling import Material, ColorScheme, get_element_material
from .utils import (
    performance_monitor,
    timing_decorator,
    create_directory,
    check_memory_limit,
    get_optimal_batch_size,
)


# Image sites functionality
def get_image_shifts(frac_coords: np.ndarray, tol: float = 0.05) -> set:
    """
    Get lattice shifts needed to create image sites for sites near cell boundaries

    Args:
        frac_coords: Fractional coordinates
        tol: Tolerance for boundary detection

    Returns:
        Set of (i,j,k) shift tuples
    """
    shifts = set()

    # Check each dimension for boundary proximity
    for i, coord in enumerate(frac_coords):
        if coord < tol:
            # Near lower boundary
            shift = [0, 0, 0]
            shift[i] = -1
            shifts.add(tuple(shift))
        elif coord > 1 - tol:
            # Near upper boundary
            shift = [0, 0, 0]
            shift[i] = 1
            shifts.add(tuple(shift))

    # Add corner/edge cases for 2D boundaries
    if len([c for c in frac_coords if c < tol or c > 1 - tol]) >= 2:
        for i in range(3):
            for j in range(i + 1, 3):
                if (frac_coords[i] < tol or frac_coords[i] > 1 - tol) and (
                    frac_coords[j] < tol or frac_coords[j] > 1 - tol
                ):
                    # Add diagonal shifts
                    for si in [-1, 1]:
                        for sj in [-1, 1]:
                            shift = [0, 0, 0]
                            shift[i] = si
                            shift[j] = sj
                            shifts.add(tuple(shift))

    # remove the origin if present
    shifts.discard((0, 0, 0))
    return shifts


def get_image_sites_from_shifts(
    position: List[float],
    lattice_vectors: List[List[float]],
    cell_boundary_tol: float = 0.05,
    min_dist_dedup: float = 1e-5,
) -> np.ndarray:
    """
    Compute image-site cartesian coordinates for sites near cell boundaries

    Args:
        position: Cartesian position of the site
        lattice_vectors: Lattice vectors defining the unit cell
        cell_boundary_tol: Tolerance for boundary detection
        min_dist_dedup: Minimum distance to avoid duplicates

    Returns:
        Array of image site cartesian coordinates
    """
    if not HAS_NUMPY:
        return np.array([]).reshape(0, 3)

    # Convert cartesian to fractional coordinates
    lattice_matrix = np.array(lattice_vectors)
    frac_coords = np.dot(np.array(position), np.linalg.inv(lattice_matrix))

    # Get required shifts
    shifts = get_image_shifts(frac_coords, tol=cell_boundary_tol)
    if not shifts:
        return np.array([]).reshape(0, 3)

    # Convert to cartesian coordinates
    coords = []
    for shift in shifts:
        new_frac = frac_coords + np.array(shift)
        cart = np.dot(new_frac, lattice_matrix)

        # Dedupe by distance from original
        if np.linalg.norm(cart - np.array(position)) > min_dist_dedup:
            coords.append(cart)

    return np.array(coords) if coords else np.array([]).reshape(0, 3)


class CrystalRenderer:
    """
    Crystal structure renderer using PyVista
    """

    def __init__(self, quality: str = "medium", offscreen: bool = True):
        """
        Initialize the renderer

        Args:
            quality: Rendering quality ('low', 'medium', 'high', 'ultra')
            offscreen: Whether to use offscreen rendering
        """
        if not HAS_PYVISTA:
            raise ImportError("PyVista is required for CrystalRenderer")

        self.quality = quality
        self.offscreen = offscreen
        self.plotter = None
        self.scene_objects = []

        self.quality_settings = get_quality_settings(quality)

        self.performance_settings = PERFORMANCE_SETTINGS.copy()

        self._initialize_plotter()

    def _initialize_plotter(self):
        """Initialize PyVista plotter with optimized settings"""
        if self.offscreen:
            self.plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        else:
            self.plotter = pv.Plotter(window_size=(800, 600))

        # Apply default rendering settings
        self.plotter.set_background(DEFAULT_RENDER_SETTINGS["background_color"])

        # Enable GPU acceleration if available
        if hasattr(self.plotter, "enable_gpu_acceleration"):
            try:
                self.plotter.enable_gpu_acceleration()
            except:
                pass  # GPU acceleration not available

    def clear_scene(self):
        """Clear all objects from the scene"""
        if self.plotter:
            self.plotter.clear()
        self.scene_objects.clear()

    def add_structure(self, structure: Any, **kwargs) -> None:
        """
        Add crystal structure to the scene
        Supports pymatgen Structure, ASE Atoms, position/element arrays, or dict

        Args:
            structure: Structure object, positions list, or dict with structure data
            **kwargs: Additional rendering options including:
                - show_atoms (bool): Whether to show atoms (default: True)
                - show_bonds (bool): Whether to show bonds (default: False)
                - show_cell (bool): Whether to show unit cell (default: True)
                - atom_scale (float): Scale factor for atom sizes (default: 1.0)
                - color_scheme (str): Color scheme for atoms ('vesta', 'jmol', 'cpk') (default: 'vesta')
                - sphere_resolution (int): Resolution for atom spheres (default: quality-dependent)
                - bond_cutoff (float): Maximum bond distance (default: 3.0)
                - bond_radius (float): Radius of bond cylinders (default: 0.1)
                - show_faces (bool): Whether to show cell faces (default: True)
                - show_edges (bool): Whether to show cell edges (default: True)
                - face_opacity (float): Opacity of cell faces (default: 0.1)
                - show_labels (bool): Whether to show element labels (default: False)
                - show_image_sites (bool): Whether to show image sites on cell boundaries (default: False)
                - cell_boundary_tol (float): Tolerance for cell boundary detection (default: 0.05)
        """
        performance_monitor.start("add_structure")

        try:
            # Handle different input types
            if isinstance(structure, (list, tuple)) and len(structure) >= 2:
                # Handle positions, elements arrays: (positions, elements)
                positions = structure[0]
                elements = structure[1]
                lattice_vectors = kwargs.get("lattice_vectors")
            elif isinstance(structure, dict):
                # Handle dict with 'positions', 'elements', 'lattice_vectors' keys
                positions = structure.get("positions", [])
                elements = structure.get("elements", [])
                lattice_vectors = structure.get("lattice_vectors")
                if lattice_vectors and "lattice_vectors" not in kwargs:
                    kwargs["lattice_vectors"] = lattice_vectors
            else:
                # Handle structure objects (pymatgen Structure, ASE Atoms)
                positions, elements, lattice_vectors = convert_structure_to_arrays(
                    structure
                )
                if lattice_vectors and "lattice_vectors" not in kwargs:
                    kwargs["lattice_vectors"] = lattice_vectors

            # Convert positions to Vector3D
            pos_vectors = [Vector3D(*pos) for pos in positions]

            # Render atoms
            if kwargs.get("show_atoms", True):
                self._add_atoms(pos_vectors, elements, **kwargs)

            # Render image sites (atoms near cell boundaries)
            if kwargs.get("show_image_sites", False) and lattice_vectors:
                self._add_image_sites(positions, elements, lattice_vectors, **kwargs)

            # Render bonds
            if kwargs.get("show_bonds", False):
                self._add_bonds(pos_vectors, elements, **kwargs)

            # Render unit cell
            if lattice_vectors and kwargs.get("show_cell", True):
                # Remove lattice_vectors from kwargs to avoid duplicate argument error
                cell_kwargs = kwargs.copy()
                cell_kwargs.pop("lattice_vectors", None)
                # Filter kwargs to only pass arguments relevant to cell rendering
                cell_render_kwargs = {
                    "show_faces": cell_kwargs.get("show_faces", True),
                    "show_edges": cell_kwargs.get("show_edges", True),
                    "face_opacity": cell_kwargs.get("face_opacity", 0.1),
                }
                self._add_unit_cell(lattice_vectors, **cell_render_kwargs)

            # Render vectors (forces, magnetic moments, etc.)
            if "vectors" in kwargs:
                self._add_vectors(kwargs["vectors"], **kwargs)

            # Add labels
            if kwargs.get("show_labels", False):
                self._add_labels(pos_vectors, elements, **kwargs)

        finally:
            performance_monitor.stop("add_structure")

    def _add_atoms(self, positions: List[Vector3D], elements: List[str], **kwargs):
        """Add atoms to the scene"""
        scale = kwargs.get("atom_scale", 1.0)
        color_scheme = kwargs.get("color_scheme", "vesta")
        sphere_resolution = kwargs.get("sphere_resolution", None)

        # Batch render atoms for performance
        batch_size = get_optimal_batch_size()
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i : i + batch_size]
            batch_elements = elements[i : i + batch_size]

            atoms = atom_renderer.render_atoms_batch(
                batch_positions,
                batch_elements,
                scale,
                color_scheme=color_scheme,
                sphere_resolution=sphere_resolution,
            )

            for atom in atoms:
                self._add_mesh_to_scene(atom)

    def _add_bonds(self, positions: List[Vector3D], elements: List[str], **kwargs):
        """Add bonds to the scene"""
        bond_cutoff = kwargs.get("bond_cutoff", 3.0)
        bond_radius = kwargs.get("bond_radius", 0.1)

        # Find bonds
        bonds = self._find_bonds(positions, elements, bond_cutoff)

        # Batch render bonds
        batch_size = get_optimal_batch_size()
        for i in range(0, len(bonds), batch_size):
            batch_bonds = bonds[i : i + batch_size]
            bond_objects = bond_renderer.render_bonds_batch(batch_bonds, bond_radius)

            for bond in bond_objects:
                self._add_mesh_to_scene(bond)

    def _find_bonds(
        self, positions: List[Vector3D], elements: List[str], cutoff: float
    ) -> List[Tuple[Vector3D, Vector3D, str]]:
        """Find bonds between atoms within cutoff distance"""
        bonds = []

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = calculate_distance(positions[i], positions[j])
                if distance <= cutoff:
                    bond_type = self._determine_bond_type(
                        elements[i], elements[j], distance
                    )
                    bonds.append((positions[i], positions[j], bond_type))

        return bonds

    def _determine_bond_type(self, elem1: str, elem2: str, distance: float) -> str:
        """Determine bond type based on elements and distance"""
        # Simple bond type determination
        # In practice, this would use more sophisticated bonding analysis
        radius1 = get_atomic_radius(elem1)
        radius2 = get_atomic_radius(elem2)
        expected_distance = radius1 + radius2

        if distance < expected_distance * 0.8:
            return "triple"
        elif distance < expected_distance * 0.9:
            return "double"
        else:
            return "single"

    def _add_unit_cell(self, lattice_vectors: List[List[float]], **kwargs):
        """Add unit cell to the scene"""
        # Convert to Vector3D
        vectors = [Vector3D(*vec) for vec in lattice_vectors]

        cell_data = cell_renderer.render_unit_cell(vectors, **kwargs)
        self._add_mesh_to_scene(cell_data)

    def _add_vectors(self, vectors: List[Dict[str, Any]], **kwargs):
        """Add vectors (forces, magnetic moments, etc.) to the scene"""
        scale = kwargs.get("vector_scale", 1.0)

        for vector_data in vectors:
            start_pos = Vector3D(*vector_data["position"])
            direction = Vector3D(*vector_data["direction"])
            color = vector_data.get("color", (1.0, 0.0, 0.0))

            vector_obj = vector_renderer.render_vector(
                start_pos, direction, color, scale
            )
            self._add_mesh_to_scene(vector_obj)

    def _add_labels(self, positions: List[Vector3D], elements: List[str], **kwargs):
        """Add labels to the scene"""
        for pos, elem in zip(positions, elements):
            label_renderer.add_label(pos, elem, **kwargs)

        labels = label_renderer.render_labels()
        for label in labels:
            self._add_text_to_scene(label)

    def _add_mesh_to_scene(self, mesh_data: Dict[str, Any]):
        """Add mesh object to PyVista scene"""
        try:
            if mesh_data["type"] == "sphere":
                self._add_sphere_mesh(mesh_data)
            elif mesh_data["type"] == "cylinder":
                self._add_cylinder_mesh(mesh_data)
            elif mesh_data["type"] == "cell":
                self._add_cell_mesh(mesh_data)
            elif mesh_data["type"] == "arrow":
                self._add_arrow_mesh(mesh_data)
            # Add to scene objects list
            self.scene_objects.append(mesh_data)
        except Exception as e:
            warnings.warn(f"Failed to add mesh to scene: {e}")

    def _add_sphere_mesh(self, mesh_data: Dict[str, Any]):
        """Add sphere mesh to scene"""
        position = mesh_data["position"]
        mesh = mesh_data["mesh"]
        material = mesh_data["material"]

        # Create PyVista sphere
        sphere = pv.Sphere(
            radius=mesh["radius"],
            center=position,
            theta_resolution=mesh["resolution"],
            phi_resolution=mesh["resolution"],
        )

        # Apply material
        self.plotter.add_mesh(
            sphere,
            color=material["color"],
            opacity=material["opacity"],
            show_edges=False,
        )

    def _add_cylinder_mesh(self, mesh_data: Dict[str, Any]):
        """Add cylinder mesh to scene"""
        start_pos = mesh_data["start_position"]
        end_pos = mesh_data["end_position"]
        mesh = mesh_data["mesh"]
        material = mesh_data["material"]

        # Create PyVista cylinder
        direction = np.array(end_pos) - np.array(start_pos)
        height = np.linalg.norm(direction)
        center = (np.array(start_pos) + np.array(end_pos)) / 2

        cylinder = pv.Cylinder(
            center=center,
            direction=direction,
            radius=mesh["radius"],
            height=height,
            resolution=mesh["resolution"],
        )

        # Apply material
        self.plotter.add_mesh(
            cylinder,
            color=material["color"],
            opacity=material["opacity"],
            show_edges=False,
        )

    def _add_cell_mesh(self, mesh_data: Dict[str, Any]):
        """Add unit cell mesh to scene"""
        vertices = mesh_data["vertices"]

        if mesh_data.get("show_faces", True):
            # Create faces
            faces = [
                [0, 1, 2, 3],  # Bottom
                [4, 7, 6, 5],  # Top
                [0, 4, 5, 1],  # Front
                [2, 6, 7, 3],  # Back
                [0, 3, 7, 4],  # Left
                [1, 5, 6, 2],  # Right
            ]

            for face in faces:
                face_vertices = [vertices[i] for i in face]
                face_mesh = pv.Polygon(face_vertices)
                self.plotter.add_mesh(
                    face_mesh,
                    color=mesh_data["face_material"]["color"],
                    opacity=mesh_data["face_opacity"],
                    show_edges=False,
                )

        if mesh_data.get("show_edges", True):
            # Create edges
            edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # Bottom
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # Top
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # Vertical
            ]

            for edge in edges:
                start = vertices[edge[0]]
                end = vertices[edge[1]]
                line = pv.Line(start, end)
                self.plotter.add_mesh(line, color=(0, 0, 0), line_width=2)

    def _add_arrow_mesh(self, mesh_data: Dict[str, Any]):
        """Add arrow mesh to scene"""
        start_pos = mesh_data["start_position"]
        end_pos = mesh_data["end_position"]
        color = mesh_data["color"]

        # Create arrow using PyVista
        arrow = pv.Arrow(
            start=start_pos,
            direction=np.array(end_pos) - np.array(start_pos),
            tip_length=0.1,
            tip_radius=0.05,
            shaft_radius=0.02,
        )

        self.plotter.add_mesh(arrow, color=color, show_edges=False)

    def _add_text_to_scene(self, label_data: Dict[str, Any]):
        """Add text label to scene"""
        position = label_data["position"]
        text = label_data["text"]
        color = label_data["color"]
        size = label_data["size"]

        # Add text actor
        self.plotter.add_text(
            text, position=position, color=color, font_size=int(size), shadow=True
        )

    def set_camera(self, preset: str = "isometric", **kwargs):
        """Set camera position and orientation"""
        camera_config = get_camera_preset(preset)

        if "position" in kwargs:
            camera_config["position"] = kwargs["position"]
        if "look_at" in kwargs:
            camera_config["look_at"] = kwargs["look_at"]
        if "up_direction" in kwargs:
            camera_config["up_direction"] = kwargs["up_direction"]

        self.plotter.camera_position = [
            camera_config["position"],
            camera_config["look_at"],
            camera_config["up_direction"],
        ]

    def render(
        self, output_path: Optional[str] = None, format: str = "png", **kwargs
    ) -> Optional[Any]:
        """
        Render the scene

        Args:
            output_path: Path to save the rendered image
            format: Output format ('png', 'jpg', 'svg', etc.)
            **kwargs: Additional rendering options

        Returns:
            Rendered image data if no output path specified
        """
        performance_monitor.start("render")

        try:
            # Update camera if needed
            if (
                not hasattr(self.plotter.camera, "position")
                or self.plotter.camera.position is None
            ):
                self.set_camera()

            if output_path:
                # Save to file
                create_directory(os.path.dirname(output_path))
                self.plotter.screenshot(output_path, transparent_background=True)
                return None
            else:
                # Return image data
                return self.plotter.screenshot(transparent_background=True)

        finally:
            performance_monitor.stop("render")

    def show(self, **kwargs):
        """Show the interactive plot"""
        if self.offscreen:
            warnings.warn("Cannot show interactive plot in offscreen mode")
            return

        self.plotter.show(**kwargs)

    def close(self):
        """Close the renderer and free resources"""
        if self.plotter:
            self.plotter.close()
            self.plotter = None
        self.scene_objects.clear()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "render_time": performance_monitor.get_average_time("render"),
            "add_structure_time": performance_monitor.get_average_time("add_structure"),
            "memory_usage_mb": performance_monitor.metrics,
        }


# Convenience functions
def render_structure(
    positions: List[List[float]],
    elements: List[str],
    output_path: str = None,
    quality: str = "medium",
    **kwargs,
) -> Optional[Any]:
    """
    Function for structure rendering

    Args:
        positions: Atomic positions
        elements: Element symbols
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional options

    Returns:
        Rendered image data or None if saved to file
    """
    renderer = CrystalRenderer(quality=quality)
    try:
        renderer.add_structure((positions, elements), **kwargs)
        return renderer.render(output_path=output_path)
    finally:
        renderer.close()


def batch_render_structures(
    structures: List[Dict[str, Any]],
    output_dir: str = "renders",
    quality: str = "medium",
    **kwargs,
) -> List[str]:
    """
    Render multiple structures in batch

    Args:
        structures: List of structure dictionaries with 'positions' and 'elements'
        output_dir: Output directory
        quality: Rendering quality
        **kwargs: Additional options

    Returns:
        List of output file paths
    """
    create_directory(output_dir)
    output_paths = []

    renderer = CrystalRenderer(quality=quality)

    try:
        for i, structure in enumerate(structures):
            renderer.clear_scene()
            renderer.add_structure(
                (structure["positions"], structure["elements"]), **kwargs
            )

            output_path = os.path.join(output_dir, f"structure_{i:04d}.png")
            renderer.render(output_path=output_path)
            output_paths.append(output_path)

    finally:
        renderer.close()

    return output_paths


def create_interactive_scene(
    positions: List[List[float]], elements: List[str], **kwargs
) -> CrystalRenderer:
    """
    Create interactive scene for exploration

    Args:
        positions: Atomic positions
        elements: Element symbols
        **kwargs: Additional options

    Returns:
        CrystalRenderer instance for interaction
    """
    renderer = CrystalRenderer(offscreen=False)
    renderer.add_structure(positions, elements, **kwargs)
    return renderer


def export_animation_frames(
    positions_list: List[List[List[float]]],
    elements: List[str],
    output_dir: str = "animation_frames",
    quality: str = "medium",
    **kwargs,
) -> List[str]:
    """
    Export animation frames from trajectory

    Args:
        positions_list: List of position sets for each frame
        elements: Element symbols (constant across frames)
        output_dir: Output directory
        quality: Rendering quality
        **kwargs: Additional options

    Returns:
        List of frame file paths
    """
    create_directory(output_dir)
    frame_paths = []

    renderer = CrystalRenderer(quality=quality)

    try:
        for i, positions in enumerate(positions_list):
            renderer.clear_scene()
            renderer.add_structure(positions, elements, **kwargs)

            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            renderer.render(output_path=frame_path)
            frame_paths.append(frame_path)

    finally:
        renderer.close()

    return frame_paths


def optimize_rendering_settings(
    quality: str = "medium", max_memory_mb: float = 1024
) -> Dict[str, Any]:
    """
    Get optimized rendering settings based on system capabilities

    Args:
        quality: Base quality setting
        max_memory_mb: Maximum memory to use

    Returns:
        Optimized settings dictionary
    """
    settings = get_quality_settings(quality)
    settings["max_memory_mb"] = max_memory_mb
    settings["batch_size"] = get_optimal_batch_size(max_memory_mb)

    return settings


# Structure conversion functions
def convert_pymatgen_structure(
    structure: Any,
) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """
    Convert pymatgen Structure to positions, elements, and lattice vectors

    Args:
        structure: pymatgen Structure object

    Returns:
        Tuple of (positions, elements, lattice_vectors)
    """
    if not HAS_PYMATGEN or not isinstance(structure, PymatgenStructure):
        raise TypeError("pymatgen Structure object required")

    # Extract positions and elements
    positions = [site.coords.tolist() for site in structure.sites]
    elements = [site.species_string for site in structure.sites]

    # Extract lattice vectors
    lattice_vectors = None
    if hasattr(structure, "lattice") and structure.lattice is not None:
        lattice_vectors = structure.lattice.matrix.tolist()

    return positions, elements, lattice_vectors


def convert_ase_atoms(
    atoms: Any,
) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """
    Convert ASE Atoms to positions, elements, and lattice vectors

    Args:
        atoms: ASE Atoms object

    Returns:
        Tuple of (positions, elements, lattice_vectors)
    """
    if not HAS_ASE or not isinstance(atoms, Atoms):
        raise TypeError("ASE Atoms object required")

    # Extract positions and elements
    positions = atoms.positions.tolist()
    elements = [atom.symbol for atom in atoms]

    # Extract lattice vectors (cell)
    lattice_vectors = None
    if hasattr(atoms, "cell") and atoms.cell is not None:
        lattice_vectors = atoms.cell.tolist()

    return positions, elements, lattice_vectors


def convert_structure_to_arrays(
    structure: Any,
) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """
    Convert various structure objects to positions, elements, and lattice vectors

    Args:
        structure: Structure object (pymatgen Structure, ASE Atoms, or dict)

    Returns:
        Tuple of (positions, elements, lattice_vectors)
    """
    if HAS_PYMATGEN and isinstance(structure, PymatgenStructure):
        return convert_pymatgen_structure(structure)
    elif HAS_ASE and isinstance(structure, Atoms):
        return convert_ase_atoms(structure)
    elif isinstance(structure, dict):
        # Handle dict with 'positions', 'elements', 'lattice_vectors' keys
        positions = structure.get("positions", [])
        elements = structure.get("elements", [])
        lattice_vectors = structure.get("lattice_vectors")
        return positions, elements, lattice_vectors
    else:
        raise TypeError(
            f"Unsupported structure type: {type(structure)}. "
            "Supported types: pymatgen Structure, ASE Atoms, or dict"
        )


# Enhanced rendering functions with structure object support
def render_pymatgen_structure(
    structure: Any, output_path: str = None, quality: str = "medium", **kwargs
) -> Optional[Any]:
    """
    Render pymatgen Structure object

    Args:
        structure: pymatgen Structure object
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional rendering options

    Returns:
        Rendered image data or None if saved to file
    """
    positions, elements, lattice_vectors = convert_pymatgen_structure(structure)

    # Add lattice vectors to kwargs if available
    if lattice_vectors and "lattice_vectors" not in kwargs:
        kwargs["lattice_vectors"] = lattice_vectors

    return render_structure(
        positions=positions,
        elements=elements,
        output_path=output_path,
        quality=quality,
        **kwargs,
    )


def render_ase_structure(
    atoms: Any, output_path: str = None, quality: str = "medium", **kwargs
) -> Optional[Any]:
    """
    Render ASE Atoms object

    Args:
        atoms: ASE Atoms object
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional rendering options

    Returns:
        Rendered image data or None if saved to file
    """
    positions, elements, lattice_vectors = convert_ase_atoms(atoms)

    # Add lattice vectors to kwargs if available
    if lattice_vectors and "lattice_vectors" not in kwargs:
        kwargs["lattice_vectors"] = lattice_vectors

    return render_structure(
        positions=positions,
        elements=elements,
        output_path=output_path,
        quality=quality,
        **kwargs,
    )


def render_structure_object(
    structure: Any, output_path: str = None, quality: str = "medium", **kwargs
) -> Optional[Any]:
    """
    Render any supported structure object (pymatgen Structure, ASE Atoms, or dict)

    Args:
        structure: Structure object
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional rendering options

    Returns:
        Rendered image data or None if saved to file
    """
    positions, elements, lattice_vectors = convert_structure_to_arrays(structure)

    # Add lattice vectors to kwargs if available
    if lattice_vectors and "lattice_vectors" not in kwargs:
        kwargs["lattice_vectors"] = lattice_vectors

    return render_structure(
        positions=positions,
        elements=elements,
        output_path=output_path,
        quality=quality,
        **kwargs,
    )


# Enhanced CrystalRenderer with structure object support
class CrystalRenderer:
    """
    Supports pymatgen Structure and ASE Atoms objects
    """

    def __init__(self, quality: str = "medium", offscreen: bool = True):
        """
        Initialize the renderer

        Args:
            quality: Rendering quality ('low', 'medium', 'high', 'ultra')
            offscreen: Whether to use offscreen rendering
        """
        if not HAS_PYVISTA:
            raise ImportError("PyVista is required for CrystalRenderer")

        self.quality = quality
        self.offscreen = offscreen
        self.plotter = None
        self.scene_objects = []

        self.quality_settings = get_quality_settings(quality)

        self.performance_settings = PERFORMANCE_SETTINGS.copy()

        self._initialize_plotter()

    def _initialize_plotter(self):
        """Initialize PyVista plotter with optimized settings"""
        if self.offscreen:
            self.plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
        else:
            self.plotter = pv.Plotter(window_size=(800, 600))

        self.plotter.set_background(DEFAULT_RENDER_SETTINGS["background_color"])

        if hasattr(self.plotter, "enable_gpu_acceleration"):
            try:
                self.plotter.enable_gpu_acceleration()
            except:
                pass

    def clear_scene(self):
        """Clear all objects from the scene"""
        if self.plotter:
            self.plotter.clear()
        self.scene_objects.clear()

    def add_structure(self, structure: Any, **kwargs) -> None:
        """
        Add crystal structure to the scene
        Supports pymatgen Structure, ASE Atoms, or position/element arrays

        Args:
            structure: Structure object or positions list
            **kwargs: Additional rendering options including:
                - show_atoms (bool): Whether to show atoms (default: True)
                - show_bonds (bool): Whether to show bonds (default: False)
                - show_cell (bool): Whether to show unit cell (default: True)
                - atom_scale (float): Scale factor for atom sizes (default: 1.0)
                - color_scheme (str): Color scheme for atoms ('vesta', 'jmol', 'cpk') (default: 'vesta')
                - sphere_resolution (int): Resolution for atom spheres (default: quality-dependent)
                - bond_cutoff (float): Maximum bond distance (default: 3.0)
                - bond_radius (float): Radius of bond cylinders (default: 0.1)
                - show_faces (bool): Whether to show cell faces (default: True)
                - show_edges (bool): Whether to show cell edges (default: True)
                - face_opacity (float): Opacity of cell faces (default: 0.1)
                - show_labels (bool): Whether to show element labels (default: False)
                - show_image_sites (bool): Whether to show image sites on cell boundaries (default: False)
                - cell_boundary_tol (float): Tolerance for cell boundary detection (default: 0.05)
        """
        performance_monitor.start("add_structure")

        try:
            # Convert structure to arrays
            if isinstance(structure, (list, tuple)) and len(structure) >= 2:
                # Handle positions, elements arrays
                positions = structure[0]
                elements = structure[1]
                lattice_vectors = kwargs.get("lattice_vectors")
            else:
                # Handle structure objects
                positions, elements, lattice_vectors = convert_structure_to_arrays(
                    structure
                )
                if lattice_vectors and "lattice_vectors" not in kwargs:
                    kwargs["lattice_vectors"] = lattice_vectors

            # Convert positions to Vector3D
            pos_vectors = [Vector3D(*pos) for pos in positions]

            # Render atoms
            if kwargs.get("show_atoms", True):
                self._add_atoms(pos_vectors, elements, **kwargs)

            # Render image sites (atoms near cell boundaries)
            if kwargs.get("show_image_sites", False) and lattice_vectors:
                image_positions, image_elements = _add_image_sites_static(
                    positions, elements, lattice_vectors, **kwargs
                )
                if image_positions:
                    # Convert to Vector3D
                    image_pos_vectors = [Vector3D(*pos) for pos in image_positions]

                    # Render image atoms with transparency
                    scale = kwargs.get("atom_scale", 1.0)
                    color_scheme = kwargs.get("color_scheme", "vesta")
                    sphere_resolution = kwargs.get(
                        "sphere_resolution",
                        self.quality_settings.get("sphere_resolution", 12),
                    )

                    # Batch render image atoms with transparency
                    batch_size = get_optimal_batch_size()
                    for i in range(0, len(image_pos_vectors), batch_size):
                        batch_positions = image_pos_vectors[i : i + batch_size]
                        batch_elements = image_elements[i : i + batch_size]

                        # Create image atoms with transparency (50% opacity)
                        image_atoms = atom_renderer.render_atoms_batch(
                            batch_positions,
                            batch_elements,
                            scale,
                            color_scheme=color_scheme,
                            sphere_resolution=sphere_resolution,
                        )

                        # Apply transparency to image atoms
                        for atom in image_atoms:
                            if hasattr(atom, "material"):
                                atom["material"]["opacity"] = 0.5  # 50% transparency
                            self._add_mesh_to_scene(atom)

            # Render bonds
            if kwargs.get("show_bonds", False):
                self._add_bonds(pos_vectors, elements, **kwargs)

            # Render unit cell
            if lattice_vectors and kwargs.get("show_cell", True):
                # Remove lattice_vectors from kwargs to avoid duplicate argument error
                cell_kwargs = kwargs.copy()
                cell_kwargs.pop("lattice_vectors", None)
                # Filter kwargs to only pass arguments relevant to cell rendering
                cell_render_kwargs = {
                    "show_faces": cell_kwargs.get("show_faces", True),
                    "show_edges": cell_kwargs.get("show_edges", True),
                    "face_opacity": cell_kwargs.get("face_opacity", 0.1),
                }
                self._add_unit_cell(lattice_vectors, **cell_render_kwargs)

            # Render vectors (forces, magnetic moments, etc.)
            if "vectors" in kwargs:
                self._add_vectors(kwargs["vectors"], **kwargs)

            # Add labels
            if kwargs.get("show_labels", False):
                self._add_labels(pos_vectors, elements, **kwargs)

        finally:
            performance_monitor.stop("add_structure")

    def _add_atoms(self, positions: List[Vector3D], elements: List[str], **kwargs):
        """Add atoms to the scene"""
        scale = kwargs.get("atom_scale", 1.0)
        color_scheme = kwargs.get("color_scheme", "vesta")
        sphere_resolution = kwargs.get("sphere_resolution", None)

        # Batch render atoms for performance
        batch_size = get_optimal_batch_size()
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i : i + batch_size]
            batch_elements = elements[i : i + batch_size]

            atoms = atom_renderer.render_atoms_batch(
                batch_positions,
                batch_elements,
                scale,
                color_scheme=color_scheme,
                sphere_resolution=sphere_resolution,
            )

            for atom in atoms:
                self._add_mesh_to_scene(atom)

    def _add_bonds(self, positions: List[Vector3D], elements: List[str], **kwargs):
        """Add bonds to the scene"""
        bond_cutoff = kwargs.get("bond_cutoff", 3.0)
        bond_radius = kwargs.get("bond_radius", 0.1)

        # Find bonds
        bonds = self._find_bonds(positions, elements, bond_cutoff)

        # Batch render bonds
        batch_size = get_optimal_batch_size()
        for i in range(0, len(bonds), batch_size):
            batch_bonds = bonds[i : i + batch_size]
            bond_objects = bond_renderer.render_bonds_batch(batch_bonds, bond_radius)

            for bond in bond_objects:
                self._add_mesh_to_scene(bond)

    def _find_bonds(
        self, positions: List[Vector3D], elements: List[str], cutoff: float
    ) -> List[Tuple[Vector3D, Vector3D, str]]:
        """Find bonds between atoms within cutoff distance"""
        bonds = []

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = calculate_distance(positions[i], positions[j])
                if distance <= cutoff:
                    bond_type = self._determine_bond_type(
                        elements[i], elements[j], distance
                    )
                    bonds.append((positions[i], positions[j], bond_type))

        return bonds

    def _determine_bond_type(self, elem1: str, elem2: str, distance: float) -> str:
        """Determine bond type based on elements and distance"""
        # Simple bond type determination
        # In practice, this would use more sophisticated bonding analysis
        radius1 = get_atomic_radius(elem1)
        radius2 = get_atomic_radius(elem2)
        expected_distance = radius1 + radius2

        if distance < expected_distance * 0.8:
            return "triple"
        elif distance < expected_distance * 0.9:
            return "double"
        else:
            return "single"

    def _add_unit_cell(self, lattice_vectors: List[List[float]], **kwargs):
        """Add unit cell to the scene"""
        # Convert to Vector3D
        vectors = [Vector3D(*vec) for vec in lattice_vectors]

        cell_data = cell_renderer.render_unit_cell(vectors, **kwargs)
        self._add_mesh_to_scene(cell_data)

    def _add_vectors(self, vectors: List[Dict[str, Any]], **kwargs):
        """Add vectors (forces, magnetic moments, etc.) to the scene"""
        scale = kwargs.get("vector_scale", 1.0)

        for vector_data in vectors:
            start_pos = Vector3D(*vector_data["position"])
            direction = Vector3D(*vector_data["direction"])
            color = vector_data.get("color", (1.0, 0.0, 0.0))

            vector_obj = vector_renderer.render_vector(
                start_pos, direction, color, scale
            )
            self._add_mesh_to_scene(vector_obj)

    def _add_labels(self, positions: List[Vector3D], elements: List[str], **kwargs):
        """Add labels to the scene"""
        for pos, elem in zip(positions, elements):
            label_renderer.add_label(pos, elem, **kwargs)

        labels = label_renderer.render_labels()
        for label in labels:
            self._add_text_to_scene(label)

    def _add_mesh_to_scene(self, mesh_data: Dict[str, Any]):
        """Add mesh object to PyVista scene"""
        try:
            if mesh_data["type"] == "sphere":
                self._add_sphere_mesh(mesh_data)
            elif mesh_data["type"] == "cylinder":
                self._add_cylinder_mesh(mesh_data)
            elif mesh_data["type"] == "cell":
                self._add_cell_mesh(mesh_data)
            elif mesh_data["type"] == "arrow":
                self._add_arrow_mesh(mesh_data)
            # Add to scene objects list
            self.scene_objects.append(mesh_data)
        except Exception as e:
            warnings.warn(f"Failed to add mesh to scene: {e}")

    def _add_sphere_mesh(self, mesh_data: Dict[str, Any]):
        """Add sphere mesh to scene"""
        position = mesh_data["position"]
        mesh = mesh_data["mesh"]
        material = mesh_data["material"]

        # Create PyVista sphere
        sphere = pv.Sphere(
            radius=mesh["radius"],
            center=position,
            theta_resolution=mesh["resolution"],
            phi_resolution=mesh["resolution"],
        )

        # Apply material
        self.plotter.add_mesh(
            sphere,
            color=material["color"],
            opacity=material["opacity"],
            show_edges=False,
        )

    def _add_cylinder_mesh(self, mesh_data: Dict[str, Any]):
        """Add cylinder mesh to scene"""
        start_pos = mesh_data["start_position"]
        end_pos = mesh_data["end_position"]
        mesh = mesh_data["mesh"]
        material = mesh_data["material"]

        # Create PyVista cylinder
        direction = np.array(end_pos) - np.array(start_pos)
        height = np.linalg.norm(direction)
        center = (np.array(start_pos) + np.array(end_pos)) / 2

        cylinder = pv.Cylinder(
            center=center,
            direction=direction,
            radius=mesh["radius"],
            height=height,
            resolution=mesh["resolution"],
        )

        # Apply material
        self.plotter.add_mesh(
            cylinder,
            color=material["color"],
            opacity=material["opacity"],
            show_edges=False,
        )

    def _add_cell_mesh(self, mesh_data: Dict[str, Any]):
        """Add unit cell mesh to scene"""
        vertices = mesh_data["vertices"]

        if mesh_data.get("show_faces", True):
            # Create faces
            faces = [
                [0, 1, 2, 3],  # Bottom
                [4, 7, 6, 5],  # Top
                [0, 4, 5, 1],  # Front
                [2, 6, 7, 3],  # Back
                [0, 3, 7, 4],  # Left
                [1, 5, 6, 2],  # Right
            ]

            for face in faces:
                face_vertices = [vertices[i] for i in face]
                face_mesh = pv.Polygon(face_vertices)
                self.plotter.add_mesh(
                    face_mesh,
                    color=mesh_data["face_material"]["color"],
                    opacity=mesh_data["face_opacity"],
                    show_edges=False,
                )

        if mesh_data.get("show_edges", True):
            # Create edges
            edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],  # Bottom
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],  # Top
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7],  # Vertical
            ]

            for edge in edges:
                start = vertices[edge[0]]
                end = vertices[edge[1]]
                line = pv.Line(start, end)
                self.plotter.add_mesh(line, color=(0, 0, 0), line_width=2)

    def _add_arrow_mesh(self, mesh_data: Dict[str, Any]):
        """Add arrow mesh to scene"""
        start_pos = mesh_data["start_position"]
        end_pos = mesh_data["end_position"]
        color = mesh_data["color"]

        # Create arrow using PyVista
        arrow = pv.Arrow(
            start=start_pos,
            direction=np.array(end_pos) - np.array(start_pos),
            tip_length=0.1,
            tip_radius=0.05,
            shaft_radius=0.02,
        )

        self.plotter.add_mesh(arrow, color=color, show_edges=False)

    def _add_text_to_scene(self, label_data: Dict[str, Any]):
        """Add text label to scene"""
        position = label_data["position"]
        text = label_data["text"]
        color = label_data["color"]
        size = label_data["size"]

        # Add text actor
        self.plotter.add_text(
            text, position=position, color=color, font_size=int(size), shadow=True
        )

    def set_camera(self, preset: str = "isometric", **kwargs):
        """Set camera position and orientation"""
        camera_config = get_camera_preset(preset)

        if "position" in kwargs:
            camera_config["position"] = kwargs["position"]
        if "look_at" in kwargs:
            camera_config["look_at"] = kwargs["look_at"]
        if "up_direction" in kwargs:
            camera_config["up_direction"] = kwargs["up_direction"]

        self.plotter.camera_position = [
            camera_config["position"],
            camera_config["look_at"],
            camera_config["up_direction"],
        ]

    def render(
        self, output_path: Optional[str] = None, format: str = "png", **kwargs
    ) -> Optional[Any]:
        """
        Render the scene

        Args:
            output_path: Path to save the rendered image
            format: Output format ('png', 'jpg', 'svg', etc.)
            **kwargs: Additional rendering options

        Returns:
            Rendered image data if no output path specified
        """
        performance_monitor.start("render")

        try:
            # Update camera if needed
            if (
                not hasattr(self.plotter.camera, "position")
                or self.plotter.camera.position is None
            ):
                self.set_camera()

            if output_path:
                # Save to file
                create_directory(os.path.dirname(output_path))
                self.plotter.screenshot(output_path, transparent_background=True)
                return None
            else:
                # Return image data
                return self.plotter.screenshot(transparent_background=True)

        finally:
            performance_monitor.stop("render")

    def show(self, **kwargs):
        """Show the interactive plot"""
        if self.offscreen:
            warnings.warn("Cannot show interactive plot in offscreen mode")
            return

        self.plotter.show(**kwargs)

    def close(self):
        """Close the renderer and free resources"""
        if self.plotter:
            self.plotter.close()
            self.plotter = None
        self.scene_objects.clear()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "render_time": performance_monitor.get_average_time("render"),
            "add_structure_time": performance_monitor.get_average_time("add_structure"),
            "memory_usage_mb": performance_monitor.metrics,
        }


# Convenience functions
def render_structure(
    positions: List[List[float]],
    elements: List[str],
    output_path: str = None,
    quality: str = "medium",
    **kwargs,
) -> Optional[Any]:
    """
    Function for structure rendering

    Args:
        positions: Atomic positions
        elements: Element symbols
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional options

    Returns:
        Rendered image data or None if saved to file
    """
    renderer = CrystalRenderer(quality=quality)
    try:
        renderer.add_structure((positions, elements), **kwargs)
        return renderer.render(output_path=output_path)
    finally:
        renderer.close()


def batch_render_structures(
    structures: List[Dict[str, Any]],
    output_dir: str = "renders",
    quality: str = "medium",
    **kwargs,
) -> List[str]:
    """
    Render multiple structures in batch

    Args:
        structures: List of structure dictionaries with 'positions' and 'elements'
        output_dir: Output directory
        quality: Rendering quality
        **kwargs: Additional options

    Returns:
        List of output file paths
    """
    create_directory(output_dir)
    output_paths = []

    renderer = CrystalRenderer(quality=quality)

    try:
        for i, structure in enumerate(structures):
            renderer.clear_scene()
            renderer.add_structure(structure, **kwargs)

            output_path = os.path.join(output_dir, f"structure_{i:04d}.png")
            renderer.render(output_path=output_path)
            output_paths.append(output_path)

    finally:
        renderer.close()

    return output_paths


def create_interactive_scene(
    positions: List[List[float]], elements: List[str], **kwargs
) -> CrystalRenderer:
    """
    Create interactive scene for exploration

    Args:
        positions: Atomic positions
        elements: Element symbols
        **kwargs: Additional options

    Returns:
        CrystalRenderer instance for interaction
    """
    renderer = CrystalRenderer(offscreen=False)
    renderer.add_structure((positions, elements), **kwargs)
    return renderer


def export_animation_frames(
    positions_list: List[List[List[float]]],
    elements: List[str],
    output_dir: str = "animation_frames",
    quality: str = "medium",
    **kwargs,
) -> List[str]:
    """
    Export animation frames from trajectory

    Args:
        positions_list: List of position sets for each frame
        elements: Element symbols (constant across frames)
        output_dir: Output directory
        quality: Rendering quality
        **kwargs: Additional options

    Returns:
        List of frame file paths
    """
    create_directory(output_dir)
    frame_paths = []

    renderer = CrystalRenderer(quality=quality)

    try:
        for i, positions in enumerate(positions_list):
            renderer.clear_scene()
            renderer.add_structure((positions, elements), **kwargs)

            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            renderer.render(output_path=frame_path)
            frame_paths.append(frame_path)

    finally:
        renderer.close()

    return frame_paths


def optimize_rendering_settings(
    quality: str = "medium", max_memory_mb: float = 1024
) -> Dict[str, Any]:
    """
    Get optimized rendering settings based on system capabilities

    Args:
        quality: Base quality setting
        max_memory_mb: Maximum memory to use

    Returns:
        Optimized settings dictionary
    """
    settings = get_quality_settings(quality)
    settings["max_memory_mb"] = max_memory_mb
    settings["batch_size"] = get_optimal_batch_size(max_memory_mb)

    return settings


# Structure conversion functions
def convert_pymatgen_structure(
    structure: Any,
) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """
    Convert pymatgen Structure to positions, elements, and lattice vectors

    Args:
        structure: pymatgen Structure object

    Returns:
        Tuple of (positions, elements, lattice_vectors)
    """
    if not HAS_PYMATGEN or not isinstance(structure, PymatgenStructure):
        raise TypeError("pymatgen Structure object required")

    # Extract positions and elements
    positions = [site.coords.tolist() for site in structure.sites]
    elements = [site.species_string for site in structure.sites]

    # Extract lattice vectors
    lattice_vectors = None
    if hasattr(structure, "lattice") and structure.lattice is not None:
        lattice_vectors = structure.lattice.matrix.tolist()

    return positions, elements, lattice_vectors


def convert_ase_atoms(
    atoms: Any,
) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """
    Convert ASE Atoms to positions, elements, and lattice vectors

    Args:
        atoms: ASE Atoms object

    Returns:
        Tuple of (positions, elements, lattice_vectors)
    """
    if not HAS_ASE or not isinstance(atoms, Atoms):
        raise TypeError("ASE Atoms object required")

    # Extract positions and elements
    positions = atoms.positions.tolist()
    elements = [atom.symbol for atom in atoms]

    # Extract lattice vectors (cell)
    lattice_vectors = None
    if hasattr(atoms, "cell") and atoms.cell is not None:
        lattice_vectors = atoms.cell.tolist()

    return positions, elements, lattice_vectors


def convert_structure_to_arrays(
    structure: Any,
) -> Tuple[List[List[float]], List[str], Optional[List[List[float]]]]:
    """
    Convert various structure objects to positions, elements, and lattice vectors

    Args:
        structure: Structure object (pymatgen Structure, ASE Atoms, or dict)

    Returns:
        Tuple of (positions, elements, lattice_vectors)
    """
    if HAS_PYMATGEN and isinstance(structure, PymatgenStructure):
        return convert_pymatgen_structure(structure)
    elif HAS_ASE and isinstance(structure, Atoms):
        return convert_ase_atoms(structure)
    elif isinstance(structure, dict):
        # Handle dict with 'positions', 'elements', 'lattice_vectors' keys
        positions = structure.get("positions", [])
        elements = structure.get("elements", [])
        lattice_vectors = structure.get("lattice_vectors")
        return positions, elements, lattice_vectors
    else:
        raise TypeError(
            f"Unsupported structure type: {type(structure)}. "
            "Supported types: pymatgen Structure, ASE Atoms, or dict"
        )


# Enhanced rendering functions with structure object support
def render_pymatgen_structure(
    structure: Any, output_path: str = None, quality: str = "medium", **kwargs
) -> Optional[Any]:
    """
    Render pymatgen Structure object

    Args:
        structure: pymatgen Structure object
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional rendering options

    Returns:
        Rendered image data or None if saved to file
    """
    positions, elements, lattice_vectors = convert_pymatgen_structure(structure)

    # Add lattice vectors to kwargs if available
    if lattice_vectors and "lattice_vectors" not in kwargs:
        kwargs["lattice_vectors"] = lattice_vectors

    return render_structure(
        positions=positions,
        elements=elements,
        output_path=output_path,
        quality=quality,
        **kwargs,
    )


def render_ase_structure(
    atoms: Any, output_path: str = None, quality: str = "medium", **kwargs
) -> Optional[Any]:
    """
    Render ASE Atoms object

    Args:
        atoms: ASE Atoms object
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional rendering options

    Returns:
        Rendered image data or None if saved to file
    """
    positions, elements, lattice_vectors = convert_ase_atoms(atoms)

    # Add lattice vectors to kwargs if available
    if lattice_vectors and "lattice_vectors" not in kwargs:
        kwargs["lattice_vectors"] = lattice_vectors

    return render_structure(
        positions=positions,
        elements=elements,
        output_path=output_path,
        quality=quality,
        **kwargs,
    )


def render_structure_object(
    structure: Any, output_path: str = None, quality: str = "medium", **kwargs
) -> Optional[Any]:
    """
    Render any supported structure object (pymatgen Structure, ASE Atoms, or dict)

    Args:
        structure: Structure object
        output_path: Output file path
        quality: Rendering quality
        **kwargs: Additional rendering options

    Returns:
        Rendered image data or None if saved to file
    """
    positions, elements, lattice_vectors = convert_structure_to_arrays(structure)

    # Add lattice vectors to kwargs if available
    if lattice_vectors and "lattice_vectors" not in kwargs:
        kwargs["lattice_vectors"] = lattice_vectors

    return render_structure(
        positions, elements, output_path=output_path, quality=quality, **kwargs
    )


def _add_image_sites_static(
    positions: List[List[float]],
    elements: List[str],
    lattice_vectors: List[List[float]],
    **kwargs,
):
    """
    Add image sites (atoms near cell boundaries) to the scene (static version)

    Args:
        positions: List of atomic positions (cartesian coordinates)
        elements: List of element symbols
        lattice_vectors: Lattice vectors defining the unit cell
        **kwargs: Additional rendering options
    """
    if not HAS_NUMPY:
        return

    cell_boundary_tol = kwargs.get("cell_boundary_tol", 0.05)
    scale = kwargs.get("atom_scale", 1.0)
    color_scheme = kwargs.get("color_scheme", "vesta")
    sphere_resolution = kwargs.get("sphere_resolution", 12)

    # Collect all image sites
    image_positions = []
    image_elements = []

    for pos, elem in zip(positions, elements):
        # Calculate image sites for this atom
        image_coords = get_image_sites_from_shifts(
            pos, lattice_vectors, cell_boundary_tol=cell_boundary_tol
        )

        if len(image_coords) > 0:
            for image_pos in image_coords:
                image_positions.append(image_pos.tolist())
                image_elements.append(elem)

    return image_positions, image_elements


# Export functions
__all__ = [
    "CrystalRenderer",
    "render_structure",
    "batch_render_structures",
    "create_interactive_scene",
    "export_animation_frames",
    "optimize_rendering_settings",
    "convert_pymatgen_structure",
    "convert_ase_atoms",
    "convert_structure_to_arrays",
    "render_pymatgen_structure",
    "render_ase_structure",
    "render_structure_object",
]
