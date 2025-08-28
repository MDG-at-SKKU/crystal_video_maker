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
    DEFAULT_RENDER_SETTINGS,
    PERFORMANCE_SETTINGS,
)
from .camera import get_camera_preset
from .geometry import Vector3D, calculate_distance, calculate_center_of_mass
from .rendering import (
    atom_renderer,
    bond_renderer,
    cell_renderer,
    vector_renderer,
    label_renderer,
)
from .styling import Material, ColorScheme, get_element_material
from .structure import convert_structure_to_arrays, validate_structure_data
from .scene import SceneManager
from .boundary import add_image_sites_static, get_image_sites_from_shifts
from .performance import performance_monitor, get_optimal_batch_size
from .caching import normalize_color, interpolate_colors, get_contrasting_color
from .utils import create_directory



class CrystalRenderer:
    """
    Supports pymatgen Structure and ASE Atoms objects
    """

    def __init__(self, quality: str = "medium", offscreen: bool = True, notebook: bool = False):
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
        self.scene_manager = SceneManager(quality=quality, offscreen=offscreen, notebook=notebook)
        self.quality_settings = get_quality_settings(quality)
        self.performance_settings = PERFORMANCE_SETTINGS.copy()

    def clear_scene(self):
        """Clear all objects from the scene"""
        self.scene_manager.clear_scene()

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
                image_positions, image_elements = add_image_sites_static(
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
        """Add mesh object to scene"""
        self.scene_manager.add_mesh_to_scene(mesh_data)

    def set_camera(self, preset: str = "isometric", **kwargs):
        """Set camera position and orientation"""
        camera_config = get_camera_preset(preset)

        if "position" in kwargs:
            camera_config["position"] = kwargs["position"]
        if "look_at" in kwargs:
            camera_config["look_at"] = kwargs["look_at"]
        if "up_direction" in kwargs:
            camera_config["up_direction"] = kwargs["up_direction"]

        self.scene_manager.plotter.camera_position = [
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
                not hasattr(self.scene_manager.plotter.camera, "position")
                or self.scene_manager.plotter.camera.position is None
            ):
                self.set_camera()

            return self.scene_manager.render_scene(output_path)

        finally:
            performance_monitor.stop("render")

    def show(self, **kwargs):
        """Show the interactive plot"""
        if self.offscreen:
            warnings.warn("Cannot show interactive plot in offscreen mode")
            return

        self.scene_manager.plotter.show(**kwargs)

    def close(self):
        """Close the renderer and free resources"""
        self.scene_manager.close()

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



# Export functions
__all__ = [
    "CrystalRenderer",
    "render_structure",
    "batch_render_structures",
    "create_interactive_scene",
    "export_animation_frames",
    "optimize_rendering_settings",
]
