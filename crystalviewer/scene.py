"""
Scene management for PyVista rendering
Handles plotter initialization, mesh addition, and scene operations
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

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

from .config import DEFAULT_RENDER_SETTINGS


class SceneManager:
    """
    Manages PyVista scene and plotter operations
    """

    def __init__(self, quality: str = "medium", offscreen: bool = True, notebook: bool = False):
        """
        Initialize scene manager

        Args:
            quality: Rendering quality
            offscreen: Whether to use offscreen rendering
        """
        if not HAS_PYVISTA:
            raise ImportError("PyVista is required for SceneManager")

        self.quality = quality
        self.offscreen = offscreen
        self.plotter = None
        self.scene_objects = []
        self.notebook = notebook
        self._initialize_plotter()
        

    def _initialize_plotter(self):
        """Initialize PyVista plotter with optimized settings"""
        if self.offscreen:
            self.plotter = pv.Plotter(notebook=self.notebook, off_screen=True, window_size=(800, 600))
        else:
            self.plotter = pv.Plotter(notebook=self.notebook,window_size=(800, 600))

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

    def add_mesh_to_scene(self, mesh_data: Dict[str, Any]):
        """
        Add mesh object to scene based on its type

        Args:
            mesh_data: Dictionary containing mesh information
        """
        try:
            mesh_type = mesh_data.get("type")

            if mesh_type == "sphere":
                self._add_sphere_mesh(mesh_data)
            elif mesh_type == "cylinder":
                self._add_cylinder_mesh(mesh_data)
            elif mesh_type == "cell":
                self._add_cell_mesh(mesh_data)
            elif mesh_type == "arrow":
                self._add_arrow_mesh(mesh_data)
            elif mesh_type == "text":
                self._add_text_mesh(mesh_data)
            else:
                warnings.warn(f"Unknown mesh type: {mesh_type}")

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
                [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
                [4, 5], [5, 6], [6, 7], [7, 4],  # Top
                [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical
            ]

            for edge in edges:
                edge_vertices = [vertices[i] for i in edge]
                edge_mesh = pv.Line(edge_vertices[0], edge_vertices[1])
                self.plotter.add_mesh(
                    edge_mesh,
                    color=mesh_data["edge_material"]["color"],
                    line_width=mesh_data.get("line_width", 2),
                    opacity=1.0,
                )

    def _add_arrow_mesh(self, mesh_data: Dict[str, Any]):
        """Add arrow/vector mesh to scene"""
        start_pos = mesh_data["start_position"]
        direction = mesh_data["direction"]
        scale = mesh_data.get("scale", 1.0)
        material = mesh_data["material"]

        # Create arrow using PyVista
        arrow = pv.Arrow(
            start=start_pos,
            direction=direction,
            scale=scale,
        )

        self.plotter.add_mesh(
            arrow,
            color=material["color"],
            opacity=material["opacity"],
            show_edges=False,
        )

    def _add_text_mesh(self, mesh_data: Dict[str, Any]):
        """Add text mesh to scene"""
        position = mesh_data["position"]
        text = mesh_data["text"]
        size = mesh_data.get("size", 12)
        color = mesh_data.get("color", (0, 0, 0))

        # Add text to plotter
        self.plotter.add_text(
            text,
            position=position,
            size=size,
            color=color,
            font="arial",
        )

    def render_scene(self, output_path: Optional[str] = None) -> Optional[Any]:
        """
        Render the scene

        Args:
            output_path: Path to save the rendered image

        Returns:
            Rendered image data if no output path, None otherwise
        """
        if output_path:
            self.plotter.screenshot(output_path)
            return None
        else:
            return self.plotter.screenshot()

    def close(self):
        """Close the plotter and clean up resources"""
        if self.plotter:
            self.plotter.close()
            self.plotter = None
        self.scene_objects.clear()

    def get_scene_info(self) -> Dict[str, Any]:
        """Get information about the current scene"""
        return {
            "num_objects": len(self.scene_objects),
            "quality": self.quality,
            "offscreen": self.offscreen,
            "has_plotter": self.plotter is not None
        }
