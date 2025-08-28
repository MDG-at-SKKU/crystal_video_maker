from .core import CrystalRenderer
from .geometry import *
from .rendering import *
from .styling import *
from .utils import *
from .config import *
from .camera import *
from .structure import *
from .scene import *
from .boundary import *
from .performance import *
from .caching import *

__version__ = "1.0.0"
__all__ = [
    "CrystalRenderer",
    "BaseRenderer",
    "render_structure",
    "batch_render_structures",
    "create_interactive_scene",
    "export_animation_frames",
    "optimize_rendering_settings",
    # Camera utilities
    "create_camera_position",
    "calculate_optimal_camera_distance",
    "create_rotated_camera_position",
    "interpolate_camera_positions",
    "create_camera_path",
    "get_camera_preset",
    "calculate_camera_bounds",
    "create_orbital_camera_path",
    # Structure utilities
    "convert_structure_to_arrays",
    "validate_structure_data",
    "get_structure_info",
    # Scene management
    "SceneManager",
    # Boundary utilities
    "get_image_shifts",
    "get_image_sites_from_shifts",
    "add_image_sites_static",
    "is_near_boundary",
    "apply_periodic_boundary_conditions",
    "get_minimum_image_distance",
    # Performance utilities
    "performance_monitor",
    "PerformanceMonitor",
    "timing_decorator",
    "get_optimal_batch_size",
    # Caching utilities
    "cache_mesh_data",
    "get_cached_mesh_data",
    "clear_mesh_cache",
    "normalize_color",
    "interpolate_colors",
    "get_contrasting_color",
    "vectorized_distance",
    "batch_process_items",
    "export_to_file",
    "import_from_file",
    "clear_all_caches",
    "get_cache_info",
]
