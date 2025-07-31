"""
Crystal Video Maker
"""

from .constants import ELEMENT_RADIUS
from .common_types import Xyz, AnyStructure
from .common_enum import SiteCoords, LabelEnum
from .visualization import structure_3d

from .utils import (
    get_elem_colors,
    get_default_colors,
    normalize_color,
    get_first_matching_site_prop,
    get_struct_prop,
    make_dir,
    check_name_available,
    get_site_symbol
)

from .editor import (
    prepare_image_name_with_checker,
    prepare_image_byte,
    save_as_image_file,
    images_to_bytes,
    batch_save_images,
    video_maker,
    gif_maker,
    parallel_image_loading
)

__version__ = "0.0.1"
__all__ = [
    "ELEMENT_RADIUS",
    "Xyz", 
    "AnyStructure",
    "SiteCoords",
    "LabelEnum",
    "structure_3d",
    "prepare_image_name_with_checker",
    "prepare_image_byte", 
    "save_as_image_file",
    "images_to_bytes",
    "batch_save_images",
    "video_maker",
    "gif_maker",
    "parallel_image_loading",
    "get_elem_colors",
    "get_default_colors",
    "normalize_color",
    "get_first_matching_site_prop",
    "get_struct_prop",
    "make_dir",
    "check_name_available",
    "get_site_symbol"
]
