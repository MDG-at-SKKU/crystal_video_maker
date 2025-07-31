"""
Image and media processing functions
"""

from .image import (
    prepare_image_name_with_checker,
    prepare_image_byte,
    save_as_image_file,
    images_to_bytes,
    batch_save_images,
    reserve_available_names,
    batch_convert_to_bytes,
    save_images_with_metadata,
    create_image_gallery_html
)

from .media import (
    video_maker,
    gif_maker,
    parallel_image_loading,
    create_optimized_media_batch,
    create_media_with_progress,
    optimize_gif_size,
    extract_frames_from_video
)

__all__ = [
    "prepare_image_name_with_checker",
    "prepare_image_byte",
    "save_as_image_file",
    "images_to_bytes",
    "batch_save_images",
    "reserve_available_names",
    "batch_convert_to_bytes",
    "save_images_with_metadata",
    "create_image_gallery_html",
    "video_maker",
    "gif_maker", 
    "parallel_image_loading",
    "create_optimized_media_batch",
    "create_media_with_progress",
    "optimize_gif_size",
    "extract_frames_from_video"
]
