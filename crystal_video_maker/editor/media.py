"""
Media processing functions for creating videos and GIFs with optimization
"""

import os
import io
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import warnings


try:
    import moviepy.editor as mpy
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    warnings.warn("moviepy not available - video creation disabled")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    warnings.warn("imageio not available - some media functions disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available - GIF creation disabled")

@lru_cache(maxsize=100)
def _get_media_config(fmt: str, quality: str = "standard") -> Dict[str, Any]:
    """
    Get cached media configuration for different formats
    
    Args:
        fmt: Media format (mp4, avi, gif, etc.)
        quality: Quality setting
        
    Returns:
        Configuration dictionary
    """
    configs = {
        "mp4": {
            "codec": "libx264",
            "audio": False,
            "preset": "medium" if quality == "standard" else quality
        },
        "avi": {
            "codec": "png",
            "audio": False
        },
        "gif": {
            "optimize": True,
            "loop": 0,
            "quality": 85 if quality == "standard" else int(quality) if quality.isdigit() else 85
        },
        "webm": {
            "codec": "libvpx-vp9",
            "audio": False
        }
    }
    return configs.get(fmt.lower(), configs["mp4"])

def parallel_image_loading(
    image_bytes_list: List[bytes],
    target_format: str = "array"
) -> List[Any]:
    """
    Load images from bytes in parallel for better performance
    
    Args:
        image_bytes_list: List of image bytes
        target_format: Target format ("array" for numpy, "pil" for PIL)
        
    Returns:
        List of loaded images
    """
    if not image_bytes_list:
        return []
    
    def load_single_image(byte_data: bytes):
        if target_format == "array" and IMAGEIO_AVAILABLE:
            return imageio.imread(io.BytesIO(byte_data))
        elif target_format == "pil" and PIL_AVAILABLE:
            return Image.open(io.BytesIO(byte_data))
        else:
            return byte_data
    
    # Use parallel processing for multiple images
    if len(image_bytes_list) > 1:
        with ThreadPoolExecutor(max_workers=min(16, len(image_bytes_list))) as executor:
            return list(executor.map(load_single_image, image_bytes_list))
    else:
        return [load_single_image(image_bytes_list[0])]

def video_maker(
    image_bytes_list: List[bytes],
    name: str = "video",
    fmt: str = "mp4",
    fps: int = 1,
    quality: str = "standard",
    output_dir: str = "."
) -> str:
    """
    Create video from list of image bytes
    
    Args:
        image_bytes_list: List of image bytes
        name: Output filename (without extension)
        fmt: Video format
        fps: Frames per second
        quality: Quality setting
        output_dir: Output directory
        
    Returns:
        Path to created video file
    """
    if not MOVIEPY_AVAILABLE:
        raise ImportError("moviepy is required for video creation")
    
    if not image_bytes_list:
        raise ValueError("No images provided")
    
    # Load images in parallel
    img_arrays = parallel_image_loading(image_bytes_list, "array")
    
    # Get media configuration
    config = _get_media_config(fmt, quality)
    
    # Create video clip
    clip = mpy.ImageSequenceClip(img_arrays, fps=fps)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output path
    output_path = os.path.join(output_dir, f"{name}.{fmt}")
    
    try:
        # Write video file
        clip.write_videofile(output_path, fps=fps, **config, verbose=False, logger=None)
    finally:
        # Clean up
        clip.close()
        del img_arrays
    
    return output_path

def gif_maker(
    image_bytes_list: List[bytes],
    name: str = "animation.gif",
    duration: int = 100,
    loop: int = 0,
    optimize: bool = True,
    output_dir: str = "."
) -> str:
    """
    Create GIF from list of image bytes
    
    Args:
        image_bytes_list: List of image bytes
        name: Output filename
        duration: Duration per frame in milliseconds
        loop: Number of loops (0 = infinite)
        optimize: Whether to optimize GIF
        output_dir: Output directory
        
    Returns:
        Path to created GIF file
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for GIF creation")
    
    if not image_bytes_list:
        raise ValueError("No images provided")
    
    # Load images in parallel
    img_objects = parallel_image_loading(image_bytes_list, "pil")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output path
    if not name.endswith('.gif'):
        name += '.gif'
    output_path = os.path.join(output_dir, name)
    
    try:
        # Save as GIF
        img_objects[0].save(
            output_path,
            save_all=True,
            append_images=img_objects[1:],
            optimize=optimize,
            duration=duration,
            loop=loop
        )
    finally:
        # Clean up PIL images
        for img in img_objects:
            if hasattr(img, 'close'):
                img.close()
    
    return output_path

def create_optimized_media_batch(
    image_bytes_list: List[bytes],
    outputs: List[Dict[str, Any]],
    base_name: str = "media",
    output_dir: str = "."
) -> List[str]:
    """
    Create multiple media formats from same image sequence efficiently
    
    Args:
        image_bytes_list: List of image bytes
        outputs: List of output configurations
        base_name: Base filename
        output_dir: Output directory
        
    Returns:
        List of created file paths
    """
    if not image_bytes_list:
        raise ValueError("No images provided")
    
    results = []
    
    for i, output_config in enumerate(outputs):
        output_type = output_config.get('type', 'video')
        params = output_config.get('params', {})
        
        # Set default name if not provided
        if 'name' not in params:
            params['name'] = f"{base_name}_{i+1}"
        
        # Set output directory
        params['output_dir'] = output_dir
        
        try:
            if output_type == "video":
                result_path = video_maker(image_bytes_list, **params)
            elif output_type == "gif":
                result_path = gif_maker(image_bytes_list, **params)
            else:
                warnings.warn(f"Unknown output type: {output_type}")
                continue
            
            results.append(result_path)
            
        except Exception as e:
            warnings.warn(f"Failed to create {output_type}: {e}")
            continue
    
    return results

def create_media_with_progress(
    image_bytes_list: List[bytes],
    output_type: str = "video",
    progress_callback: Optional[callable] = None,
    **kwargs
) -> str:
    """
    Create media with progress callback
    
    Args:
        image_bytes_list: List of image bytes
        output_type: Type of media to create
        progress_callback: Function to call with progress updates
        **kwargs: Additional arguments for media creation
        
    Returns:
        Path to created media file
    """
    def update_progress(stage: str, percent: float):
        if progress_callback:
            progress_callback(stage, percent)
    
    update_progress("Loading images", 0)
    
    if output_type.lower() == "video":
        result = video_maker(image_bytes_list, **kwargs)
    elif output_type.lower() == "gif":
        result = gif_maker(image_bytes_list, **kwargs)
    else:
        raise ValueError(f"Unsupported output type: {output_type}")
    
    update_progress("Complete", 100)
    return result

def optimize_gif_size(
    gif_path: str,
    max_colors: int = 256,
    resize_factor: float = 1.0
) -> str:
    """
    Optimize GIF file size
    
    Args:
        gif_path: Path to input GIF
        max_colors: Maximum number of colors
        resize_factor: Resize factor (1.0 = no resize)
        
    Returns:
        Path to optimized GIF
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for GIF optimization")
    
    # Create output path
    base_name = os.path.splitext(gif_path)[0]
    optimized_path = f"{base_name}_optimized.gif"
    
    # Open and process GIF
    with Image.open(gif_path) as img:
        frames = []
        
        try:
            while True:
                frame = img.copy()
                
                # Resize if needed
                if resize_factor != 1.0:
                    new_size = (
                        int(frame.width * resize_factor),
                        int(frame.height * resize_factor)
                    )
                    frame = frame.resize(new_size, Image.Resampling.LANCZOS)
                
                # Reduce colors
                if max_colors < 256:
                    frame = frame.quantize(colors=max_colors)
                
                frames.append(frame)
                img.seek(img.tell() + 1)
                
        except EOFError:
            pass  # End of frames
    
    # Save optimized GIF
    if frames:
        frames[0].save(
            optimized_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=img.info.get('duration', 100),
            loop=img.info.get('loop', 0)
        )
    
    return optimized_path

def extract_frames_from_video(
    video_path: str,
    output_dir: str = "frames",
    frame_format: str = "png"
) -> List[str]:
    """
    Extract frames from video file
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        frame_format: Format for extracted frames
        
    Returns:
        List of extracted frame paths
    """
    if not MOVIEPY_AVAILABLE:
        raise ImportError("moviepy is required for video frame extraction")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with mpy.VideoFileClip(video_path) as clip:
        frame_paths = []
        
        for i, frame in enumerate(clip.iter_frames()):
            frame_path = os.path.join(output_dir, f"frame_{i:06d}.{frame_format}")
            
            if IMAGEIO_AVAILABLE:
                imageio.imwrite(frame_path, frame)
            else:
                # Fallback using PIL
                if PIL_AVAILABLE:
                    Image.fromarray(frame).save(frame_path)
                else:
                    raise ImportError("Either imageio or PIL is required")
            
            frame_paths.append(frame_path)
    
    return frame_paths

# Legacy aliases for backward compatibility
create_video = video_maker
create_gif = gif_maker
load_images_parallel = parallel_image_loading
