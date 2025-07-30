"""
Media creation functions for videos and GIFs with performance improvements
"""

import os
import io
from typing import List, Optional, Union
from pathlib import Path

def video_maker(
    image_bytes_list: List[bytes],
    name: str = "video",
    fmt: str = "mp4", 
    fps: int = 1,
    output_dir: str = ".",
    quality: str = "medium"
) -> str:
    """
    Create video from image bytes with performance improvements

    Args:
        image_bytes_list: List of image bytes to convert to video
        name: Base name for output video file
        fmt: Video format ("mp4", "avi", "mov")
        fps: Frames per second
        output_dir: Output directory
        quality: Video quality ("low", "medium", "high", "ultra")

    Returns:
        Path to created video file
    """
    try:
        import moviepy.editor as mpy
        import imageio

        if not image_bytes_list or all(not img_bytes for img_bytes in image_bytes_list):
            print("No valid image bytes provided")
            return ""

        # Filter out empty bytes
        valid_bytes = [img_bytes for img_bytes in image_bytes_list if img_bytes]

        if not valid_bytes:
            print("No valid image data found")
            return ""

        # Convert bytes to image arrays efficiently
        def bytes_to_array(img_bytes):
            try:
                return imageio.imread(io.BytesIO(img_bytes))
            except Exception as e:
                print(f"Error reading image bytes: {e}")
                return None

        # Use parallel processing for image loading
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(valid_bytes))) as executor:
            image_arrays = list(executor.map(bytes_to_array, valid_bytes))

        # Filter out failed conversions
        image_arrays = [img for img in image_arrays if img is not None]

        if not image_arrays:
            print("Failed to convert image bytes to arrays")
            return ""

        # Create video clip
        clip = mpy.ImageSequenceClip(image_arrays, fps=fps)

        # Set quality parameters
        quality_settings = {
            "low": {"bitrate": "500k", "preset": "fast"},
            "medium": {"bitrate": "1500k", "preset": "medium"}, 
            "high": {"bitrate": "3000k", "preset": "slow"},
            "ultra": {"bitrate": "5000k", "preset": "veryslow"}
        }

        settings = quality_settings.get(quality, quality_settings["medium"])

        # Create output path
        output_path = os.path.join(output_dir, f"{name}.{fmt}")

        # Write video with optimized settings
        if fmt.lower() == "mp4":
            clip.write_videofile(
                output_path,
                fps=fps,
                codec='libx264',
                bitrate=settings["bitrate"],
                preset=settings["preset"],
                verbose=False,
                logger=None
            )
        else:
            clip.write_videofile(output_path, fps=fps, verbose=False, logger=None)

        # Clean up
        clip.close()

        return output_path

    except ImportError:
        print("moviepy not available. Please install: pip install moviepy")
        return ""
    except Exception as e:
        print(f"Error creating video: {e}")
        return ""

def gif_maker(
    image_bytes_list: List[bytes],
    name: str = "animation.gif",
    duration: int = 100,
    loop: int = 0,
    output_dir: str = ".",
    quality: int = 85,
    optimize: bool = True
) -> str:
    """
    Create GIF from image bytes with performance improvements

    Args:
        image_bytes_list: List of image bytes
        name: Output GIF filename
        duration: Duration between frames in milliseconds
        loop: Number of loops (0 for infinite)
        output_dir: Output directory
        quality: GIF quality (1-100)
        optimize: Whether to optimize GIF size

    Returns:
        Path to created GIF file
    """
    try:
        from PIL import Image

        if not image_bytes_list or all(not img_bytes for img_bytes in image_bytes_list):
            print("No valid image bytes provided")
            return ""

        # Filter out empty bytes
        valid_bytes = [img_bytes for img_bytes in image_bytes_list if img_bytes]

        if not valid_bytes:
            print("No valid image data found")
            return ""

        # Convert bytes to PIL Images efficiently
        def bytes_to_pil(img_bytes):
            try:
                return Image.open(io.BytesIO(img_bytes))
            except Exception as e:
                print(f"Error opening image: {e}")
                return None

        # Use parallel processing for image loading
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(valid_bytes))) as executor:
            pil_images = list(executor.map(bytes_to_pil, valid_bytes))

        # Filter out failed conversions
        pil_images = [img for img in pil_images if img is not None]

        if not pil_images:
            print("Failed to convert image bytes to PIL Images")
            return ""

        # Ensure consistent format and size
        if len(pil_images) > 1:
            # Convert all to same mode and size as first image
            target_size = pil_images[0].size
            target_mode = pil_images[0].mode

            normalized_images = []
            for img in pil_images:
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                if img.mode != target_mode:
                    img = img.convert(target_mode)
                normalized_images.append(img)

            pil_images = normalized_images

        # Create output path
        if not name.endswith('.gif'):
            name += '.gif'
        output_path = os.path.join(output_dir, name)

        # Save GIF with optimized settings
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:] if len(pil_images) > 1 else [],
            duration=duration,
            loop=loop,
            optimize=optimize,
            quality=quality
        )

        return output_path

    except ImportError:
        print("PIL not available. Please install: pip install Pillow")
        return ""
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return ""

def create_video_high_quality(
    image_bytes_list: List[bytes],
    name: str = "hq_video",
    fps: int = 24,
    output_dir: str = ".",
    resolution: tuple = (1920, 1080)
) -> str:
    """
    Create high-quality video with custom resolution

    Args:
        image_bytes_list: List of image bytes
        name: Output video name
        fps: Frames per second
        output_dir: Output directory
        resolution: Output resolution (width, height)

    Returns:
        Path to created video
    """
    return video_maker(
        image_bytes_list,
        name=name,
        fmt="mp4",
        fps=fps,
        output_dir=output_dir,
        quality="ultra"
    )

def create_gif_optimized(
    image_bytes_list: List[bytes],
    name: str = "optimized.gif",
    output_dir: str = ".",
    quality: int = 95
) -> str:
    """
    Create optimized GIF with best settings

    Args:
        image_bytes_list: List of image bytes
        name: Output GIF name
        output_dir: Output directory  
        quality: GIF quality (1-100)

    Returns:
        Path to created GIF
    """
    return gif_maker(
        image_bytes_list,
        name=name,
        duration=200,  # Slower animation
        output_dir=output_dir,
        quality=quality,
        optimize=True
    )

def batch_create_media(
    image_sequences: dict,
    output_dir: str = "media_output",
    create_video: bool = True,
    create_gif: bool = True,
    fps: int = 2
) -> dict:
    """
    Create multiple videos and GIFs from image sequences

    Args:
        image_sequences: Dict mapping names to lists of image bytes
        output_dir: Output directory
        create_video: Whether to create videos
        create_gif: Whether to create GIFs
        fps: Frames per second for videos

    Returns:
        Dictionary with created file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for sequence_name, image_bytes in image_sequences.items():
        if not image_bytes:
            continue

        sequence_results = {}

        if create_video:
            video_path = video_maker(
                image_bytes,
                name=f"{sequence_name}_video",
                fps=fps,
                output_dir=output_dir
            )
            if video_path:
                sequence_results['video'] = video_path

        if create_gif:
            gif_path = gif_maker(
                image_bytes,
                name=f"{sequence_name}.gif",
                output_dir=output_dir
            )
            if gif_path:
                sequence_results['gif'] = gif_path

        if sequence_results:
            results[sequence_name] = sequence_results

    return results

def convert_video_format(
    input_path: str,
    output_path: str,
    target_format: str = "mp4"
) -> str:
    """
    Convert video to different format

    Args:
        input_path: Input video file path
        output_path: Output video file path
        target_format: Target format

    Returns:
        Path to converted video
    """
    try:
        import moviepy.editor as mpy

        clip = mpy.VideoFileClip(input_path)
        clip.write_videofile(output_path, verbose=False, logger=None)
        clip.close()

        return output_path

    except ImportError:
        print("moviepy not available for video conversion")
        return ""
    except Exception as e:
        print(f"Error converting video: {e}")
        return ""
