"""
Image processing functions with massive performance improvements
"""

import os
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Optional
import plotly.graph_objects as go

def prepare_image_name_with_checker(
    name: str = "fig", 
    number: int = 0, 
    numbering_rule: str = "000", 
    fmt: str = "png",
    result_dir: str = "result"
) -> str:
    """
    Generate available image filename with collision checking

    Args:
        name: Base name for the image file
        number: Starting number for filename
        numbering_rule: Number formatting (e.g., "000" for zero-padding)
        fmt: File format extension
        result_dir: Directory to save in

    Returns:
        Available filename string
    """
    from ..utils import check_name_available

    width = numbering_rule.count("0")
    i = number

    while True:
        filename = f"{name}-{i:0{width}d}.{fmt}"
        if check_name_available(filename, result_dir):
            return filename
        i += 1

def prepare_image_byte(
    figs: Union[go.Figure, List[go.Figure]],
    fmt: str = "png",
    max_workers: Optional[int] = None
) -> List[bytes]:
    """
    Convert Plotly figures to image bytes with parallel processing

    Args:
        figs: Single figure or list of figures
        fmt: Image format ("png", "jpg", "svg", "pdf")
        max_workers: Maximum number of worker threads

    Returns:
        List of image bytes
    """
    if not isinstance(figs, list):
        figs = [figs]

    if not figs:
        return []

    # Use parallel processing for better performance
    max_workers = max_workers or min(8, len(figs), (os.cpu_count() or 1) + 4)

    def convert_single_fig(fig):
        try:
            return fig.to_image(format=fmt)
        except Exception as e:
            print(f"Error converting figure to {fmt}: {e}")
            return b''

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        image_bytes = list(executor.map(convert_single_fig, figs))

    return [img_bytes for img_bytes in image_bytes if img_bytes]

def save_as_image_file(
    figs: Union[go.Figure, List[go.Figure]],
    name: str = "fig",
    number: int = 0,
    numbering_rule: str = "000", 
    fmt: str = "png",
    result_dir: str = "result",
    with_sequence: bool = False
) -> List[str]:
    """
    Save figures as image files with performance improvements

    Args:
        figs: Single figure or list of figures to save
        name: Base name for image files
        number: Starting number for filenames
        numbering_rule: Number formatting rule
        fmt: Image format
        result_dir: Directory to save files in
        with_sequence: Whether to use sequential numbering

    Returns:
        List of saved filenames
    """
    from ..utils import make_dir, reserve_available_names

    # Ensure result directory exists
    make_dir(result_dir)

    if not isinstance(figs, list):
        figs = [figs]

    if not figs:
        return []

    # Generate filenames efficiently
    if with_sequence:
        width = numbering_rule.count("0")
        filenames = [f"{name}-{i:0{width}d}.{fmt}" for i in range(len(figs))]
    else:
        filenames = reserve_available_names(name, len(figs), result_dir, fmt, numbering_rule)

    # Save files with parallel processing
    def save_single_file(args):
        fig, filename = args
        filepath = os.path.join(result_dir, filename)
        try:
            fig.write_image(filepath)
            return filename
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=min(8, len(figs))) as executor:
        saved_files = list(executor.map(save_single_file, zip(figs, filenames)))

    return [f for f in saved_files if f is not None]

def batch_convert_images(
    figs: List[go.Figure],
    output_dir: str = "images",
    base_name: str = "fig",
    fmt: str = "png",
    max_workers: Optional[int] = None
) -> dict:
    """
    Convert and save multiple figures efficiently in batch

    Args:
        figs: List of figures to convert
        output_dir: Output directory
        base_name: Base name for files
        fmt: Image format
        max_workers: Maximum worker threads

    Returns:
        Dictionary with conversion results and timing info
    """
    import time
    from ..utils import make_dir

    start_time = time.time()
    make_dir(output_dir)

    # Convert to bytes
    image_bytes = prepare_image_byte(figs, fmt, max_workers)

    # Save files
    saved_files = save_as_image_file(
        figs, base_name, 0, "000", fmt, output_dir, with_sequence=True
    )

    end_time = time.time()

    return {
        'saved_files': saved_files,
        'num_files': len(saved_files),
        'total_time': end_time - start_time,
        'avg_time_per_file': (end_time - start_time) / len(saved_files) if saved_files else 0
    }

def image_bytes_to_files(
    image_bytes_list: List[bytes],
    output_dir: str = "images",
    base_name: str = "img",
    fmt: str = "png"
) -> List[str]:
    """
    Save image bytes directly to files

    Args:
        image_bytes_list: List of image bytes
        output_dir: Output directory
        base_name: Base name for files
        fmt: File format extension

    Returns:
        List of saved filenames
    """
    from ..utils import make_dir, reserve_available_names

    make_dir(output_dir)

    if not image_bytes_list:
        return []

    # Generate filenames
    filenames = reserve_available_names(base_name, len(image_bytes_list), output_dir, fmt)

    # Save files
    def save_bytes_to_file(args):
        img_bytes, filename = args
        if not img_bytes:
            return None

        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'wb') as f:
                f.write(img_bytes)
            return filename
        except Exception as e:
            print(f"Error saving {filename}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=min(8, len(image_bytes_list))) as executor:
        saved_files = list(executor.map(save_bytes_to_file, zip(image_bytes_list, filenames)))

    return [f for f in saved_files if f is not None]

def optimize_image_quality(
    fig: go.Figure,
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0
) -> go.Figure:
    """
    Optimize figure for high-quality image output

    Args:
        fig: Plotly figure to optimize
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for resolution

    Returns:
        Optimized figure
    """
    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=12),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return fig

def create_image_grid(
    image_bytes_list: List[bytes],
    grid_cols: int = 3,
    output_path: str = "image_grid.png"
) -> str:
    """
    Create a grid of images from image bytes

    Args:
        image_bytes_list: List of image bytes
        grid_cols: Number of columns in grid
        output_path: Output file path

    Returns:
        Path to saved grid image
    """
    try:
        from PIL import Image
        import math

        if not image_bytes_list:
            return ""

        # Load images
        images = []
        for img_bytes in image_bytes_list:
            if img_bytes:
                img = Image.open(io.BytesIO(img_bytes))
                images.append(img)

        if not images:
            return ""

        # Calculate grid dimensions
        n_images = len(images)
        grid_rows = math.ceil(n_images / grid_cols)

        # Get image dimensions (assume all same size)
        img_width, img_height = images[0].size

        # Create grid image
        grid_width = grid_cols * img_width
        grid_height = grid_rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height), 'white')

        # Place images in grid
        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * img_width
            y = row * img_height
            grid_image.paste(img, (x, y))

        # Save grid
        grid_image.save(output_path)
        return output_path

    except ImportError:
        print("PIL not available for image grid creation")
        return ""
    except Exception as e:
        print(f"Error creating image grid: {e}")
        return ""
