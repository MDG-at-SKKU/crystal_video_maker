"""
Image processing and manipulation functions with high-performance batch operations
"""

import os
import io
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import warnings
from ..utils import make_dir

import plotly.graph_objects as go

@lru_cache(maxsize=1000)
def _get_dir_files(directory: str) -> set:
    """
    Get cached set of files in directory for fast lookup
    
    Args:
        directory: Directory path
        
    Returns:
        Set of filenames in directory
    """
    try:
        return set(os.listdir(directory)) if os.path.exists(directory) else set()
    except (OSError, PermissionError):
        return set()

def check_name_available(input_name: str, result_dir: str = "result") -> bool:
    """
    Check if filename is available in directory
    
    Args:
        input_name: Filename to check
        result_dir: Directory to check in
        
    Returns:
        True if name is available
    """
    existing_files = _get_dir_files(result_dir)
    return input_name not in existing_files

def prepare_image_name_with_checker(
    name: str = "fig",
    number: int = 0,
    numbering_rule: str = "000",
    fmt: str = "png",
    result_dir: str = "result"
) -> str:
    """
    Generate available image filename with numbering
    
    Args:
        name: Base filename
        number: Starting number
        numbering_rule: Numbering format (e.g., "000" for 3 digits)
        fmt: File format extension
        result_dir: Target directory
        
    Returns:
        Available filename
    """
    width = numbering_rule.count("0")
    i = number
    
    while True:
        img_name = f"{name}-{i:0{width}d}.{fmt}"
        if check_name_available(img_name, result_dir):
            return img_name
        i += 1

def reserve_available_names(
    base_name: str,
    count: int,
    result_dir: str = "result",
    fmt: str = "png",
    numbering_rule: str = "000"
) -> List[str]:
    """
    Reserve multiple available filenames efficiently
    
    Args:
        base_name: Base filename
        count: Number of names to reserve
        result_dir: Target directory
        fmt: File format extension
        numbering_rule: Numbering format
        
    Returns:
        List of available filenames
    """
    width = numbering_rule.count("0")
    existing_files = _get_dir_files(result_dir)
    
    available_names = []
    i = 0
    while len(available_names) < count:
        name = f"{base_name}-{i:0{width}d}.{fmt}"
        if name not in existing_files:
            available_names.append(name)
        i += 1
    
    return available_names

def prepare_image_byte(figs: Union[go.Figure, List[go.Figure]]) -> Union[bytes, List[bytes]]:
    """
    Convert Plotly figures to image bytes (legacy function for compatibility)
    
    Args:
        figs: Single figure or list of figures
        
    Returns:
        Image bytes or list of image bytes
    """
    if isinstance(figs, go.Figure):
        return figs.to_image(format="png")
    elif isinstance(figs, list):
        return [fig.to_image(format="png") for fig in figs]
    else:
        raise ValueError("Input must be go.Figure or list of go.Figure")

def images_to_bytes(figs: Union[go.Figure, List[go.Figure]]) -> List[bytes]:
    """
    Convert Plotly figures to image bytes with parallel processing
    
    Args:
        figs: Single figure or list of figures
        
    Returns:
        List of image bytes
    """
    if isinstance(figs, go.Figure):
        figs = [figs]
    
    if not figs:
        return []
    
    # Use parallel processing for multiple images
    if len(figs) > 1:
        with ThreadPoolExecutor(max_workers=min(16, len(figs))) as executor:
            return list(executor.map(lambda fig: fig.to_image(format="png"), figs))
    else:
        return [figs[0].to_image(format="png")]

def save_as_image_file(
    figs: Union[go.Figure, List[go.Figure]],
    name: str = "fig",
    number: int = 0,
    result_dir: str = "result",
    fmt: str = "png",
    numbering_rule: str = "000"
) -> List[str]:
    """
    Save figures as image files (legacy function for compatibility)
    
    Args:
        figs: Single figure or list of figures
        name: Base filename
        number: Starting number
        result_dir: Target directory
        fmt: File format
        numbering_rule: Numbering format
        
    Returns:
        List of saved filenames
    """
    if isinstance(figs, go.Figure):
        figs = [figs]
    
    make_dir(result_dir)
    saved_files = []
    
    for i, fig in enumerate(figs):
        filename = prepare_image_name_with_checker(
            name, number + i, numbering_rule, fmt, result_dir
        )
        filepath = os.path.join(result_dir, filename)
        fig.write_image(filepath)
        saved_files.append(filename)
    
    return saved_files

def batch_save_images(
    figs: List[go.Figure],
    base_name: str = "fig",
    result_dir: str = "result",
    fmt: str = "png",
    numbering_rule: str = "000",
    parallel: bool = True
) -> List[str]:
    """
    Save multiple images efficiently with batch processing
    
    Args:
        figs: List of Plotly figures
        base_name: Base filename
        result_dir: Target directory
        fmt: File format extension
        numbering_rule: Numbering format
        parallel: Whether to use parallel processing
        
    Returns:
        List of saved filenames
    """
    if not figs:
        return []
    
    # Create directory
    make_dir(result_dir)
    
    # Reserve all filenames at once
    filenames = reserve_available_names(
        base_name, len(figs), result_dir, fmt, numbering_rule
    )
    
    def save_single_image(args):
        fig, filename = args
        filepath = os.path.join(result_dir, filename)
        fig.write_image(filepath)
        return filename
    
    if parallel and len(figs) > 1:
        # Parallel saving
        with ThreadPoolExecutor(max_workers=min(8, len(figs))) as executor:
            saved_files = list(executor.map(save_single_image, zip(figs, filenames)))
    else:
        # Sequential saving
        saved_files = [save_single_image((fig, filename)) for fig, filename in zip(figs, filenames)]
    
    return saved_files

def batch_convert_to_bytes(
    figs: List[go.Figure],
    format: str = "png",
    **kwargs
) -> List[bytes]:
    """
    Convert multiple figures to bytes in batch
    
    Args:
        figs: List of Plotly figures
        format: Image format
        **kwargs: Additional arguments for to_image()
        
    Returns:
        List of image bytes
    """
    if not figs:
        return []
    
    def convert_single(fig):
        return fig.to_image(format=format, **kwargs)
    
    # Use parallel processing for better performance
    with ThreadPoolExecutor(max_workers=min(16, len(figs))) as executor:
        return list(executor.map(convert_single, figs))

def save_images_with_metadata(
    figs: List[go.Figure],
    metadata: List[Dict[str, Any]],
    base_name: str = "fig",
    result_dir: str = "result",
    fmt: str = "png"
) -> List[Dict[str, str]]:
    """
    Save images with associated metadata
    
    Args:
        figs: List of Plotly figures
        metadata: List of metadata dictionaries
        base_name: Base filename
        result_dir: Target directory
        fmt: File format
        
    Returns:
        List of dictionaries with filename and metadata
    """
    if len(figs) != len(metadata):
        raise ValueError("Number of figures must match number of metadata entries")
    
    make_dir(result_dir)
    results = []
    
    filenames = reserve_available_names(base_name, len(figs), result_dir, fmt)
    
    for fig, meta, filename in zip(figs, metadata, filenames):
        filepath = os.path.join(result_dir, filename)
        fig.write_image(filepath)
        
        result = {
            'filename': filename,
            'filepath': filepath,
            **meta
        }
        results.append(result)
    
    return results

def create_image_gallery_html(
    image_files: List[str],
    title: str = "Image Gallery",
    output_path: str = "gallery.html"
) -> str:
    """
    Create HTML gallery from image files
    
    Args:
        image_files: List of image file paths
        title: Gallery title
        output_path: Output HTML file path
        
    Returns:
        Path to created HTML file
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .image-container {{ text-align: center; }}
            .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .image-title {{ margin-top: 10px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <div class="gallery">
    """
    
    for i, img_file in enumerate(image_files):
        html_content += f"""
            <div class="image-container">
                <img src="{img_file}" alt="Image {i+1}">
                <div class="image-title">{os.path.basename(img_file)}</div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path

# Legacy aliases for backward compatibility
prepare_image_bytes = images_to_bytes
save_images_batch = batch_save_images
