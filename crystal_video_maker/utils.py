"""
Utility functions with massive performance improvements through caching and batch processing
"""

import os
import shutil
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Set, Optional, Union
import multiprocessing as mp

# Global thread pool for I/O operations
_thread_pool = ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 1) + 4))

@lru_cache(maxsize=1000)
def _get_dir_files(directory: str) -> Set[str]:
    """
    Cached directory file listing

    Args:
        directory: Directory path to scan

    Returns:
        Set of filenames in the directory
    """
    try:
        return set(os.listdir(directory)) if os.path.exists(directory) else set()
    except (OSError, PermissionError):
        return set()

def make_dir(target_path: str) -> None:
    """
    Create directory with error handling

    Args:
        target_path: Path to directory to create
    """
    try:
        os.makedirs(target_path, exist_ok=True)
    except OSError as e:
        print(f"Error: Failed to create directory : {target_path}\n{e}")

def check_name_available(input_name: str, result_dir: str = "result") -> bool:
    """
    Check if filename is available in directory

    Args:
        input_name: Filename to check
        result_dir: Directory to check in

    Returns:
        True if filename is available, False otherwise
    """
    target_path = Path(result_dir) / input_name
    return not target_path.exists()

def batch_check_names_available(names: List[str], result_dir: str = "result") -> List[bool]:
    """
    Batch check multiple filenames for availability

    Args:
        names: List of filenames to check
        result_dir: Directory to check in

    Returns:
        List of boolean values indicating availability
    """
    existing_files = _get_dir_files(result_dir)
    return [name not in existing_files for name in names]

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
        base_name: Base name for files
        count: Number of names to reserve
        result_dir: Directory to check
        fmt: File format extension
        numbering_rule: Numbering format (e.g., "000" for zero-padding)

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

def parallel_file_operation(operation, files: List[str], max_workers: Optional[int] = None) -> List:
    """
    Execute file operations in parallel

    Args:
        operation: Function to apply to each file
        files: List of file paths
        max_workers: Maximum number of worker threads

    Returns:
        List of operation results
    """
    max_workers = max_workers or min(16, len(files), (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(operation, files))

def clear_cache():
    """
    Clear all LRU caches for memory management
    """
    _get_dir_files.cache_clear()
    from .constants import get_element_radius, get_scaled_radii_dict
    get_element_radius.cache_clear()
    get_scaled_radii_dict.cache_clear()
