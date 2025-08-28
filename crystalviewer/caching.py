"""
Caching utilities for expensive computations
Mesh caching, color caching, and general-purpose caching
"""

import os
import hashlib
import pickle
from typing import Dict, Any, Optional, List, Tuple, Union
from functools import lru_cache
import warnings

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Cache for expensive computations
_mesh_cache = {}
_color_cache = {}
_file_hash_cache = {}


def get_file_hash(filepath: str) -> str:
    """Get hash of file for caching purposes"""
    if not os.path.exists(filepath):
        return ""

    # Check cache first
    if filepath in _file_hash_cache:
        return _file_hash_cache[filepath]

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    file_hash = hash_md5.hexdigest()
    _file_hash_cache[filepath] = file_hash
    return file_hash


def cache_mesh_data(key: str, mesh_data: Any) -> None:
    """Cache mesh data for reuse"""
    global _mesh_cache
    if len(_mesh_cache) > 100:
        # Remove oldest entry (simple FIFO)
        oldest_key = next(iter(_mesh_cache))
        del _mesh_cache[oldest_key]

    _mesh_cache[key] = mesh_data


def get_cached_mesh_data(key: str) -> Optional[Any]:
    """Retrieve cached mesh data"""
    return _mesh_cache.get(key)


def clear_mesh_cache() -> None:
    """Clear mesh cache"""
    global _mesh_cache
    _mesh_cache.clear()


def get_mesh_cache_size() -> int:
    """Get number of cached mesh items"""
    return len(_mesh_cache)


@lru_cache(maxsize=1000)
def normalize_color(
    color: Union[str, Tuple[float, float, float], List[float]],
) -> Tuple[float, float, float]:
    """Normalize color input to RGB tuple"""
    if isinstance(color, str):
        color_map = {
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "white": (1.0, 1.0, 1.0),
            "gray": (0.5, 0.5, 0.5),
        }
        return color_map.get(color.lower(), (0.5, 0.5, 0.5))

    elif isinstance(color, (tuple, list)) and len(color) == 3:
        if all(isinstance(c, int) and c > 1 for c in color):
            return tuple(c / 255.0 for c in color)
        return tuple(float(c) for c in color)

    return (0.5, 0.5, 0.5)


@lru_cache(maxsize=500)
def interpolate_colors(
    color1: Tuple[float, float, float],
    color2: Tuple[float, float, float],
    n_steps: int = 10,
) -> List[Tuple[float, float, float]]:
    """Interpolate between two colors"""
    if n_steps < 2:
        return [color1, color2]

    colors = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        interpolated = tuple(c1 + t * (c2 - c1) for c1, c2 in zip(color1, color2))
        colors.append(interpolated)
    return colors


@lru_cache(maxsize=100)
def get_contrasting_color(
    background_color: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Get contrasting text color for background"""
    r, g, b = background_color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b

    return (0.0, 0.0, 0.0) if luminance > 0.5 else (1.0, 1.0, 1.0)


def vectorized_distance(points1: Any, points2: Any) -> Any:
    """Calculate distances between point sets efficiently"""
    if not HAS_NUMPY:
        if hasattr(points1, "__len__") and hasattr(points2, "__len__"):
            return [
                ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
                ** 0.5
                for p1, p2 in zip(points1, points2)
            ]
        return 0.0

    points1 = np.array(points1)
    points2 = np.array(points2)
    return np.linalg.norm(points1 - points2, axis=1)


def batch_process_items(
    items: List[Any], process_func: callable, batch_size: int = 100
) -> List[Any]:
    """Process items in batches for memory efficiency"""
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_results = [process_func(item) for item in batch]
        results.extend(batch_results)
    return results


def export_to_file(data: Any, filepath: str, format: str = "pickle") -> None:
    """Export data to file with different formats"""
    if format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    elif format == "json":
        import json
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    else:
        raise ValueError(f"Unsupported export format: {format}")


def import_from_file(filepath: str, format: str = "pickle") -> Any:
    """Import data from file"""
    if format == "pickle":
        with open(filepath, "rb") as f:
            return pickle.load(f)
    elif format == "json":
        import json
        with open(filepath, "r") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported import format: {format}")


def clear_all_caches() -> None:
    """Clear all caches"""
    clear_mesh_cache()
    _color_cache.clear()
    _file_hash_cache.clear()
    normalize_color.cache_clear()
    interpolate_colors.cache_clear()
    get_contrasting_color.cache_clear()


def get_cache_info() -> Dict[str, Any]:
    """Get information about all caches"""
    return {
        'mesh_cache_size': get_mesh_cache_size(),
        'color_cache_size': len(_color_cache),
        'file_hash_cache_size': len(_file_hash_cache),
        'normalize_color_cache_info': normalize_color.cache_info(),
        'interpolate_colors_cache_info': interpolate_colors.cache_info(),
        'contrasting_color_cache_info': get_contrasting_color.cache_info(),
    }
