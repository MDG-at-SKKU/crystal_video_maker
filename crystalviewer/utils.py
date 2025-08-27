"""
Utility functions for Rendering
"""

import os
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import hashlib
import pickle
from functools import lru_cache
import time

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy not available - some optimizations disabled")

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Cache for expensive computations
_mesh_cache = {}
_color_cache = {}


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0.0


def check_memory_limit(max_memory_mb: float = 1024) -> bool:
    """Check if memory usage is within limits"""
    return get_memory_usage() < max_memory_mb


def create_directory(path: str) -> None:
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_hash(filepath: str) -> str:
    """Get hash of file for caching purposes"""
    if not os.path.exists(filepath):
        return ""

    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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


def cache_mesh_data(key: str, mesh_data: Any) -> None:
    """Cache mesh data for reuse"""
    global _mesh_cache
    if len(_mesh_cache) > 100:
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


def timing_decorator(func: callable) -> callable:
    """Decorator to time function execution"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def get_optimal_batch_size(memory_limit_mb: float = 512) -> int:
    """Calculate optimal batch size based on memory limits"""
    base_memory_per_atom = 0.1  # MB per atom (approximate)
    max_atoms = int(memory_limit_mb / base_memory_per_atom)
    return min(max_atoms, 10000)


def validate_coordinates(coordinates: Any) -> bool:
    """Validate coordinate array format"""
    if not HAS_NUMPY:
        return isinstance(coordinates, (list, tuple)) and len(coordinates) > 0

    try:
        coords = np.array(coordinates)
        return coords.ndim == 2 and coords.shape[1] == 3
    except (ValueError, TypeError):
        return False


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division that handles division by zero"""
    return a / b if b != 0 else default


def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range"""
    return max(min_val, min(max_val, value))


def format_number(num: float, precision: int = 3) -> str:
    """Format number with consistent precision"""
    return f"{num:.{precision}f}"


def get_unique_elements(elements: List[str]) -> List[str]:
    """Get unique elements while preserving order"""
    seen = set()
    unique = []
    for element in elements:
        if element not in seen:
            seen.add(element)
            unique.append(element)
    return unique


def create_progress_callback(
    total_steps: int, description: str = "Processing"
) -> callable:
    """Create progress callback function"""

    def progress_callback(current_step: int, message: str = ""):
        progress = (current_step / total_steps) * 100
        print(f"{description}: {progress:.1f}% - {message}")

    return progress_callback


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


# Performance monitoring
class PerformanceMonitor:
    """Monitor performance metrics"""

    def __init__(self):
        self.start_time = None
        self.metrics = {}

    def start(self, operation: str):
        """Start timing an operation"""
        self.start_time = time.time()
        if operation not in self.metrics:
            self.metrics[operation] = []

    def stop(self, operation: str):
        """Stop timing and record metrics"""
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        self.metrics[operation].append(elapsed)
        self.start_time = None

    def get_average_time(self, operation: str) -> float:
        """Get average time for operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        return sum(self.metrics[operation]) / len(self.metrics[operation])

    def get_total_time(self, operation: str) -> float:
        """Get total time for operation"""
        if operation not in self.metrics:
            return 0.0
        return sum(self.metrics[operation])

    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.start_time = None


# Global performance monitor instance
# performance_monitor = PerformanceMonitor()
