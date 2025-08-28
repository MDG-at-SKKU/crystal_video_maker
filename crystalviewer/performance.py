"""
Performance monitoring utilities
Memory usage tracking, timing, and performance optimization
"""

import time
import warnings
from typing import Dict, Any, Callable, Optional
from functools import wraps

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def get_memory_usage() -> float:
    """Get current memory usage in MB"""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0.0


def check_memory_limit(max_memory_mb: float = 1024) -> bool:
    """Check if memory usage is within limits"""
    return get_memory_usage() < max_memory_mb


def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution"""
    @wraps(func)
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

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        for operation, times in self.metrics.items():
            summary[operation] = {
                'count': len(times),
                'total_time': sum(times),
                'average_time': sum(times) / len(times) if times else 0,
                'min_time': min(times) if times else 0,
                'max_time': max(times) if times else 0
            }
        return summary


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
