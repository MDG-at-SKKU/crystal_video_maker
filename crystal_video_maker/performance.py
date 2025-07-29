"""
Performance monitoring and benchmarking utilities.
"""
import time
import functools
import cProfile
import pstats
from typing import Callable, Any, Dict
import numpy as np


def profile_performance(func: Callable) -> Callable:
    """Decorator to profile function performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        profiler.disable()
        
        # Print timing info
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        
        # Print top function calls
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative').print_stats(10)
        
        return result
    return wrapper


def benchmark_optimization(
    original_func: Callable,
    optimized_func: Callable,
    test_data: Any,
    iterations: int = 5
) -> Dict[str, float]:
    """Benchmark original vs optimized function performance."""
    
    def time_function(func, data, iters):
        times = []
        for _ in range(iters):
            start = time.time()
            func(data)
            end = time.time()
            times.append(end - start)
        return np.mean(times), np.std(times)
    
    original_mean, original_std = time_function(original_func, test_data, iterations)
    optimized_mean, optimized_std = time_function(optimized_func, test_data, iterations)
    
    speedup = original_mean / optimized_mean
    
    return {
        'original_time': original_mean,
        'original_std': original_std,
        'optimized_time': optimized_mean,
        'optimized_std': optimized_std,
        'speedup': speedup,
        'improvement_percent': (speedup - 1) * 100
    }


class PerformanceTracker:
    """Track performance metrics during execution."""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def time_block(self, name: str):
        """Context manager for timing code blocks."""
        return TimedBlock(name, self)
    
    def log_timing(self, name: str, duration: float):
        """Log timing information."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary."""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
                'count': len(times)
            }
        return summary


class TimedBlock:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, tracker: PerformanceTracker):
        self.name = name
        self.tracker = tracker
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.tracker.log_timing(self.name, duration)


# Global performance tracker
performance_tracker = PerformanceTracker()
