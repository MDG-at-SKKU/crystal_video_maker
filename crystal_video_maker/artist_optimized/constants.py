"""
Performance constants and optimized utilities for crystal_video_maker.artist
"""
import numpy as np
from typing import Tuple
from functools import lru_cache

# Pre-computed constants for vectorized operations
UNIT_CELL_OFFSETS = np.array(list(
    (i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
    if not (i == 0 and j == 0 and k == 0)
), dtype=np.int8)  # 26 offsets excluding (0,0,0)

IDENTITY_3X3 = np.eye(3, dtype=np.float64)

# Optimized cell edge definitions
CELL_EDGES = np.array([
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
    (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)
], dtype=np.uint8)

@lru_cache(maxsize=256)
def cached_lattice_transform(lattice_matrix_hash: int, coords_hash: int) -> np.ndarray:
    """Cached lattice coordinate transformations."""
    # This would be called with actual matrix and coords data
    pass

# Constants for disordered site rendering optimization
WEDGE_RESOLUTION_MIN = 8
WEDGE_RESOLUTION_MAX = 24
PIE_SLICE_POINTS_MIN = 6
PIE_SLICE_POINTS_MAX = 20
