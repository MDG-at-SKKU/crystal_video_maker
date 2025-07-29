"""
Parallel processing utilities for crystal_video_maker.artist
"""
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Callable, Tuple
import numpy as np
from functools import partial

from pymatgen.core import Structure, PeriodicSite


class ParallelStructureProcessor:
    """High-performance parallel structure processing."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
    
    def process_structures_parallel(
        self,
        structures: Dict[str, Structure],
        processing_func: Callable,
        use_processes: bool = False,
        **kwargs
    ) -> List[Any]:
        """Process multiple structures in parallel."""
        if len(structures) < 3:  # Use sequential for small datasets
            return [processing_func(struct, **kwargs) for struct in structures.values()]
        
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(processing_func, struct, **kwargs)
                for struct in structures.values()
            ]
            return [future.result() for future in futures]
    
    def process_sites_parallel(
        self,
        sites: List[PeriodicSite],
        processing_func: Callable,
        batch_size: int = 100,
        **kwargs
    ) -> List[Any]:
        """Process sites in parallel batches."""
        if len(sites) < batch_size:
            return [processing_func(site, **kwargs) for site in sites]
        
        # Create batches
        batches = [sites[i:i + batch_size] for i in range(0, len(sites), batch_size)]
        
        def process_batch(batch):
            return [processing_func(site, **kwargs) for site in batch]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batch_results = list(executor.map(process_batch, batches))
        
        # Flatten results
        return [item for batch in batch_results for item in batch]
    
    def vectorized_distance_calculation(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        chunk_size: int = 10000
    ) -> np.ndarray:
        """Vectorized distance calculation with chunking for memory efficiency."""
        n1, n2 = coords1.shape[0], coords2.shape[0]
        
        if n1 * n2 < chunk_size:
            # Direct calculation for small arrays
            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            return np.sqrt(np.sum(diff**2, axis=2))
        
        # Chunked calculation for large arrays
        results = []
        for i in range(0, n1, chunk_size):
            end_i = min(i + chunk_size, n1)
            chunk1 = coords1[i:end_i]
            
            diff = chunk1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            chunk_distances = np.sqrt(np.sum(diff**2, axis=2))
            results.append(chunk_distances)
        
        return np.vstack(results)


# Global instance for easy access
parallel_processor = ParallelStructureProcessor()


def parallel_image_site_generation(
    structures: Dict[str, Structure],
    cell_boundary_tol: float = 0.0,
    max_workers: int = None
) -> Dict[str, List[np.ndarray]]:
    """Generate image sites for multiple structures in parallel."""
    from crystal_video_maker.artist.hands import get_image_sites
    
    def process_structure(struct_item):
        struct_key, structure = struct_item
        image_sites = []
        for site in structure:
            images = get_image_sites(site, structure.lattice, cell_boundary_tol)
            image_sites.extend(images)
        return struct_key, image_sites
    
    max_workers = max_workers or mp.cpu_count()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_structure, item)
            for item in structures.items()
        ]
        results = [future.result() for future in futures]
    
    return dict(results)


def parallel_bond_calculation(
    structure: Structure,
    nn_calculator,
    max_workers: int = None,
    chunk_size: int = 50
) -> List[Tuple[int, List[Dict]]]:
    """Calculate bonds for structure sites in parallel."""
    def calculate_bonds_for_sites(site_indices):
        results = []
        for site_idx in site_indices:
            try:
                connections = nn_calculator.get_nn_info(structure, n=site_idx)
                results.append((site_idx, connections))
            except ValueError as e:
                results.append((site_idx, []))  # Empty connections on error
        return results
    
    # Create chunks of site indices
    n_sites = len(structure)
    site_chunks = [
        list(range(i, min(i + chunk_size, n_sites)))
        for i in range(0, n_sites, chunk_size)
    ]
    
    max_workers = max_workers or min(mp.cpu_count(), len(site_chunks))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        chunk_results = list(executor.map(calculate_bonds_for_sites, site_chunks))
    
    # Flatten results
    return [item for chunk in chunk_results for item in chunk]


def optimize_plotly_traces_parallel(
    trace_data: List[Dict[str, Any]],
    processing_func: Callable,
    max_workers: int = None
) -> List[Dict[str, Any]]:
    """Process Plotly trace data in parallel."""
    if len(trace_data) < 10:  # Sequential for small datasets
        return [processing_func(trace) for trace in trace_data]
    
    max_workers = max_workers or min(mp.cpu_count(), len(trace_data))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(processing_func, trace_data))
