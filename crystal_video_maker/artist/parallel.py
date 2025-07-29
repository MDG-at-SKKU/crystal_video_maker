from tqdm import tqdm
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from crystal_video_maker.artist.plot import structure_3d
import multiprocessing as mp


def create_figures_with_progress(xdatcar, max_workers=None):
    if max_workers is None:
        max_workers = min(len(xdatcar), mp.cpu_count())
    
    figs = [None] * len(xdatcar)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 작업 제출
        future_to_index = {
            executor.submit(structure_3d, xdatcar[i]): i 
            for i in range(len(xdatcar))
        }
        
        # 진행 상황 표시와 함께 결과 수집
        for future in tqdm(as_completed(future_to_index), 
                          total=len(xdatcar), 
                          desc="Creating structure figures"):
            index = future_to_index[future]
            try:
                figs[index] = future.result()
            except Exception as exc:
                print(f'Structure {index} generated an exception: {exc}')
    
    return figs



def optimized_crystal_video_workflow(xdatcar, max_workers=None):
    
    if max_workers is None:
        max_workers = min(len(xdatcar), mp.cpu_count())
    
    print(f"Processing {len(xdatcar)} structures with {max_workers} workers...")
    
    # 병렬로 fig 객체 생성
    figs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(structure_3d, xdatcar[i]): i 
            for i in range(len(xdatcar))
        }
        
        figs = [None] * len(xdatcar)
        for future in tqdm(as_completed(future_to_index), 
                          total=len(xdatcar), 
                          desc="Creating figures"):
            index = future_to_index[future]
            figs[index] = future.result()
    
    # 병렬로 image byte 변환
    def fig_to_bytes(fig):
        return fig.to_image(format="png")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        image_bytes = list(tqdm(
            executor.map(fig_to_bytes, figs),
            total=len(figs),
            desc="Converting to bytes"
        ))
    
    return figs, image_bytes


def batch_crystal_video_workflow(xdatcar, batch_size=20, max_workers=2):
    
    all_figs = []
    all_image_bytes = []
    
    for start_idx in tqdm(range(0, len(xdatcar), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(xdatcar))
        batch = xdatcar[start_idx:end_idx]
        
        # 배치 단위로 처리
        batch_figs = []
        batch_bytes = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # fig 생성
            batch_figs = list(executor.map(structure_3d, batch))
            
            # 이미지 변환
            batch_bytes = list(executor.map(lambda f: f.to_image(format="png"), batch_figs))
        
        all_figs.extend(batch_figs)
        all_image_bytes.extend(batch_bytes)
        
        import gc
        gc.collect()
        
        import time
        time.sleep(0.1)
    
    return all_figs, all_image_bytes

