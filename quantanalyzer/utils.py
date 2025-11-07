"""
Utility functions for quantanalyzer
"""
import functools
import hashlib
import pickle
import os
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import logging


def parallelize_dataframe_operation(
    df: pd.DataFrame,
    func: Callable,
    groupby_level: Union[int, str] = 0,
    max_workers: Optional[int] = None,
    use_process_pool: bool = False,
    chunk_size: Optional[int] = None
) -> pd.DataFrame:
    """
    Parallelize DataFrame operations by grouping
    
    Args:
        df: Input DataFrame
        func: Function to apply to each group
        groupby_level: Level to group by (0 for first level, 1 for second level, etc.)
        max_workers: Maximum number of workers (default: number of CPUs)
        use_process_pool: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        chunk_size: Size of chunks to process (for very large datasets)
        
    Returns:
        Processed DataFrame
    """
    logger = logging.getLogger(__name__)
    
    # Group the DataFrame
    grouped = df.groupby(level=groupby_level)
    
    # If chunk_size is specified, process in chunks
    if chunk_size is not None:
        logger.debug(f"Processing in chunks of size {chunk_size}")
        groups = list(grouped)
        results = []
        
        # Process in chunks
        for i in range(0, len(groups), chunk_size):
            chunk = groups[i:i+chunk_size]
            chunk_dict = dict(chunk)
            
            # Create executor
            executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor
            with executor_class(max_workers=max_workers) as executor:
                # Submit tasks
                futures = {
                    executor.submit(func, group): name 
                    for name, group in chunk_dict.items()
                }
                
                # Collect results
                chunk_results = {}
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        chunk_results[name] = future.result()
                    except Exception as e:
                        logger.error(f"Error processing group {name}: {e}")
                        raise RuntimeError(f"Error processing group {name}: {e}")
            
            results.append(chunk_results)
        
        # Combine all results
        combined_results = {}
        for chunk_result in results:
            combined_results.update(chunk_result)
        
        result_dict = combined_results
    else:
        # Process all at once
        # Create executor
        executor_class = ProcessPoolExecutor if use_process_pool else ThreadPoolExecutor
        with executor_class(max_workers=max_workers) as executor:
            # Submit tasks
            futures = {
                executor.submit(func, group): name 
                for name, group in grouped
            }
            
            # Collect results
            result_dict = {}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result_dict[name] = future.result()
                except Exception as e:
                    logger.error(f"Error processing group {name}: {e}")
                    raise RuntimeError(f"Error processing group {name}: {e}")
    
    # Combine results
    if result_dict:
        result_df = pd.concat(result_dict.values())
        # Preserve the original index order if possible
        # Use a safer reindexing approach to avoid issues with non-unique indices
        try:
            # 确保索引类型匹配
            if isinstance(result_df.index, pd.MultiIndex) and isinstance(df.index, pd.MultiIndex):
                # 对于MultiIndex，我们需要更小心地处理
                return result_df.sort_index()
            else:
                return result_df.reindex(df.index, level=groupby_level)
        except (NotImplementedError, TypeError):
            # Fallback: return without reindexing but sort by index
            return result_df.sort_index()
    else:
        return pd.DataFrame()


def memoize(func: Callable) -> Callable:
    """
    Memoization decorator for functions with hashable arguments
    
    Args:
        func: Function to memoize
        
    Returns:
        Memoized function
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a key from the arguments
        key = str(args) + str(sorted(kwargs.items()))
        key_hash = hashlib.md5(key.encode()).hexdigest()
        
        if key_hash in cache:
            return cache[key_hash]
        
        result = func(*args, **kwargs)
        cache[key_hash] = result
        return result
    
    return wrapper


def cache_dataframe_computation(
    cache_dir: str = ".cache"
) -> Callable:
    """
    Decorator to cache DataFrame computations to disk
    
    Args:
        cache_dir: Directory to store cache files
        
    Returns:
        Decorator function
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Try to load from cache
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass  # If cache is corrupted, recompute
            
            # Compute and cache result
            result = func(*args, **kwargs)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except:
                pass  # If caching fails, continue anyway
            
            return result
        
        return wrapper
    
    return decorator


def optimize_dataframe_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame dtypes to reduce memory usage
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with optimized dtypes
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type != 'object':
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df_optimized[col] = df_optimized[col].astype(np.float32)
        
        # Convert object columns with few unique values to category
        elif df_optimized[col].dtype == 'object':
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized


def get_progress_bar(total: int, description: str = "Processing") -> Callable:
    """
    Create a progress bar callback function
    
    Args:
        total: Total number of items to process
        description: Description of the process
        
    Returns:
        Callback function to update progress
    """
    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc=description)
        
        def update(n: int = 1):
            pbar.update(n)
            
        def close():
            pbar.close()
            
        update.close = close
        return update
    except ImportError:
        # If tqdm is not available, return a dummy function
        def update(n: int = 1):
            pass
            
        def close():
            pass
            
        update.close = close
        return update