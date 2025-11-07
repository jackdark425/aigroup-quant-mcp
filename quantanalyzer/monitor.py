"""
Performance monitoring utilities for quantanalyzer
"""
import time
import psutil
import functools
import logging
from typing import Any, Callable, Dict, Optional
from memory_profiler import memory_usage


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking execution time and resource usage
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def monitor(self, 
                name: Optional[str] = None, 
                log_level: int = logging.INFO,
                track_memory: bool = False) -> Callable:
        """
        Decorator to monitor function performance
        
        Args:
            name: Name for the monitoring (defaults to function name)
            log_level: Logging level for the monitoring messages
            track_memory: Whether to track memory usage
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_name = name or func.__name__
                
                # Get initial system stats
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                initial_cpu = process.cpu_percent()
                
                self.logger.log(log_level, f"开始执行 {func_name}")
                start_time = time.time()
                
                try:
                    if track_memory:
                        # Track memory usage during function execution
                        mem_usage, result = memory_usage(
                            (func, args, kwargs), 
                            retval=True,
                            interval=0.1
                        )
                        max_memory = max(mem_usage) if mem_usage else 0
                    else:
                        result = func(*args, **kwargs)
                        max_memory = None
                except Exception as e:
                    elapsed_time = time.time() - start_time
                    self.logger.error(f"{func_name} 执行出错，耗时: {elapsed_time:.2f}秒")
                    raise e
                
                # Get final system stats
                end_time = time.time()
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                final_cpu = process.cpu_percent()
                
                elapsed_time = end_time - start_time
                memory_diff = final_memory - initial_memory
                
                # Log performance metrics
                if track_memory and max_memory:
                    self.logger.log(
                        log_level, 
                        f"{func_name} 执行完成 - "
                        f"耗时: {elapsed_time:.2f}秒, "
                        f"内存变化: {memory_diff:.2f}MB, "
                        f"峰值内存: {max_memory:.2f}MB"
                    )
                else:
                    self.logger.log(
                        log_level, 
                        f"{func_name} 执行完成 - "
                        f"耗时: {elapsed_time:.2f}秒, "
                        f"内存变化: {memory_diff:.2f}MB"
                    )
                
                return result
            
            return wrapper
        return decorator
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """
        Get current system statistics
        
        Returns:
            Dictionary with system statistics
        """
        process = psutil.Process()
        
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'system_cpu_percent': psutil.cpu_percent(),
            'system_memory_percent': psutil.virtual_memory().percent,
            'system_memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }


# Global performance monitor instance
monitor = PerformanceMonitor()


def profile_function(name: Optional[str] = None, 
                    log_level: int = logging.INFO,
                    track_memory: bool = False) -> Callable:
    """
    Decorator to profile function performance
    
    Args:
        name: Name for the profiling (defaults to function name)
        log_level: Logging level for the profiling messages
        track_memory: Whether to track memory usage
        
    Returns:
        Decorator function
    """
    return monitor.monitor(name, log_level, track_memory)