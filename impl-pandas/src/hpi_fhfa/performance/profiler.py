"""Performance profiling utilities"""

import time
import psutil
import tracemalloc
import functools
import logging
from typing import Dict, Any, Callable, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Performance profile result"""
    function_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    cpu_percent: float
    memory_mb: float
    memory_peak_mb: float
    args_info: Dict[str, Any]
    result_info: Dict[str, Any]
    error: Optional[str] = None


class PerformanceProfiler:
    """Profile performance of HPI calculations"""
    
    def __init__(self, track_memory: bool = True, track_cpu: bool = True):
        """Initialize profiler
        
        Args:
            track_memory: Whether to track memory usage
            track_cpu: Whether to track CPU usage
        """
        self.track_memory = track_memory
        self.track_cpu = track_cpu
        self.results: List[ProfileResult] = []
        self._process = psutil.Process()
        
    def profile(self, func: Callable) -> Callable:
        """Decorator to profile a function
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start tracking
            start_time = datetime.now()
            start_cpu = self._process.cpu_percent() if self.track_cpu else 0
            
            if self.track_memory:
                tracemalloc.start()
                start_memory = self._process.memory_info().rss / 1024 / 1024  # MB
            
            error = None
            result = None
            
            try:
                # Execute function
                result = func(*args, **kwargs)
            except Exception as e:
                error = str(e)
                raise
            finally:
                # End tracking
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # CPU usage
                cpu_percent = 0
                if self.track_cpu:
                    cpu_percent = self._process.cpu_percent() - start_cpu
                
                # Memory usage
                memory_mb = 0
                memory_peak_mb = 0
                if self.track_memory:
                    current_memory = self._process.memory_info().rss / 1024 / 1024
                    memory_mb = current_memory - start_memory
                    
                    current, peak = tracemalloc.get_traced_memory()
                    memory_peak_mb = peak / 1024 / 1024
                    tracemalloc.stop()
                
                # Extract info about args and result
                args_info = self._extract_args_info(args, kwargs)
                result_info = self._extract_result_info(result) if result else {}
                
                # Create profile result
                profile_result = ProfileResult(
                    function_name=func.__name__,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_peak_mb=memory_peak_mb,
                    args_info=args_info,
                    result_info=result_info,
                    error=error
                )
                
                self.results.append(profile_result)
                logger.info(f"Profiled {func.__name__}: {duration:.2f}s, {memory_mb:.1f}MB")
            
            return result
        
        return wrapper
    
    def _extract_args_info(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract information about function arguments"""
        info = {}
        
        # Extract DataFrame info
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                info[f'arg{i}_shape'] = arg.shape
                info[f'arg{i}_memory_mb'] = arg.memory_usage(deep=True).sum() / 1024 / 1024
            elif isinstance(arg, np.ndarray):
                info[f'arg{i}_shape'] = arg.shape
                info[f'arg{i}_dtype'] = str(arg.dtype)
            elif isinstance(arg, (list, dict)):
                info[f'arg{i}_len'] = len(arg)
        
        # Extract from kwargs
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                info[f'{key}_shape'] = value.shape
            elif isinstance(value, (list, dict)):
                info[f'{key}_len'] = len(value)
                
        return info
    
    def _extract_result_info(self, result: Any) -> Dict[str, Any]:
        """Extract information about function result"""
        info = {}
        
        if isinstance(result, pd.DataFrame):
            info['shape'] = result.shape
            info['memory_mb'] = result.memory_usage(deep=True).sum() / 1024 / 1024
            info['columns'] = list(result.columns)
        elif isinstance(result, np.ndarray):
            info['shape'] = result.shape
            info['dtype'] = str(result.dtype)
        elif isinstance(result, dict):
            info['keys'] = list(result.keys())
            info['size'] = len(result)
        elif isinstance(result, (list, tuple)):
            info['length'] = len(result)
            
        return info
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of profiling results
        
        Returns:
            DataFrame with profiling summary
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                'function': result.function_name,
                'duration_s': result.duration_seconds,
                'cpu_percent': result.cpu_percent,
                'memory_mb': result.memory_mb,
                'peak_memory_mb': result.memory_peak_mb,
                'timestamp': result.start_time,
                'error': result.error is not None
            })
        
        return pd.DataFrame(data)
    
    def get_bottlenecks(self, top_n: int = 5) -> pd.DataFrame:
        """Identify performance bottlenecks
        
        Args:
            top_n: Number of top bottlenecks to return
            
        Returns:
            DataFrame with bottleneck analysis
        """
        summary = self.get_summary()
        if summary.empty:
            return pd.DataFrame()
        
        # Find slowest functions
        slow_funcs = summary.nlargest(top_n, 'duration_s')
        
        # Find memory-intensive functions
        memory_funcs = summary.nlargest(top_n, 'memory_mb')
        
        # Combine and deduplicate
        bottlenecks = pd.concat([slow_funcs, memory_funcs]).drop_duplicates()
        
        # Add bottleneck type
        bottlenecks['bottleneck_type'] = 'mixed'
        bottlenecks.loc[bottlenecks.index.isin(slow_funcs.index), 'bottleneck_type'] = 'time'
        bottlenecks.loc[bottlenecks.index.isin(memory_funcs.index), 'bottleneck_type'] = 'memory'
        
        return bottlenecks.sort_values('duration_s', ascending=False)
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate performance report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        summary = self.get_summary()
        bottlenecks = self.get_bottlenecks()
        
        report = ["Performance Profile Report", "=" * 50, ""]
        
        # Overall statistics
        if not summary.empty:
            report.extend([
                "Overall Statistics:",
                f"Total functions profiled: {len(summary)}",
                f"Total duration: {summary['duration_s'].sum():.2f}s",
                f"Average duration: {summary['duration_s'].mean():.2f}s",
                f"Total memory used: {summary['memory_mb'].sum():.1f}MB",
                f"Peak memory: {summary['peak_memory_mb'].max():.1f}MB",
                ""
            ])
        
        # Function summary
        report.extend(["Function Summary:", "-" * 30])
        for _, row in summary.iterrows():
            report.append(
                f"{row['function']}: {row['duration_s']:.2f}s, "
                f"{row['memory_mb']:.1f}MB"
            )
        
        # Bottlenecks
        if not bottlenecks.empty:
            report.extend(["", "Performance Bottlenecks:", "-" * 30])
            for _, row in bottlenecks.iterrows():
                report.append(
                    f"{row['function']} ({row['bottleneck_type']}): "
                    f"{row['duration_s']:.2f}s, {row['memory_mb']:.1f}MB"
                )
        
        report_str = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
        
        return report_str
    
    def clear(self):
        """Clear profiling results"""
        self.results.clear()


def profile_function(track_memory: bool = True, track_cpu: bool = True):
    """Decorator to profile a single function
    
    Args:
        track_memory: Whether to track memory usage
        track_cpu: Whether to track CPU usage
        
    Returns:
        Decorator function
    """
    profiler = PerformanceProfiler(track_memory=track_memory, track_cpu=track_cpu)
    
    def decorator(func):
        return profiler.profile(func)
    
    return decorator


class ProfileContext:
    """Context manager for profiling code blocks"""
    
    def __init__(self, name: str, profiler: Optional[PerformanceProfiler] = None):
        """Initialize profile context
        
        Args:
            name: Name of the code block
            profiler: Optional profiler instance
        """
        self.name = name
        self.profiler = profiler or PerformanceProfiler()
        self._start_time = None
        self._start_memory = None
        
    def __enter__(self):
        """Enter context"""
        self._start_time = time.time()
        if self.profiler.track_memory:
            tracemalloc.start()
            self._start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context"""
        duration = time.time() - self._start_time
        
        memory_mb = 0
        if self.profiler.track_memory:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_mb = current_memory - self._start_memory
            tracemalloc.stop()
        
        logger.info(f"Profile {self.name}: {duration:.2f}s, {memory_mb:.1f}MB")