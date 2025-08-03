"""Performance optimization utilities"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import psutil
import gc


logger = logging.getLogger(__name__)


class DataOptimizer:
    """Optimize data structures for performance"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, 
                          deep: bool = True,
                          categorical_threshold: float = 0.5) -> pd.DataFrame:
        """Optimize DataFrame memory usage
        
        Args:
            df: DataFrame to optimize
            deep: Whether to perform deep optimization
            categorical_threshold: Threshold for converting to categorical
            
        Returns:
            Optimized DataFrame
        """
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=['int', 'float']).columns:
            df_optimized[col] = DataOptimizer._optimize_numeric_column(df_optimized[col])
        
        # Optimize object columns
        for col in df_optimized.select_dtypes(include=['object']).columns:
            # Convert to categorical if appropriate
            if df_optimized[col].nunique() / len(df_optimized) < categorical_threshold:
                df_optimized[col] = df_optimized[col].astype('category')
            elif deep:
                # Try to infer better dtype
                try:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                except:
                    try:
                        df_optimized[col] = pd.to_datetime(df_optimized[col])
                    except:
                        pass
        
        # Log memory reduction
        original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - optimized_memory / original_memory) * 100
        
        logger.info(f"Memory optimization: {original_memory:.1f}MB -> {optimized_memory:.1f}MB "
                   f"({reduction:.1f}% reduction)")
        
        return df_optimized
    
    @staticmethod
    def _optimize_numeric_column(col: pd.Series) -> pd.Series:
        """Optimize numeric column dtype"""
        col_min = col.min()
        col_max = col.max()
        
        # Check if integer
        if col.dtype.kind in 'iu':
            if col_min >= 0:
                # Unsigned integer optimization
                if col_max < 255:
                    return col.astype(np.uint8)
                elif col_max < 65535:
                    return col.astype(np.uint16)
                elif col_max < 4294967295:
                    return col.astype(np.uint32)
            else:
                # Signed integer optimization
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    return col.astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    return col.astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    return col.astype(np.int32)
        
        # Float optimization
        elif col.dtype.kind == 'f':
            if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                return col.astype(np.float32)
        
        return col
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """Split DataFrame into chunks for processing
        
        Args:
            df: DataFrame to chunk
            chunk_size: Size of each chunk (auto-determined if None)
            
        Returns:
            List of DataFrame chunks
        """
        if chunk_size is None:
            # Auto-determine based on memory
            available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
            df_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Use 25% of available memory per chunk
            chunk_memory = available_memory * 0.25
            chunk_size = max(1, int(len(df) * chunk_memory / df_memory))
        
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunks.append(df.iloc[i:i + chunk_size])
        
        logger.info(f"Split DataFrame into {len(chunks)} chunks of size {chunk_size}")
        return chunks


class MemoryOptimizer:
    """Optimize memory usage during computation"""
    
    def __init__(self, target_memory_mb: Optional[float] = None):
        """Initialize memory optimizer
        
        Args:
            target_memory_mb: Target memory usage in MB
        """
        if target_memory_mb is None:
            # Use 75% of available memory
            target_memory_mb = psutil.virtual_memory().available / 1024 / 1024 * 0.75
        
        self.target_memory_mb = target_memory_mb
        
    def optimize_computation(self, func, *args, **kwargs):
        """Run computation with memory optimization
        
        Args:
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        # Force garbage collection before
        gc.collect()
        
        # Monitor memory during execution
        initial_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Force garbage collection after
            gc.collect()
        
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB "
                   f"(+{memory_increase:.1f}MB)")
        
        return result
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    @staticmethod
    def estimate_memory_requirements(df: pd.DataFrame, 
                                   operations: List[str]) -> Dict[str, float]:
        """Estimate memory requirements for operations
        
        Args:
            df: Input DataFrame
            operations: List of operations to perform
            
        Returns:
            Dictionary of operation -> estimated memory (MB)
        """
        base_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        estimates = {}
        
        operation_multipliers = {
            'sort': 1.5,
            'groupby': 2.0,
            'merge': 2.5,
            'pivot': 3.0,
            'regression': 2.0,
            'matrix_multiply': 2.5
        }
        
        for op in operations:
            multiplier = operation_multipliers.get(op, 1.5)
            estimates[op] = base_memory * multiplier
        
        estimates['total'] = sum(estimates.values())
        return estimates


class ComputationOptimizer:
    """Optimize computational operations"""
    
    @staticmethod
    def optimize_matrix_operations(A: np.ndarray, B: Optional[np.ndarray] = None) -> Any:
        """Optimize matrix operations
        
        Args:
            A: First matrix
            B: Optional second matrix
            
        Returns:
            Optimization recommendations
        """
        recommendations = []
        
        # Check if sparse
        sparsity = 1 - (np.count_nonzero(A) / A.size)
        if sparsity > 0.7:
            recommendations.append("Use sparse matrix representation")
        
        # Check condition number
        if A.shape[0] == A.shape[1]:  # Square matrix
            try:
                cond = np.linalg.cond(A)
                if cond > 1e10:
                    recommendations.append("Matrix is ill-conditioned, consider regularization")
            except:
                pass
        
        # Check for symmetry
        if A.shape[0] == A.shape[1]:
            if np.allclose(A, A.T):
                recommendations.append("Matrix is symmetric, use specialized algorithms")
        
        # Size recommendations
        if A.size > 1e7:
            recommendations.append("Consider chunking or out-of-core computation")
        
        return recommendations
    
    @staticmethod
    def vectorize_operation(func, data: Union[pd.DataFrame, np.ndarray]) -> Any:
        """Vectorize operation for better performance
        
        Args:
            func: Function to vectorize
            data: Data to operate on
            
        Returns:
            Vectorized result
        """
        if isinstance(data, pd.DataFrame):
            # Use DataFrame vectorization
            return data.apply(func, axis=1, raw=True)
        else:
            # Use NumPy vectorization
            return np.vectorize(func)(data)


class ParallelProcessor:
    """Parallel processing utilities"""
    
    def __init__(self, n_workers: Optional[int] = None, use_threads: bool = False):
        """Initialize parallel processor
        
        Args:
            n_workers: Number of workers (None for auto)
            use_threads: Use threads instead of processes
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        self.n_workers = n_workers
        self.use_threads = use_threads
        self.executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
    def process_chunks(self, func, chunks: List[Any], **kwargs) -> List[Any]:
        """Process chunks in parallel
        
        Args:
            func: Function to apply to each chunk
            chunks: List of data chunks
            **kwargs: Additional arguments for func
            
        Returns:
            List of results
        """
        logger.info(f"Processing {len(chunks)} chunks with {self.n_workers} workers")
        
        with self.executor_class(max_workers=self.n_workers) as executor:
            # Create partial function with kwargs
            process_func = partial(func, **kwargs)
            
            # Submit all tasks
            futures = [executor.submit(process_func, chunk) for chunk in chunks]
            
            # Collect results
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    raise
        
        return results
    
    def parallel_apply(self, df: pd.DataFrame, func, axis: int = 1) -> pd.Series:
        """Apply function to DataFrame in parallel
        
        Args:
            df: DataFrame to process
            func: Function to apply
            axis: Axis to apply along
            
        Returns:
            Result series
        """
        # Split DataFrame
        chunks = np.array_split(df, self.n_workers)
        
        # Process in parallel
        with self.executor_class(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(lambda chunk: chunk.apply(func, axis=axis), chunk)
                for chunk in chunks
            ]
            
            results = [future.result() for future in futures]
        
        # Combine results
        return pd.concat(results)
    
    @staticmethod
    def optimize_worker_count(data_size: int, operation_cost: str = 'medium') -> int:
        """Determine optimal number of workers
        
        Args:
            data_size: Size of data to process
            operation_cost: Cost of operation ('light', 'medium', 'heavy')
            
        Returns:
            Optimal number of workers
        """
        cpu_count = mp.cpu_count()
        
        # Base recommendations
        cost_multipliers = {
            'light': 2.0,    # More workers for light operations
            'medium': 1.0,   # Standard
            'heavy': 0.5     # Fewer workers for heavy operations
        }
        
        multiplier = cost_multipliers.get(operation_cost, 1.0)
        
        # Adjust based on data size
        if data_size < 1000:
            return 1  # Serial processing for small data
        elif data_size < 10000:
            return min(2, cpu_count)
        elif data_size < 100000:
            return min(int(cpu_count * multiplier), cpu_count)
        else:
            return cpu_count


def optimize_hpi_calculation(transactions: pd.DataFrame,
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Optimize HPI calculation performance
    
    Args:
        transactions: Transaction data
        config: Optional configuration
        
    Returns:
        Optimization recommendations
    """
    recommendations = {
        'data_optimizations': [],
        'computation_optimizations': [],
        'memory_optimizations': [],
        'parallel_optimizations': []
    }
    
    # Data optimizations
    data_opt = DataOptimizer()
    optimized_df = data_opt.optimize_dataframe(transactions)
    memory_saved = (transactions.memory_usage(deep=True).sum() - 
                   optimized_df.memory_usage(deep=True).sum()) / 1024 / 1024
    
    if memory_saved > 10:
        recommendations['data_optimizations'].append(
            f"Optimize DataFrame dtypes to save {memory_saved:.1f}MB"
        )
    
    # Check for chunking needs
    if len(transactions) > 1000000:
        recommendations['memory_optimizations'].append(
            "Process data in chunks for large datasets"
        )
    
    # Parallel processing recommendations
    if len(transactions) > 10000:
        optimal_workers = ParallelProcessor.optimize_worker_count(
            len(transactions), 'medium'
        )
        recommendations['parallel_optimizations'].append(
            f"Use parallel processing with {optimal_workers} workers"
        )
    
    return recommendations