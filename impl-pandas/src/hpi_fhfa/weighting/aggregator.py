"""Geographic aggregation pipeline for multi-level index calculation"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import date
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.weights import WeightSet, WeightType
from ..algorithms.regression import RegressionResults
from .laspeyres import LaspeyresIndex, LaspeyresIndexResult


@dataclass
class AggregationLevel:
    """Definition of a geographic aggregation level"""
    name: str  # e.g., 'tract', 'cbsa', 'state', 'national'
    parent_level: Optional[str]  # e.g., 'cbsa' for tract, 'national' for cbsa
    id_column: str  # Column name in data, e.g., 'tract_id', 'cbsa_id'


@dataclass
class AggregatedIndices:
    """Results from multi-level aggregation"""
    indices_by_level: Dict[str, pd.DataFrame]  # level -> DataFrame with indices
    weights_used: Dict[str, Dict[str, WeightSet]]  # level -> geography -> weights
    coverage_stats: Dict[str, Dict[str, float]]  # level -> geography -> coverage
    aggregation_tree: Dict[str, List[str]]  # parent -> list of children


class GeographicAggregator:
    """
    Aggregate price indices across multiple geographic levels
    
    This class handles the hierarchical aggregation of indices from
    tract level up through CBSA, state, and national levels, applying
    appropriate weights at each level.
    """
    
    def __init__(self,
                 aggregation_levels: Optional[List[AggregationLevel]] = None,
                 parallel: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize aggregator
        
        Args:
            aggregation_levels: List of aggregation levels (default: tract->cbsa->national)
            parallel: Whether to use parallel processing
            max_workers: Maximum number of parallel workers
        """
        if aggregation_levels is None:
            # Default FHFA hierarchy
            aggregation_levels = [
                AggregationLevel('tract', 'cbsa', 'tract_id'),
                AggregationLevel('cbsa', 'national', 'cbsa_id'),
                AggregationLevel('national', None, 'national')
            ]
        
        self.aggregation_levels = {level.name: level for level in aggregation_levels}
        self.parallel = parallel
        self.max_workers = max_workers
        
    def aggregate_indices(self,
                        base_indices: Dict[str, Union[pd.DataFrame, RegressionResults]],
                        weights: Dict[str, WeightSet],
                        geography_mapping: pd.DataFrame,
                        weight_type: WeightType = WeightType.VALUE,
                        base_period: Optional[date] = None) -> AggregatedIndices:
        """
        Aggregate indices across multiple geographic levels
        
        Args:
            base_indices: Base level indices (tract or supertract)
                        Can be DataFrames with ['date', 'index_value'] or RegressionResults
            weights: Weight sets by geography and period
            geography_mapping: DataFrame with geographic hierarchy
                             (e.g., tract_id, cbsa_id, state_id columns)
            weight_type: Type of weights to use
            base_period: Base period for indices
            
        Returns:
            AggregatedIndices with results at all levels
        """
        # Convert RegressionResults to DataFrames if needed
        tract_indices = self._prepare_base_indices(base_indices)
        
        # Initialize results
        indices_by_level = {}
        weights_used = {}
        coverage_stats = {}
        aggregation_tree = {}
        
        # Start with base level (tract/supertract)
        base_level_name = 'tract'
        indices_by_level[base_level_name] = self._combine_tract_indices(tract_indices)
        
        # Aggregate up the hierarchy
        current_indices = tract_indices
        current_level = self.aggregation_levels.get(base_level_name)
        
        while current_level and current_level.parent_level:
            parent_level = self.aggregation_levels[current_level.parent_level]
            
            # Get unique parent geographies
            if parent_level.name == 'national':
                parent_ids = ['USA']
            else:
                parent_ids = geography_mapping[parent_level.id_column].unique()
            
            # Aggregate to parent level
            if self.parallel and len(parent_ids) > 1:
                parent_indices, parent_weights, parent_coverage = \
                    self._parallel_aggregate_to_level(
                        current_indices, weights, geography_mapping,
                        current_level, parent_level, weight_type, base_period
                    )
            else:
                parent_indices, parent_weights, parent_coverage = \
                    self._aggregate_to_level(
                        current_indices, weights, geography_mapping,
                        current_level, parent_level, weight_type, base_period
                    )
            
            # Store results
            indices_by_level[parent_level.name] = self._combine_geography_indices(parent_indices)
            weights_used[parent_level.name] = parent_weights
            coverage_stats[parent_level.name] = parent_coverage
            
            # Build aggregation tree
            for parent_id in parent_ids:
                if current_level.name == 'tract':
                    # Get tracts in this parent
                    mask = geography_mapping[parent_level.id_column] == parent_id
                    if geography_mapping.index.name == current_level.id_column:
                        children = geography_mapping.loc[mask].index.tolist()
                    elif current_level.id_column in geography_mapping.columns:
                        children = geography_mapping.loc[mask, current_level.id_column].tolist()
                    else:
                        # If tract_id is neither index nor column, try using the index
                        children = geography_mapping.loc[mask].index.tolist()
                else:
                    # Get all geographies at current level
                    children = list(current_indices.keys())
                    
                aggregation_tree[parent_id] = children
            
            # Move up the hierarchy
            current_indices = parent_indices
            current_level = parent_level
        
        return AggregatedIndices(
            indices_by_level=indices_by_level,
            weights_used=weights_used,
            coverage_stats=coverage_stats,
            aggregation_tree=aggregation_tree
        )
    
    def _prepare_base_indices(self,
                            base_indices: Dict[str, Union[pd.DataFrame, RegressionResults]],
                            base_value: float = 100.0) -> Dict[str, pd.DataFrame]:
        """Convert RegressionResults to DataFrames if needed"""
        prepared = {}
        
        for geo_id, data in base_indices.items():
            if isinstance(data, RegressionResults):
                # Convert to DataFrame
                index_values = base_value * np.exp(data.log_returns)
                prepared[geo_id] = pd.DataFrame({
                    'date': data.period_dates,
                    'index_value': index_values,
                    'geography_id': geo_id
                })
            else:
                # Already a DataFrame
                df = data.copy()
                if 'geography_id' not in df.columns:
                    df['geography_id'] = geo_id
                prepared[geo_id] = df
                
        return prepared
    
    def _aggregate_to_level(self,
                          current_indices: Dict[str, pd.DataFrame],
                          weights: Dict[str, WeightSet],
                          geography_mapping: pd.DataFrame,
                          current_level: AggregationLevel,
                          parent_level: AggregationLevel,
                          weight_type: WeightType,
                          base_period: Optional[date]) -> Tuple[Dict, Dict, Dict]:
        """Aggregate indices from current level to parent level"""
        parent_indices = {}
        parent_weights = {}
        parent_coverage = {}
        
        # Get unique parent geographies
        if parent_level.name == 'national':
            parent_ids = ['USA']
        else:
            parent_ids = geography_mapping[parent_level.id_column].unique()
        
        # Use Laspeyres index calculator
        laspeyres = LaspeyresIndex(base_period=base_period)
        
        for parent_id in parent_ids:
            # Get children for this parent
            if current_level.name == 'tract':
                mask = geography_mapping[parent_level.id_column] == parent_id
                if geography_mapping.index.name == current_level.id_column:
                    child_ids = geography_mapping.loc[mask].index.tolist()
                elif current_level.id_column in geography_mapping.columns:
                    child_ids = geography_mapping.loc[mask, current_level.id_column].tolist()
                else:
                    # If tract_id is neither index nor column, try using the index
                    child_ids = geography_mapping.loc[mask].index.tolist()
            else:
                # For higher levels, need different logic
                child_ids = [k for k in current_indices.keys() if self._is_child_of(k, parent_id, geography_mapping)]
            
            # Get indices for children
            child_indices = {cid: current_indices[cid] for cid in child_ids if cid in current_indices}
            
            if not child_indices:
                continue
            
            # Calculate aggregated index
            try:
                result = laspeyres.calculate_index(
                    tract_indices=child_indices,
                    weights=weights,
                    geography_id=parent_id if parent_level.name != 'national' else 'national'
                )
                
                parent_indices[parent_id] = result.index_values
                parent_weights[parent_id] = result.base_period_weights
                parent_coverage[parent_id] = result.coverage_rate
                
            except Exception as e:
                print(f"Failed to aggregate {parent_id}: {e}")
                continue
        
        return parent_indices, parent_weights, parent_coverage
    
    def _parallel_aggregate_to_level(self,
                                   current_indices: Dict[str, pd.DataFrame],
                                   weights: Dict[str, WeightSet],
                                   geography_mapping: pd.DataFrame,
                                   current_level: AggregationLevel,
                                   parent_level: AggregationLevel,
                                   weight_type: WeightType,
                                   base_period: Optional[date]) -> Tuple[Dict, Dict, Dict]:
        """Parallel version of _aggregate_to_level"""
        parent_indices = {}
        parent_weights = {}
        parent_coverage = {}
        
        # Get unique parent geographies
        if parent_level.name == 'national':
            parent_ids = ['USA']
        else:
            parent_ids = geography_mapping[parent_level.id_column].unique()
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_parent = {}
            for parent_id in parent_ids:
                future = executor.submit(
                    self._aggregate_single_geography,
                    parent_id, current_indices, weights, geography_mapping,
                    current_level, parent_level, weight_type, base_period
                )
                future_to_parent[future] = parent_id
            
            # Collect results
            for future in as_completed(future_to_parent):
                parent_id = future_to_parent[future]
                try:
                    indices, weights_used, coverage = future.result()
                    if indices is not None:
                        parent_indices[parent_id] = indices
                        parent_weights[parent_id] = weights_used
                        parent_coverage[parent_id] = coverage
                except Exception as e:
                    print(f"Failed to aggregate {parent_id}: {e}")
        
        return parent_indices, parent_weights, parent_coverage
    
    def _aggregate_single_geography(self,
                                  parent_id: str,
                                  current_indices: Dict[str, pd.DataFrame],
                                  weights: Dict[str, WeightSet],
                                  geography_mapping: pd.DataFrame,
                                  current_level: AggregationLevel,
                                  parent_level: AggregationLevel,
                                  weight_type: WeightType,
                                  base_period: Optional[date]) -> Tuple[Optional[pd.DataFrame], Optional[WeightSet], Optional[float]]:
        """Aggregate a single geography (for parallel processing)"""
        # Get children for this parent
        if current_level.name == 'tract':
            mask = geography_mapping[parent_level.id_column] == parent_id
            if geography_mapping.index.name == current_level.id_column:
                child_ids = geography_mapping.loc[mask].index.tolist()
            elif current_level.id_column in geography_mapping.columns:
                child_ids = geography_mapping.loc[mask, current_level.id_column].tolist()
            else:
                # If tract_id is neither index nor column, try using the index
                child_ids = geography_mapping.loc[mask].index.tolist()
        else:
            child_ids = [k for k in current_indices.keys() if self._is_child_of(k, parent_id, geography_mapping)]
        
        # Get indices for children
        child_indices = {cid: current_indices[cid] for cid in child_ids if cid in current_indices}
        
        if not child_indices:
            return None, None, None
        
        # Calculate aggregated index
        laspeyres = LaspeyresIndex(base_period=base_period)
        
        try:
            result = laspeyres.calculate_index(
                tract_indices=child_indices,
                weights=weights,
                geography_id=parent_id if parent_level.name != 'national' else 'national'
            )
            
            return result.index_values, result.base_period_weights, result.coverage_rate
            
        except Exception as e:
            print(f"Failed to aggregate {parent_id}: {e}")
            return None, None, None
    
    def _is_child_of(self, child_id: str, parent_id: str, geography_mapping: pd.DataFrame) -> bool:
        """Check if child geography belongs to parent"""
        # This is a simplified check - in practice would need proper mapping
        return True  # Placeholder
    
    def _combine_tract_indices(self, tract_indices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all tract indices into single DataFrame"""
        all_indices = []
        
        for tract_id, df in tract_indices.items():
            df_copy = df.copy()
            if 'geography_id' not in df_copy.columns:
                df_copy['geography_id'] = tract_id
            df_copy['geography_type'] = 'tract'
            all_indices.append(df_copy)
        
        if all_indices:
            return pd.concat(all_indices, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _combine_geography_indices(self, geo_indices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine geography indices into single DataFrame"""
        all_indices = []
        
        for geo_id, df in geo_indices.items():
            df_copy = df.copy()
            if 'geography_id' not in df_copy.columns:
                df_copy['geography_id'] = geo_id
            all_indices.append(df_copy)
        
        if all_indices:
            return pd.concat(all_indices, ignore_index=True)
        else:
            return pd.DataFrame()