"""Laspeyres value-weighted index implementation"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import date
from dataclasses import dataclass

from ..models.weights import WeightSet, WeightType
from ..algorithms.regression import RegressionResults


@dataclass
class LaspeyresIndexResult:
    """Results from Laspeyres index calculation"""
    index_values: pd.DataFrame
    base_period_weights: WeightSet
    coverage_rate: float
    missing_tracts: List[str]


class LaspeyresIndex:
    """
    Calculate Laspeyres value-weighted price index
    
    The Laspeyres index uses base period quantities (housing values)
    as weights, making it a fixed-weight index that measures pure
    price change holding the housing stock constant.
    
    Formula: L_t = Σ(p_it * q_i0) / Σ(p_i0 * q_i0)
    where:
        p_it = price index for tract i at time t
        q_i0 = housing value (units × median value) for tract i in base period
    """
    
    def __init__(self, base_period: Optional[date] = None):
        """
        Initialize Laspeyres index calculator
        
        Args:
            base_period: Base period for fixed weights (default: first period)
        """
        self.base_period = base_period
        self._base_weights = None
        
    def calculate_index(self,
                       tract_indices: Dict[str, pd.DataFrame],
                       weights: Dict[str, WeightSet],
                       geography_id: str,
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None) -> LaspeyresIndexResult:
        """
        Calculate Laspeyres index for a geography
        
        Args:
            tract_indices: Dict mapping tract/supertract IDs to DataFrames with
                         columns ['date', 'index_value']
            weights: Dict mapping period keys to WeightSets
            geography_id: CBSA ID or 'national'
            start_date: Start date for index
            end_date: End date for index
            
        Returns:
            LaspeyresIndexResult with aggregated index
        """
        # Determine date range
        all_dates = set()
        for df in tract_indices.values():
            if 'date' in df.columns:
                all_dates.update(pd.to_datetime(df['date']).dt.date)
        
        if not all_dates:
            raise ValueError("No valid dates found in tract indices")
            
        all_dates = sorted(all_dates)
        
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
            
        if not all_dates:
            raise ValueError("No dates in specified range")
            
        # Determine base period
        if self.base_period is None:
            base_date = all_dates[0]
        else:
            base_date = self.base_period
            
        # Get base period weights
        base_year = base_date.year
        base_weight_key = f"{geography_id}_{base_year}"
        
        if base_weight_key not in weights:
            # Try to find closest year
            available_years = []
            for key in weights:
                if geography_id in key:
                    try:
                        year = int(key.split('_')[-1])
                        available_years.append(year)
                    except:
                        pass
            
            if available_years:
                closest_year = min(available_years, key=lambda y: abs(y - base_year))
                base_weight_key = f"{geography_id}_{closest_year}"
            else:
                raise ValueError(f"No weights available for {geography_id}")
                
        base_weights = weights[base_weight_key]
        
        # Check weight type
        if base_weights.weight_type != WeightType.VALUE:
            print(f"Warning: Using {base_weights.weight_type} weights instead of VALUE weights for Laspeyres index")
        
        # Calculate aggregated index for each date
        index_data = []
        missing_tracts_all = set()
        
        for date in all_dates:
            # Aggregate across tracts using base period weights
            weighted_sum = 0.0
            weight_sum = 0.0
            missing_tracts = []
            
            for tract_id, weight in base_weights.weights.items():
                # Find index value for this tract and date
                tract_index = None
                
                # Check if we have data for this tract
                if tract_id in tract_indices:
                    tract_df = tract_indices[tract_id]
                    date_mask = pd.to_datetime(tract_df['date']).dt.date == date
                    if date_mask.any():
                        tract_index = tract_df.loc[date_mask, 'index_value'].iloc[0]
                
                # Also check supertracts that might contain this tract
                if tract_index is None:
                    for super_id, super_df in tract_indices.items():
                        if self._tract_in_supertract(tract_id, super_id):
                            date_mask = pd.to_datetime(super_df['date']).dt.date == date
                            if date_mask.any():
                                tract_index = super_df.loc[date_mask, 'index_value'].iloc[0]
                                break
                
                if tract_index is not None:
                    weighted_sum += weight * tract_index
                    weight_sum += weight
                else:
                    missing_tracts.append(tract_id)
                    missing_tracts_all.add(tract_id)
            
            # Calculate aggregated index
            if weight_sum > 0:
                aggregated_index = weighted_sum / weight_sum
                
                # Renormalize to account for missing tracts
                if len(missing_tracts) > 0:
                    # Scale up by the inverse of coverage
                    coverage = weight_sum  # Since base weights sum to 1
                    aggregated_index = aggregated_index  # Keep as is - don't rescale
                
                index_data.append({
                    'date': date,
                    'index_value': aggregated_index,
                    'coverage': weight_sum,
                    'missing_tracts': len(missing_tracts)
                })
        
        # Create result DataFrame
        result_df = pd.DataFrame(index_data)
        
        # Calculate overall coverage rate
        total_weight = sum(base_weights.weights.values())
        covered_weight = total_weight - sum(
            base_weights.weights.get(t, 0) for t in missing_tracts_all
        )
        coverage_rate = covered_weight / total_weight if total_weight > 0 else 0.0
        
        return LaspeyresIndexResult(
            index_values=result_df,
            base_period_weights=base_weights,
            coverage_rate=coverage_rate,
            missing_tracts=list(missing_tracts_all)
        )
    
    def calculate_from_regression_results(self,
                                        regression_results: Dict[str, RegressionResults],
                                        weights: Dict[str, WeightSet],
                                        geography_id: str,
                                        base_value: float = 100.0) -> LaspeyresIndexResult:
        """
        Calculate Laspeyres index from regression results
        
        Args:
            regression_results: Dict mapping tract/supertract IDs to regression results
            weights: Dict mapping period keys to WeightSets
            geography_id: CBSA ID or 'national'
            base_value: Base index value (default 100)
            
        Returns:
            LaspeyresIndexResult
        """
        # Convert regression results to index DataFrames
        tract_indices = {}
        
        for tract_id, results in regression_results.items():
            # Convert log returns to index values
            index_values = base_value * np.exp(results.log_returns)
            
            tract_indices[tract_id] = pd.DataFrame({
                'date': results.period_dates,
                'index_value': index_values
            })
        
        return self.calculate_index(
            tract_indices=tract_indices,
            weights=weights,
            geography_id=geography_id
        )
    
    def _tract_in_supertract(self, tract_id: str, super_id: str) -> bool:
        """Check if tract is part of supertract"""
        # Simple check based on ID format
        if super_id == f"super_{tract_id}":
            return True
        if tract_id in super_id:
            return True
        return False
    
    def chain_index(self,
                   yearly_indices: Dict[int, pd.DataFrame],
                   yearly_weights: Dict[int, WeightSet]) -> pd.DataFrame:
        """
        Chain yearly Laspeyres indices together
        
        This is used when weights are updated annually rather than
        using fixed base period weights.
        
        Args:
            yearly_indices: Dict mapping years to index DataFrames
            yearly_weights: Dict mapping years to weight sets
            
        Returns:
            DataFrame with chained index
        """
        # Sort years
        years = sorted(yearly_indices.keys())
        if not years:
            raise ValueError("No yearly indices provided")
        
        # Start with first year
        chained = yearly_indices[years[0]].copy()
        chained['chained_index'] = chained['index_value']
        
        # Chain subsequent years
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            
            # Get link factor (last value of previous year)
            prev_df = yearly_indices[prev_year]
            link_factor = prev_df['index_value'].iloc[-1] / 100.0
            
            # Apply link factor to current year
            curr_df = yearly_indices[curr_year].copy()
            curr_df['chained_index'] = curr_df['index_value'] * link_factor
            
            # Append to chained series
            chained = pd.concat([chained, curr_df], ignore_index=True)
        
        return chained