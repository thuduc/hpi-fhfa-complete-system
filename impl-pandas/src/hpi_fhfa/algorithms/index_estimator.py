"""BMN Index Estimation with Weighting Schemes"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import date
from dataclasses import dataclass

from .regression import RepeatSalesRegression, RegressionResults
from .supertract import SupertractConstructor
from ..models.weights import WeightSet, WeightType, WeightCalculator
from ..models.geography import Supertract


@dataclass
class IndexResults:
    """Results from index estimation"""
    index_values: pd.DataFrame
    regression_results: Dict[str, RegressionResults]
    supertracts_used: Dict[str, List[Supertract]]
    weights_applied: Dict[str, WeightSet]
    coverage_stats: Dict[str, float]


class BMNIndexEstimator:
    """
    Complete BMN index estimation pipeline with geographic aggregation
    
    This class orchestrates the full index calculation process:
    1. Construct supertracts as needed
    2. Run repeat-sales regressions at tract/supertract level
    3. Apply weighting schemes for aggregation
    4. Calculate final indices at various geographic levels
    """
    
    def __init__(self,
                 min_half_pairs: int = 40,
                 weight_calculator: Optional[WeightCalculator] = None,
                 adjacency_data: Optional[Dict[str, List[str]]] = None):
        """
        Initialize index estimator
        
        Args:
            min_half_pairs: Minimum half-pairs for reliable estimation
            weight_calculator: Calculator for generating weights
            adjacency_data: Tract adjacency relationships
        """
        self.min_half_pairs = min_half_pairs
        self.weight_calculator = weight_calculator
        self.supertract_constructor = SupertractConstructor(
            min_half_pairs=min_half_pairs,
            adjacency_data=adjacency_data
        )
        
    def estimate_indices(self,
                       pairs_df: pd.DataFrame,
                       tract_gdf: pd.DataFrame,
                       weight_type: WeightType,
                       geography_level: str = 'cbsa',
                       start_date: Optional[date] = None,
                       end_date: Optional[date] = None,
                       base_period: Optional[date] = None) -> IndexResults:
        """
        Estimate price indices at specified geography level
        
        Args:
            pairs_df: Valid transaction pairs
            tract_gdf: Tract geographic data
            weight_type: Type of weights to use for aggregation
            geography_level: 'tract', 'cbsa', or 'national'
            start_date: Start date for analysis
            end_date: End date for analysis
            base_period: Base period for index
            
        Returns:
            IndexResults object
        """
        # Ensure dates are datetime
        pairs_df = self._prepare_pairs_data(pairs_df)
        
        # Determine analysis period
        if start_date is None:
            if len(pairs_df) > 0:
                start_date = pairs_df['second_sale_date'].min().date()
            else:
                start_date = date(2020, 1, 1)  # Default if no data
        if end_date is None:
            if len(pairs_df) > 0:
                end_date = pairs_df['second_sale_date'].max().date()
            else:
                end_date = date(2020, 12, 31)  # Default if no data
        
        # Get years to analyze
        years = list(range(start_date.year, end_date.year + 1))
        
        # Construct supertracts for each year
        all_supertracts = {}
        for year in years:
            supertracts = self.supertract_constructor.construct_supertracts(
                pairs_df, tract_gdf, year
            )
            all_supertracts[year] = supertracts
        
        # Run regressions at tract/supertract level
        regression_results = self._run_tract_regressions(
            pairs_df, all_supertracts, start_date, end_date, base_period
        )
        
        # Generate weights
        weights = self._generate_weights(
            tract_gdf, weight_type, geography_level, years
        )
        
        # Aggregate to desired geography level
        aggregated_indices = self._aggregate_indices(
            regression_results, weights, geography_level, tract_gdf
        )
        
        # Calculate coverage statistics
        coverage_stats = self._calculate_coverage(
            pairs_df, regression_results, geography_level
        )
        
        return IndexResults(
            index_values=aggregated_indices,
            regression_results=regression_results,
            supertracts_used=all_supertracts,
            weights_applied=weights,
            coverage_stats=coverage_stats
        )
    
    def _prepare_pairs_data(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure date columns are datetime"""
        df = pairs_df.copy()
        for col in ['first_sale_date', 'second_sale_date']:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        return df
    
    def _run_tract_regressions(self,
                             pairs_df: pd.DataFrame,
                             all_supertracts: Dict[int, Dict[str, Supertract]],
                             start_date: date,
                             end_date: date,
                             base_period: Optional[date]) -> Dict[str, RegressionResults]:
        """Run repeat-sales regressions for each tract/supertract"""
        regression_results = {}
        
        # Get mapping from tracts to supertracts
        tract_to_super = {}
        for year, supertracts in all_supertracts.items():
            for st_id, supertract in supertracts.items():
                for tract_id in supertract.component_tract_ids:
                    if tract_id not in tract_to_super:
                        tract_to_super[tract_id] = st_id
        
        # Group pairs by supertract
        pairs_by_super = {}
        for _, pair in pairs_df.iterrows():
            tract_id = pair['tract_id']
            super_id = tract_to_super.get(tract_id, f"super_{tract_id}")
            
            if super_id not in pairs_by_super:
                pairs_by_super[super_id] = []
            pairs_by_super[super_id].append(pair)
        
        # Run regression for each supertract
        for super_id, super_pairs in pairs_by_super.items():
            if len(super_pairs) < 10:  # Skip if too few pairs
                continue
                
            # Convert to DataFrame
            super_df = pd.DataFrame(super_pairs)
            
            # Run regression
            regression = RepeatSalesRegression(
                base_period=base_period,
                robust_weights=True
            )
            
            try:
                results = regression.fit(super_df, start_date, end_date)
                regression_results[super_id] = results
            except Exception as e:
                print(f"Regression failed for {super_id}: {e}")
                continue
        
        return regression_results
    
    def _generate_weights(self,
                        tract_gdf: pd.DataFrame,
                        weight_type: WeightType,
                        geography_level: str,
                        years: List[int]) -> Dict[str, WeightSet]:
        """Generate weights for aggregation"""
        if self.weight_calculator is None:
            # Use equal weights if no calculator provided
            return self._generate_equal_weights(tract_gdf, geography_level, years)
        
        weights = {}
        
        # Generate weights for each geography and year
        if geography_level == 'cbsa':
            for cbsa_id in tract_gdf['cbsa_id'].unique():
                for year in years:
                    weight_set = self.weight_calculator.calculate_weights(
                        weight_type, cbsa_id, year
                    )
                    if weight_set:
                        weights[f"{cbsa_id}_{year}"] = weight_set
        elif geography_level == 'national':
            for year in years:
                # Aggregate all CBSAs to national
                national_weights = {}
                for cbsa_id in tract_gdf['cbsa_id'].unique():
                    weight_set = self.weight_calculator.calculate_weights(
                        weight_type, cbsa_id, year
                    )
                    if weight_set:
                        for tract_id, weight in weight_set.weights.items():
                            national_weights[tract_id] = weight
                
                if national_weights:
                    # Normalize
                    total = sum(national_weights.values())
                    normalized = {k: v/total for k, v in national_weights.items()}
                    
                    weights[f"national_{year}"] = WeightSet(
                        geography_id='national',
                        period=year,
                        weight_type=weight_type,
                        weights=normalized
                    )
        
        return weights
    
    def _generate_equal_weights(self,
                              tract_gdf: pd.DataFrame,
                              geography_level: str,
                              years: List[int]) -> Dict[str, WeightSet]:
        """Generate equal weights as fallback"""
        weights = {}
        
        if geography_level == 'cbsa':
            for cbsa_id in tract_gdf['cbsa_id'].unique():
                cbsa_tracts = tract_gdf[tract_gdf['cbsa_id'] == cbsa_id]
                tract_ids = cbsa_tracts.index.tolist() if cbsa_tracts.index.name == 'tract_id' else cbsa_tracts['tract_id'].tolist()
                
                for year in years:
                    equal_weight = 1.0 / len(tract_ids)
                    weights[f"{cbsa_id}_{year}"] = WeightSet(
                        geography_id=cbsa_id,
                        period=year,
                        weight_type=WeightType.SAMPLE,
                        weights={tid: equal_weight for tid in tract_ids}
                    )
        
        return weights
    
    def _aggregate_indices(self,
                         regression_results: Dict[str, RegressionResults],
                         weights: Dict[str, WeightSet],
                         geography_level: str,
                         tract_gdf: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tract/supertract indices to desired geography level"""
        # Get all unique dates from regression results
        all_dates = set()
        for results in regression_results.values():
            all_dates.update(results.period_dates)
        all_dates = sorted(all_dates)
        
        # Initialize aggregated indices
        aggregated = []
        
        if geography_level == 'cbsa':
            # Aggregate by CBSA
            for cbsa_id in tract_gdf['cbsa_id'].unique():
                cbsa_indices = self._aggregate_cbsa_indices(
                    cbsa_id, regression_results, weights, all_dates, tract_gdf
                )
                for date, index_value in cbsa_indices.items():
                    aggregated.append({
                        'date': date,
                        'geography_id': cbsa_id,
                        'geography_type': 'cbsa',
                        'index_value': index_value
                    })
        
        elif geography_level == 'national':
            # Aggregate to national level
            national_indices = self._aggregate_national_indices(
                regression_results, weights, all_dates
            )
            for date, index_value in national_indices.items():
                aggregated.append({
                    'date': date,
                    'geography_id': 'USA',
                    'geography_type': 'national',
                    'index_value': index_value
                })
        
        return pd.DataFrame(aggregated)
    
    def _aggregate_cbsa_indices(self,
                              cbsa_id: str,
                              regression_results: Dict[str, RegressionResults],
                              weights: Dict[str, WeightSet],
                              all_dates: List[date],
                              tract_gdf: pd.DataFrame) -> Dict[date, float]:
        """Aggregate indices for a specific CBSA"""
        cbsa_indices = {}
        
        # Get tracts in this CBSA
        if tract_gdf.index.name == 'tract_id':
            cbsa_tracts = set(tract_gdf[tract_gdf['cbsa_id'] == cbsa_id].index)
        else:
            cbsa_tracts = set(tract_gdf[tract_gdf['cbsa_id'] == cbsa_id]['tract_id'])
        
        for date in all_dates:
            year = date.year
            weight_key = f"{cbsa_id}_{year}"
            
            if weight_key not in weights:
                continue
            
            weight_set = weights[weight_key]
            weighted_sum = 0.0
            weight_sum = 0.0
            
            # Aggregate across tracts/supertracts
            for tract_id, weight in weight_set.weights.items():
                if tract_id not in cbsa_tracts:
                    continue
                
                # Find regression results for this tract
                # (might be part of a supertract)
                for super_id, results in regression_results.items():
                    if self._tract_in_supertract(tract_id, super_id):
                        # Find index value for this date
                        if date in results.period_dates:
                            date_idx = results.period_dates.index(date)
                            log_return = results.log_returns[date_idx]
                            index_value = 100.0 * np.exp(log_return)
                            
                            weighted_sum += weight * index_value
                            weight_sum += weight
                            break
            
            if weight_sum > 0:
                cbsa_indices[date] = weighted_sum / weight_sum
        
        return cbsa_indices
    
    def _aggregate_national_indices(self,
                                  regression_results: Dict[str, RegressionResults],
                                  weights: Dict[str, WeightSet],
                                  all_dates: List[date]) -> Dict[date, float]:
        """Aggregate indices to national level"""
        national_indices = {}
        
        for date in all_dates:
            year = date.year
            weight_key = f"national_{year}"
            
            if weight_key not in weights:
                continue
            
            weight_set = weights[weight_key]
            weighted_sum = 0.0
            weight_sum = 0.0
            
            # Similar aggregation logic as CBSA but across all tracts
            for tract_id, weight in weight_set.weights.items():
                for super_id, results in regression_results.items():
                    if self._tract_in_supertract(tract_id, super_id):
                        if date in results.period_dates:
                            date_idx = results.period_dates.index(date)
                            log_return = results.log_returns[date_idx]
                            index_value = 100.0 * np.exp(log_return)
                            
                            weighted_sum += weight * index_value
                            weight_sum += weight
                            break
            
            if weight_sum > 0:
                national_indices[date] = weighted_sum / weight_sum
        
        return national_indices
    
    def _tract_in_supertract(self, tract_id: str, super_id: str) -> bool:
        """Check if tract is part of supertract"""
        # Simple check based on ID format
        if super_id == f"super_{tract_id}":
            return True
        if tract_id in super_id:
            return True
        return False
    
    def _calculate_coverage(self,
                          pairs_df: pd.DataFrame,
                          regression_results: Dict[str, RegressionResults],
                          geography_level: str) -> Dict[str, float]:
        """Calculate coverage statistics"""
        total_pairs = len(pairs_df)
        covered_pairs = sum(r.num_observations for r in regression_results.values())
        
        return {
            'total_pairs': total_pairs,
            'covered_pairs': covered_pairs,
            'coverage_rate': covered_pairs / total_pairs if total_pairs > 0 else 0.0,
            'num_regressions': len(regression_results),
            'geography_level': geography_level
        }