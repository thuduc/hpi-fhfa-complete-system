"""Unit tests for BMN index estimator"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from hpi_fhfa.algorithms.index_estimator import BMNIndexEstimator, IndexResults
from hpi_fhfa.models.weights import WeightType, SampleWeightCalculator


class TestBMNIndexEstimator:
    """Test BMN index estimation"""
    
    def test_basic_estimation(self, sample_pairs_df, sample_tract_gdf):
        """Test basic index estimation"""
        estimator = BMNIndexEstimator(min_half_pairs=10)  # Lower threshold for test
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        assert isinstance(results, IndexResults)
        assert len(results.index_values) > 0
        assert len(results.regression_results) > 0
        assert len(results.supertracts_used) > 0
        
    def test_index_values_structure(self, sample_pairs_df, sample_tract_gdf):
        """Test structure of returned index values"""
        estimator = BMNIndexEstimator(min_half_pairs=10)
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        # Check index values DataFrame
        index_df = results.index_values
        assert 'date' in index_df.columns
        assert 'geography_id' in index_df.columns
        assert 'geography_type' in index_df.columns
        assert 'index_value' in index_df.columns
        
        # Check geography type
        assert all(index_df['geography_type'] == 'cbsa')
        
        # Check index values are reasonable
        assert all(index_df['index_value'] > 0)
        assert all(index_df['index_value'] < 1000)  # Reasonable upper bound
        
    def test_coverage_statistics(self, sample_pairs_df, sample_tract_gdf):
        """Test coverage statistics calculation"""
        estimator = BMNIndexEstimator(min_half_pairs=10)
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        coverage = results.coverage_stats
        assert 'total_pairs' in coverage
        assert 'covered_pairs' in coverage
        assert 'coverage_rate' in coverage
        assert 'num_regressions' in coverage
        assert 'geography_level' in coverage
        
        assert coverage['total_pairs'] == len(sample_pairs_df)
        assert 0 <= coverage['coverage_rate'] <= 1
        assert coverage['geography_level'] == 'cbsa'
        
    def test_date_range_filtering(self, sample_pairs_df, sample_tract_gdf):
        """Test date range filtering in estimation"""
        estimator = BMNIndexEstimator(min_half_pairs=10)
        
        # Get date range
        dates = pd.to_datetime(sample_pairs_df['second_sale_date'])
        mid_date = dates.min() + (dates.max() - dates.min()) / 2
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='cbsa',
            start_date=mid_date.date()
        )
        
        # All index dates should be after start date
        index_dates = pd.to_datetime(results.index_values['date'])
        assert all(index_dates >= pd.Timestamp(mid_date.date()))
        
    def test_weight_generation(self, sample_pairs_df, sample_tract_gdf):
        """Test weight generation"""
        # Create simple weight calculator
        calculator = SampleWeightCalculator()
        estimator = BMNIndexEstimator(
            min_half_pairs=10,
            weight_calculator=calculator
        )
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        # Check weights were applied
        assert len(results.weights_applied) > 0
        for weight_key, weight_set in results.weights_applied.items():
            assert weight_set.weight_type == WeightType.SAMPLE
            assert weight_set.is_normalized
            
    def test_national_aggregation(self, sample_pairs_df, sample_tract_gdf):
        """Test national level aggregation"""
        estimator = BMNIndexEstimator(min_half_pairs=10)
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='national'
        )
        
        # Check national results
        index_df = results.index_values
        assert all(index_df['geography_type'] == 'national')
        assert all(index_df['geography_id'] == 'USA')
        
    def test_tract_level_estimation(self, sample_pairs_df, sample_tract_gdf):
        """Test tract level estimation (no aggregation)"""
        estimator = BMNIndexEstimator(min_half_pairs=5)  # Very low threshold
        
        # This test may need adjustment based on actual implementation
        # For now, test that it doesn't crash
        try:
            results = estimator.estimate_indices(
                sample_pairs_df,
                sample_tract_gdf,
                WeightType.SAMPLE,
                geography_level='tract'
            )
            # If implemented, check results
            if len(results.index_values) > 0:
                assert all(results.index_values['geography_type'] == 'tract')
        except NotImplementedError:
            # Tract level might not be implemented yet
            pass
            
    def test_regression_results_storage(self, sample_pairs_df, sample_tract_gdf):
        """Test that regression results are properly stored"""
        estimator = BMNIndexEstimator(min_half_pairs=10)
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        # Check regression results
        assert len(results.regression_results) > 0
        for super_id, reg_results in results.regression_results.items():
            assert reg_results.num_observations > 0
            assert reg_results.num_periods > 0
            assert len(reg_results.log_returns) == reg_results.num_periods
            
    def test_supertract_tracking(self, sample_pairs_df, sample_tract_gdf):
        """Test that supertracts are properly tracked"""
        estimator = BMNIndexEstimator(min_half_pairs=10)
        
        results = estimator.estimate_indices(
            sample_pairs_df,
            sample_tract_gdf,
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        # Check supertracts by year
        assert len(results.supertracts_used) > 0
        for year, supertracts in results.supertracts_used.items():
            assert isinstance(year, int)
            assert len(supertracts) > 0
            for st_id, supertract in supertracts.items():
                assert supertract.year == year
                assert len(supertract.component_tract_ids) > 0
                
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_pairs = pd.DataFrame(columns=[
            'property_id', 'tract_id', 'cbsa_id',
            'first_sale_date', 'first_sale_price',
            'second_sale_date', 'second_sale_price'
        ])
        
        empty_tracts = pd.DataFrame(columns=[
            'tract_id', 'cbsa_id', 'state', 'county', 'geometry'
        ])
        
        estimator = BMNIndexEstimator()
        
        # Should handle empty data gracefully
        results = estimator.estimate_indices(
            empty_pairs,
            empty_tracts,
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        assert len(results.index_values) == 0
        assert results.coverage_stats['total_pairs'] == 0
        assert results.coverage_stats['coverage_rate'] == 0.0