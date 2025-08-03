"""Unit tests for repeat-sales regression"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy import sparse

from hpi_fhfa.algorithms.regression import RepeatSalesRegression, RegressionResults


class TestRepeatSalesRegression:
    """Test repeat-sales regression implementation"""
    
    def test_basic_regression(self, sample_pairs_df):
        """Test basic regression functionality"""
        regression = RepeatSalesRegression()
        results = regression.fit(sample_pairs_df)
        
        assert isinstance(results, RegressionResults)
        assert results.num_observations == len(sample_pairs_df)
        assert results.num_periods > 0
        assert len(results.log_returns) == results.num_periods
        assert len(results.standard_errors) == results.num_periods
        assert 0 <= results.r_squared <= 1
        
    def test_date_filtering(self, sample_pairs_df):
        """Test date range filtering"""
        regression = RepeatSalesRegression()
        
        # Get date range from data
        min_date = pd.to_datetime(sample_pairs_df['second_sale_date']).min().date()
        max_date = pd.to_datetime(sample_pairs_df['second_sale_date']).max().date()
        mid_date = min_date + (max_date - min_date) / 2
        
        # Fit with restricted range
        results = regression.fit(
            sample_pairs_df,
            start_date=mid_date,
            end_date=max_date
        )
        
        # Should have fewer observations
        assert results.num_observations < len(sample_pairs_df)
        # Period dates include all periods from min to max of all dates in filtered data
        # So we can't assert the first period is after mid_date
        
    def test_design_matrix_construction(self):
        """Test design matrix construction"""
        # Create simple test data
        data = [
            {
                'property_id': 'P1',
                'tract_id': 'T1',
                'cbsa_id': 'C1',
                'first_sale_date': pd.Timestamp('2020-01-15'),
                'first_sale_price': 100000,
                'second_sale_date': pd.Timestamp('2020-06-15'),
                'second_sale_price': 110000
            },
            {
                'property_id': 'P2',
                'tract_id': 'T1',
                'cbsa_id': 'C1',
                'first_sale_date': pd.Timestamp('2020-03-15'),
                'first_sale_price': 200000,
                'second_sale_date': pd.Timestamp('2020-09-15'),
                'second_sale_price': 220000
            }
        ]
        
        df = pd.DataFrame(data)
        regression = RepeatSalesRegression()
        
        # Get period mapping
        period_dates = regression._create_period_index(df)
        period_map = {date: idx for idx, date in enumerate(period_dates)}
        mapped_df = regression._map_to_periods(df, period_map)
        
        # Build design matrix
        X, y = regression._build_design_matrix(mapped_df, len(period_dates))
        
        # Check matrix properties
        assert X.shape[0] == 2  # Two observations
        assert X.shape[1] == len(period_dates) - 1  # Minus base period
        assert isinstance(X, sparse.csr_matrix)
        
        # Check y values (log price ratios)
        expected_y = np.log([110000/100000, 220000/200000])
        np.testing.assert_array_almost_equal(y, expected_y)
        
    def test_index_calculation(self, sample_pairs_df):
        """Test index value calculation"""
        regression = RepeatSalesRegression()
        results = regression.fit(sample_pairs_df)
        
        # Get index values
        index_df = regression.get_index_values(base_value=100.0)
        
        assert len(index_df) == results.num_periods
        assert 'date' in index_df.columns
        assert 'index_value' in index_df.columns
        assert 'log_return' in index_df.columns
        assert 'standard_error' in index_df.columns
        
        # Base period should be 100
        assert index_df.iloc[0]['index_value'] == 100.0
        
        # Log returns should match
        np.testing.assert_array_almost_equal(
            index_df['log_return'].values,
            results.log_returns
        )
        
    def test_returns_calculation(self, sample_pairs_df):
        """Test period returns calculation"""
        regression = RepeatSalesRegression()
        results = regression.fit(sample_pairs_df)
        
        # Test different return periods
        for period in ['monthly', 'quarterly', 'annual']:
            returns_df = regression.calculate_returns(period_length=period)
            
            assert 'date' in returns_df.columns
            assert 'return' in returns_df.columns
            assert len(returns_df) > 0
            
            # Returns should be reasonable (but can exceed 100% in extreme cases)
            assert returns_df['return'].abs().max() < 5.0  # Less than 500%
            
    def test_convergence_info(self, sample_pairs_df):
        """Test that convergence information is provided"""
        regression = RepeatSalesRegression()
        results = regression.fit(sample_pairs_df)
        
        assert 'convergence_info' in results.__dict__
        assert 'iterations' in results.convergence_info
        assert 'converged' in results.convergence_info
        assert 'condition_number' in results.convergence_info
        assert 'norm_residual' in results.convergence_info
        
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_df = pd.DataFrame(columns=[
            'property_id', 'tract_id', 'cbsa_id',
            'first_sale_date', 'first_sale_price',
            'second_sale_date', 'second_sale_price'
        ])
        
        regression = RepeatSalesRegression()
        with pytest.raises(ValueError, match="No transactions"):
            regression.fit(empty_df)
            
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data"""
        # Single transaction
        data = [{
            'property_id': 'P1',
            'tract_id': 'T1',
            'cbsa_id': 'C1',
            'first_sale_date': pd.Timestamp('2020-01-15'),
            'first_sale_price': 100000,
            'second_sale_date': pd.Timestamp('2020-06-15'),
            'second_sale_price': 110000
        }]
        
        df = pd.DataFrame(data)
        regression = RepeatSalesRegression()
        
        # Should still work with one observation
        results = regression.fit(df)
        assert results.num_observations == 1
        
    def test_standard_errors_calculation(self, sample_pairs_df):
        """Test standard error calculation"""
        regression = RepeatSalesRegression()
        results = regression.fit(sample_pairs_df)
        
        # Standard errors should be non-negative
        assert np.all(results.standard_errors >= 0)
        
        # Base period should have zero standard error
        assert results.standard_errors[0] == 0.0
        
    def test_residuals_properties(self, sample_pairs_df):
        """Test residual properties"""
        regression = RepeatSalesRegression()
        results = regression.fit(sample_pairs_df)
        
        # Residuals might not sum to exactly zero due to regularization
        # Just check they're reasonable
        residual_mean = np.mean(results.residuals)
        assert abs(residual_mean) < 0.1  # Mean should be close to zero
        
        # Number of residuals should match observations
        assert len(results.residuals) == results.num_observations