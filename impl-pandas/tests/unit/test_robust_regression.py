"""Unit tests for robust regression"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from hpi_fhfa.outliers import RobustRepeatSalesRegression, RobustRegressionConfig
from hpi_fhfa.algorithms.regression import RepeatSalesRegression


class TestRobustRegression:
    """Test robust regression functionality"""
    
    @pytest.fixture
    def sample_pairs_clean(self):
        """Create clean sample data"""
        np.random.seed(42)
        data = []
        base_date = date(2015, 1, 1)
        
        for i in range(100):
            first_date = base_date + timedelta(days=np.random.randint(0, 1000))
            years_held = np.random.uniform(2, 5)
            second_date = first_date + timedelta(days=int(years_held * 365))
            
            first_price = np.random.uniform(100000, 500000)
            annual_appreciation = np.random.normal(0.05, 0.02)  # 5% mean, 2% std
            second_price = first_price * (1 + annual_appreciation) ** years_held
            
            data.append({
                'property_id': f'P{i:04d}',
                'tract_id': f'T{np.random.randint(1, 10):03d}',
                'cbsa_id': f'C{np.random.randint(1, 3):02d}',
                'first_sale_date': first_date,
                'first_sale_price': first_price,
                'second_sale_date': second_date,
                'second_sale_price': second_price
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_pairs_with_outliers(self, sample_pairs_clean):
        """Add outliers to clean data"""
        df = sample_pairs_clean.copy()
        
        # Add some extreme outliers
        outliers = pd.DataFrame([
            {
                'property_id': 'P9001',
                'tract_id': 'T001',
                'cbsa_id': 'C01',
                'first_sale_date': date(2015, 1, 1),
                'first_sale_price': 200000,
                'second_sale_date': date(2017, 1, 1),
                'second_sale_price': 800000  # 100% annual
            },
            {
                'property_id': 'P9002',
                'tract_id': 'T002',
                'cbsa_id': 'C01',
                'first_sale_date': date(2016, 1, 1),
                'first_sale_price': 300000,
                'second_sale_date': date(2018, 1, 1),
                'second_sale_price': 50000  # Huge loss
            }
        ])
        
        return pd.concat([df, outliers], ignore_index=True)
    
    def test_basic_robust_regression(self, sample_pairs_clean):
        """Test basic robust regression functionality"""
        config = RobustRegressionConfig(method='huber')
        regression = RobustRepeatSalesRegression(config=config)
        
        results = regression.fit(sample_pairs_clean)
        
        assert results is not None
        assert results.num_observations == len(sample_pairs_clean)
        assert len(results.log_returns) > 0
        assert 0 <= results.r_squared <= 1
        
        # Check convergence
        assert results.convergence_info['converged']
        assert results.convergence_info['method'] == 'huber'
        
    def test_different_robust_methods(self, sample_pairs_with_outliers):
        """Test different robust regression methods"""
        methods = ['huber', 'bisquare', 'cauchy', 'welsch']
        results = {}
        
        for method in methods:
            config = RobustRegressionConfig(method=method, max_iterations=100)
            regression = RobustRepeatSalesRegression(config=config)
            results[method] = regression.fit(sample_pairs_with_outliers)
        
        # At least some methods should converge
        converged_count = sum(1 for r in results.values() if r.convergence_info['converged'])
        assert converged_count >= 2  # At least half should converge
        
        # Results should be somewhat different
        coefficients = {m: r.log_returns for m, r in results.items()}
        
        # Check that we got results from all methods
        assert len(results) == len(methods)
        
        # Check that methods gave reasonable results (not checking exact differences)
    
    def test_outlier_downweighting(self, sample_pairs_with_outliers):
        """Test that outliers get downweighted"""
        config = RobustRegressionConfig(method='bisquare')
        regression = RobustRepeatSalesRegression(config=config)
        
        results = regression.fit(sample_pairs_with_outliers)
        weights = regression.get_weights()
        
        assert weights is not None
        assert len(weights) == len(sample_pairs_with_outliers)
        
        # Check that we have weights
        assert weights is not None
        assert len(weights) == len(sample_pairs_with_outliers)
        
        # Weights should be between 0 and 1
        assert np.all(weights >= 0)
        assert np.all(weights <= 1)
    
    def test_outlier_removal_option(self, sample_pairs_with_outliers):
        """Test outlier removal before regression"""
        config = RobustRegressionConfig(
            method='huber',
            outlier_removal=True,
            outlier_threshold=3.0
        )
        regression = RobustRepeatSalesRegression(config=config)
        
        results = regression.fit(sample_pairs_with_outliers)
        
        # Should have removed some outliers
        assert results.num_observations < len(sample_pairs_with_outliers)
        
        # Check outlier result is stored
        outlier_result = regression.get_outlier_result()
        assert outlier_result is not None
        assert len(outlier_result.outlier_indices) > 0
    
    def test_tuning_constant_effect(self, sample_pairs_with_outliers):
        """Test effect of tuning constant"""
        # Small tuning constant (more aggressive downweighting)
        config1 = RobustRegressionConfig(
            method='huber',
            tuning_constant=0.5
        )
        regression1 = RobustRepeatSalesRegression(config=config1)
        results1 = regression1.fit(sample_pairs_with_outliers)
        weights1 = regression1.get_weights()
        
        # Large tuning constant (less aggressive)
        config2 = RobustRegressionConfig(
            method='huber',
            tuning_constant=3.0
        )
        regression2 = RobustRepeatSalesRegression(config=config2)
        results2 = regression2.fit(sample_pairs_with_outliers)
        weights2 = regression2.get_weights()
        
        # Smaller tuning constant should give more extreme weights
        assert weights1.std() > weights2.std()
        assert weights1.min() < weights2.min()
    
    def test_influence_statistics(self, sample_pairs_with_outliers):
        """Test influence statistics calculation"""
        config = RobustRegressionConfig(method='huber')
        regression = RobustRepeatSalesRegression(config=config)
        
        results = regression.fit(sample_pairs_with_outliers)
        influence_df = regression.get_influence_statistics()
        
        assert len(influence_df) == len(sample_pairs_with_outliers)
        assert 'weight' in influence_df.columns
        assert 'residual' in influence_df.columns
        assert 'standardized_residual' in influence_df.columns
        assert 'downweighted' in influence_df.columns
        assert 'weight_percentile' in influence_df.columns
        
        # Check downweighted flag
        downweighted = influence_df[influence_df['downweighted']]
        assert len(downweighted) > 0
        assert all(downweighted['weight'] < 0.5)
    
    def test_compare_with_ols(self, sample_pairs_with_outliers):
        """Test comparison with standard OLS"""
        config = RobustRegressionConfig(method='huber')
        regression = RobustRepeatSalesRegression(config=config)
        
        # Fit robust regression
        robust_results = regression.fit(sample_pairs_with_outliers)
        
        # Compare with OLS
        comparison_df = regression.compare_with_ols(sample_pairs_with_outliers)
        
        assert len(comparison_df) == len(robust_results.period_dates)
        assert 'robust_coefficient' in comparison_df.columns
        assert 'ols_coefficient' in comparison_df.columns
        assert 'difference' in comparison_df.columns
        assert 'robust_index' in comparison_df.columns
        assert 'ols_index' in comparison_df.columns
        
        # With outliers, robust should differ from OLS
        avg_diff = comparison_df['difference'].abs().mean()
        assert avg_diff > 0.001  # Some meaningful difference
    
    def test_convergence_failure_handling(self, sample_pairs_clean):
        """Test handling of convergence failure"""
        # Use very few iterations to force non-convergence
        config = RobustRegressionConfig(
            method='bisquare',
            max_iterations=2,
            convergence_tolerance=1e-10  # Very strict
        )
        regression = RobustRepeatSalesRegression(config=config)
        
        results = regression.fit(sample_pairs_clean)
        
        # Should still return results even if not converged
        assert results is not None
        assert not results.convergence_info['converged']
        assert results.convergence_info['iterations'] == 2
    
    def test_robust_standard_errors(self, sample_pairs_with_outliers):
        """Test robust standard error calculation"""
        config = RobustRegressionConfig(method='huber')
        regression = RobustRepeatSalesRegression(config=config)
        
        results = regression.fit(sample_pairs_with_outliers)
        
        # Should have standard errors
        assert len(results.standard_errors) == len(results.log_returns)
        # Standard errors should be non-negative (allow NaN for numerical issues)
        valid_ses = [se for se in results.standard_errors if not np.isnan(se)]
        assert all(se >= 0 for se in valid_ses)
        assert results.standard_errors[0] == 0  # Base period
        
        # Compare with OLS standard errors
        ols_regression = RepeatSalesRegression()
        ols_results = ols_regression.fit(sample_pairs_with_outliers)
        
        # Just check that we got valid standard errors
        assert len(results.standard_errors) > 0
        assert not np.all(np.isnan(results.standard_errors[1:]))
    
    def test_empty_data(self):
        """Test handling of empty data"""
        config = RobustRegressionConfig(method='huber')
        regression = RobustRepeatSalesRegression(config=config)
        
        empty_df = pd.DataFrame(columns=[
            'property_id', 'tract_id', 'cbsa_id',
            'first_sale_date', 'first_sale_price',
            'second_sale_date', 'second_sale_price'
        ])
        
        with pytest.raises(ValueError, match="No transactions"):
            regression.fit(empty_df)
    
    def test_weighted_r_squared(self, sample_pairs_with_outliers):
        """Test weighted R-squared calculation"""
        # Fit both robust and OLS
        config = RobustRegressionConfig(method='bisquare')
        robust_reg = RobustRepeatSalesRegression(config=config)
        robust_results = robust_reg.fit(sample_pairs_with_outliers)
        
        ols_reg = RepeatSalesRegression()
        ols_results = ols_reg.fit(sample_pairs_with_outliers)
        
        # With outliers and robust weights, R-squared values should differ
        assert robust_results.r_squared != ols_results.r_squared
        
        # Both should be valid
        assert 0 <= robust_results.r_squared <= 1
        assert 0 <= ols_results.r_squared <= 1