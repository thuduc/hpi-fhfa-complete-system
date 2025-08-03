"""Unit tests for sensitivity analysis"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
import warnings

from hpi_fhfa.outliers import SensitivityAnalyzer, SensitivityResult
from hpi_fhfa.models.weights import WeightType


class TestSensitivityAnalyzer:
    """Test sensitivity analysis functionality"""
    
    @pytest.fixture
    def sample_pairs_for_sensitivity(self):
        """Create sample data for sensitivity testing"""
        np.random.seed(42)
        data = []
        base_date = date(2015, 1, 1)
        
        # Generate varied data
        for i in range(200):
            first_date = base_date + timedelta(days=np.random.randint(0, 1825))
            second_date = first_date + timedelta(days=np.random.randint(365, 1825))
            
            # Mix of normal and extreme cases
            if i < 180:
                # Normal transactions
                first_price = np.random.uniform(100000, 500000)
                annual_appreciation = np.random.normal(0.05, 0.03)
                years = (second_date - first_date).days / 365.25
                second_price = first_price * (1 + annual_appreciation) ** years
            else:
                # Some outliers
                first_price = np.random.uniform(50000, 1000000)
                if i % 2 == 0:
                    second_price = first_price * np.random.uniform(2.0, 3.0)  # High appreciation
                else:
                    second_price = first_price * np.random.uniform(0.3, 0.5)  # Big loss
            
            data.append({
                'property_id': f'P{i:04d}',
                'tract_id': f'T{i % 20:03d}',
                'cbsa_id': f'C{i % 5:02d}',
                'first_sale_date': first_date,
                'first_sale_price': first_price,
                'second_sale_date': second_date,
                'second_sale_price': second_price
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_tract_gdf(self):
        """Create minimal tract GeoDataFrame"""
        data = []
        for i in range(20):
            data.append({
                'tract_id': f'T{i:03d}',
                'cbsa_id': f'C{i % 5:02d}'
            })
        return pd.DataFrame(data).set_index('tract_id')
    
    @pytest.fixture
    def custom_scenarios(self):
        """Create custom test scenarios"""
        return [
            {
                'name': 'test_outlier_threshold',
                'type': 'outlier_threshold',
                'params': {'residual_threshold': 2.5}
            },
            {
                'name': 'test_robust_method',
                'type': 'robust_method',
                'params': {'method': 'huber'}
            },
            {
                'name': 'test_time_window',
                'type': 'time_window',
                'params': {'years_back': 3}
            }
        ]
    
    def test_basic_sensitivity_analysis(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test basic sensitivity analysis"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        # Use minimal scenarios for speed
        scenarios = [
            {
                'name': 'outlier_3std',
                'type': 'outlier_threshold',
                'params': {'residual_threshold': 3.0}
            },
            {
                'name': 'robust_huber',
                'type': 'robust_method',
                'params': {'method': 'huber'}
            }
        ]
        
        result = analyzer.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=scenarios
        )
        
        assert isinstance(result, SensitivityResult)
        assert isinstance(result.base_index, pd.DataFrame)
        assert isinstance(result.sensitivity_indices, dict)
        assert len(result.sensitivity_indices) == 2
        assert isinstance(result.impact_metrics, pd.DataFrame)
        assert isinstance(result.parameter_importance, pd.DataFrame)
        assert isinstance(result.recommendations, list)
    
    def test_default_scenarios(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test with default scenarios"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        # Get default scenarios
        scenarios = analyzer._get_default_scenarios()
        
        # Should have multiple scenario types
        scenario_types = {s['type'] for s in scenarios}
        assert 'outlier_threshold' in scenario_types
        assert 'robust_method' in scenario_types
        assert 'weight_type' in scenario_types
        assert 'time_window' in scenario_types
        assert 'data_filter' in scenario_types
        
        # Should have reasonable number of scenarios
        assert 10 <= len(scenarios) <= 20
    
    def test_parallel_execution(self, sample_pairs_for_sensitivity, sample_tract_gdf, custom_scenarios):
        """Test parallel vs sequential execution"""
        # Sequential
        analyzer_seq = SensitivityAnalyzer(parallel=False)
        result_seq = analyzer_seq.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=custom_scenarios
        )
        
        # Parallel
        analyzer_par = SensitivityAnalyzer(parallel=True, max_workers=2)
        result_par = analyzer_par.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=custom_scenarios
        )
        
        # Results should be similar (may have minor numerical differences)
        assert len(result_seq.sensitivity_indices) == len(result_par.sensitivity_indices)
        assert set(result_seq.sensitivity_indices.keys()) == set(result_par.sensitivity_indices.keys())
    
    def test_scenario_execution(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test individual scenario types"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        # Test outlier threshold scenario
        outlier_scenario = {
            'name': 'outlier_test',
            'type': 'outlier_threshold',
            'params': {'residual_threshold': 2.0}
        }
        index_df = analyzer._run_single_scenario(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            outlier_scenario,
            WeightType.VALUE
        )
        assert len(index_df) > 0
        assert 'index_value' in index_df.columns
        
        # Test robust method scenario
        robust_scenario = {
            'name': 'robust_test',
            'type': 'robust_method',
            'params': {'method': 'bisquare'}
        }
        index_df = analyzer._run_single_scenario(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            robust_scenario,
            WeightType.VALUE
        )
        assert len(index_df) > 0
        
        # Test time window scenario
        time_scenario = {
            'name': 'time_test',
            'type': 'time_window',
            'params': {'years_back': 2}
        }
        index_df = analyzer._run_single_scenario(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            time_scenario,
            WeightType.VALUE
        )
        assert len(index_df) > 0
        
        # Test data filter scenario
        filter_scenario = {
            'name': 'filter_test',
            'type': 'data_filter',
            'params': {'max_cagr': 0.2, 'min_price': 75000}
        }
        index_df = analyzer._run_single_scenario(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            filter_scenario,
            WeightType.VALUE
        )
        assert len(index_df) > 0
    
    def test_impact_metrics_calculation(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test impact metrics calculation"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        scenarios = [
            {
                'name': 'high_impact',
                'type': 'data_filter',
                'params': {'max_cagr': 0.1}  # Very restrictive
            },
            {
                'name': 'low_impact',
                'type': 'outlier_threshold',
                'params': {'residual_threshold': 5.0}  # Very loose
            }
        ]
        
        result = analyzer.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=scenarios
        )
        
        metrics = result.impact_metrics
        assert len(metrics) == 2
        assert 'mean_difference' in metrics.columns
        assert 'max_difference' in metrics.columns
        assert 'mean_pct_difference' in metrics.columns
        assert 'max_pct_difference' in metrics.columns
        assert 'rmse' in metrics.columns
        assert 'correlation' in metrics.columns
        
        # High impact scenario should have larger differences
        high_impact_row = metrics[metrics['scenario'] == 'high_impact']
        low_impact_row = metrics[metrics['scenario'] == 'low_impact']
        
        if len(high_impact_row) > 0 and len(low_impact_row) > 0:
            assert high_impact_row['max_pct_difference'].iloc[0] > low_impact_row['max_pct_difference'].iloc[0]
    
    def test_parameter_importance_ranking(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test parameter importance ranking"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        # Use scenarios with different types
        scenarios = [
            {'name': 'outlier1', 'type': 'outlier_threshold', 'params': {'residual_threshold': 2.0}},
            {'name': 'outlier2', 'type': 'outlier_threshold', 'params': {'residual_threshold': 4.0}},
            {'name': 'robust1', 'type': 'robust_method', 'params': {'method': 'huber'}},
            {'name': 'filter1', 'type': 'data_filter', 'params': {'max_cagr': 0.15}}
        ]
        
        result = analyzer.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=scenarios
        )
        
        importance = result.parameter_importance
        assert len(importance) > 0
        assert 'parameter_type' in importance.columns
        assert 'avg_impact' in importance.columns
        assert 'max_impact' in importance.columns
        
        # Should be sorted by average impact
        assert importance['avg_impact'].is_monotonic_decreasing
    
    def test_recommendations_generation(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test recommendation generation"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        # Create scenarios with high sensitivity
        scenarios = [
            {
                'name': 'extreme_filter',
                'type': 'data_filter',
                'params': {'max_cagr': 0.05}  # Very restrictive
            },
            {
                'name': 'short_window',
                'type': 'time_window',
                'params': {'years_back': 1}
            }
        ]
        
        result = analyzer.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=scenarios
        )
        
        recommendations = result.recommendations
        assert len(recommendations) > 0
        assert isinstance(recommendations[0], str)
        
        # Should mention sensitivity level
        assert any('sensitivity' in rec.lower() for rec in recommendations)
    
    def test_failed_scenario_handling(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test handling of failed scenarios"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        # Include an invalid scenario
        scenarios = [
            {
                'name': 'valid_scenario',
                'type': 'outlier_threshold',
                'params': {'residual_threshold': 3.0}
            },
            {
                'name': 'invalid_scenario',
                'type': 'unknown_type',  # Invalid type
                'params': {}
            }
        ]
        
        # Should warn but continue
        with warnings.catch_warnings(record=True) as w:
            result = analyzer.analyze_sensitivity(
                sample_pairs_for_sensitivity,
                sample_tract_gdf,
                scenarios=scenarios
            )
            
            # Should have warning about failed scenario
            assert any('failed' in str(warning.message).lower() for warning in w)
        
        # Should still have results from valid scenario
        assert 'valid_scenario' in result.sensitivity_indices
        assert 'invalid_scenario' not in result.sensitivity_indices
    
    def test_visualization_data_creation(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test creation of visualization data"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        scenarios = [
            {
                'name': 'test_scenario',
                'type': 'outlier_threshold',
                'params': {'residual_threshold': 3.0}
            }
        ]
        
        result = analyzer.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=scenarios
        )
        
        plot_data = analyzer.create_sensitivity_plots_data(result)
        
        assert 'time_series' in plot_data
        assert 'impact_summary' in plot_data
        assert 'parameter_importance' in plot_data
        
        # Time series should have all scenarios
        time_series = plot_data['time_series']
        scenarios_in_data = time_series['scenario'].unique()
        assert 'base_case' in scenarios_in_data
        assert 'test_scenario' in scenarios_in_data
    
    def test_sensitivity_report_creation(self, sample_pairs_for_sensitivity, sample_tract_gdf):
        """Test text report creation"""
        analyzer = SensitivityAnalyzer(parallel=False)
        
        scenarios = [
            {
                'name': 'test_scenario',
                'type': 'outlier_threshold',
                'params': {'residual_threshold': 2.5}
            }
        ]
        
        result = analyzer.analyze_sensitivity(
            sample_pairs_for_sensitivity,
            sample_tract_gdf,
            scenarios=scenarios
        )
        
        report = analyzer.create_sensitivity_report(result)
        
        assert isinstance(report, str)
        assert 'SENSITIVITY ANALYSIS REPORT' in report
        assert 'SENSITIVITY SUMMARY' in report
        assert 'PARAMETER IMPORTANCE' in report
        assert 'RECOMMENDATIONS' in report