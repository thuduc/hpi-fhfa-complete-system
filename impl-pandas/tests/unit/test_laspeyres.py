"""Unit tests for Laspeyres index calculation"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from hpi_fhfa.weighting import LaspeyresIndex
from hpi_fhfa.models.weights import WeightSet, WeightType
from hpi_fhfa.algorithms.regression import RegressionResults


class TestLaspeyresIndex:
    """Test Laspeyres value-weighted index implementation"""
    
    @pytest.fixture
    def sample_tract_indices(self):
        """Create sample tract index data"""
        base_date = date(2020, 1, 1)
        dates = [base_date + timedelta(days=30*i) for i in range(12)]
        
        # Three tracts with different price trends
        indices = {
            'T001': pd.DataFrame({
                'date': dates,
                'index_value': 100 * (1 + 0.005 * np.arange(12))  # 0.5% monthly growth
            }),
            'T002': pd.DataFrame({
                'date': dates,
                'index_value': 100 * (1 + 0.01 * np.arange(12))  # 1% monthly growth
            }),
            'T003': pd.DataFrame({
                'date': dates,
                'index_value': 100 * (1 - 0.002 * np.arange(12))  # -0.2% monthly decline
            })
        }
        return indices
    
    @pytest.fixture
    def sample_weights(self):
        """Create sample weight sets"""
        weights = {
            'C1_2020': WeightSet(
                geography_id='C1',
                period=2020,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.3,
                    'T002': 0.5,
                    'T003': 0.2
                }
            ),
            'C1_2021': WeightSet(
                geography_id='C1',
                period=2021,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.25,
                    'T002': 0.55,
                    'T003': 0.20
                }
            )
        }
        return weights
    
    @pytest.fixture
    def sample_regression_results(self):
        """Create sample regression results"""
        base_date = date(2020, 1, 1)
        dates = [base_date + timedelta(days=30*i) for i in range(12)]
        
        # Create regression results for two tracts
        results = {
            'T001': RegressionResults(
                log_returns=np.log([1.0, 1.005, 1.010, 1.015, 1.020, 1.025,
                                   1.030, 1.035, 1.040, 1.045, 1.050, 1.055]),
                standard_errors=np.ones(12) * 0.001,
                residuals=np.random.normal(0, 0.01, 100),
                r_squared=0.95,
                num_observations=100,
                num_periods=12,
                period_dates=dates,
                convergence_info={'converged': True, 'iterations': 10}
            ),
            'T002': RegressionResults(
                log_returns=np.log([1.0, 1.01, 1.02, 1.03, 1.04, 1.05,
                                   1.06, 1.07, 1.08, 1.09, 1.10, 1.11]),
                standard_errors=np.ones(12) * 0.002,
                residuals=np.random.normal(0, 0.01, 150),
                r_squared=0.92,
                num_observations=150,
                num_periods=12,
                period_dates=dates,
                convergence_info={'converged': True, 'iterations': 8}
            )
        }
        return results
    
    def test_basic_calculation(self, sample_tract_indices, sample_weights):
        """Test basic Laspeyres index calculation"""
        laspeyres = LaspeyresIndex()
        
        result = laspeyres.calculate_index(
            tract_indices=sample_tract_indices,
            weights=sample_weights,
            geography_id='C1'
        )
        
        assert result is not None
        assert len(result.index_values) == 12  # Monthly data
        assert result.coverage_rate == 1.0  # All tracts present
        assert len(result.missing_tracts) == 0
        
        # Check index values
        df = result.index_values
        assert 'date' in df.columns
        assert 'index_value' in df.columns
        assert 'coverage' in df.columns
        
        # First period should be 100
        assert df.iloc[0]['index_value'] == 100.0
        
        # Index should generally increase (weighted average of increasing tracts)
        assert df.iloc[-1]['index_value'] > df.iloc[0]['index_value']
    
    def test_with_base_period(self, sample_tract_indices, sample_weights):
        """Test index calculation with specific base period"""
        base_date = date(2020, 3, 1)
        laspeyres = LaspeyresIndex(base_period=base_date)
        
        result = laspeyres.calculate_index(
            tract_indices=sample_tract_indices,
            weights=sample_weights,
            geography_id='C1'
        )
        
        assert result is not None
        assert result.base_period_weights.period == 2020
    
    def test_missing_tracts(self, sample_tract_indices, sample_weights):
        """Test handling of missing tract data"""
        # Remove one tract from indices
        incomplete_indices = {
            'T001': sample_tract_indices['T001'],
            'T002': sample_tract_indices['T002']
            # T003 missing
        }
        
        laspeyres = LaspeyresIndex()
        result = laspeyres.calculate_index(
            tract_indices=incomplete_indices,
            weights=sample_weights,
            geography_id='C1'
        )
        
        assert result is not None
        assert result.coverage_rate < 1.0
        assert 'T003' in result.missing_tracts
        
        # Coverage should be 0.8 (weights of T001 + T002)
        assert abs(result.coverage_rate - 0.8) < 1e-6
    
    def test_from_regression_results(self, sample_regression_results, sample_weights):
        """Test calculation from regression results"""
        laspeyres = LaspeyresIndex()
        
        result = laspeyres.calculate_from_regression_results(
            regression_results=sample_regression_results,
            weights=sample_weights,
            geography_id='C1',
            base_value=100.0
        )
        
        assert result is not None
        assert len(result.index_values) == 12
        
        # Check that index values match expected log returns
        df = result.index_values
        first_value = df.iloc[0]['index_value']
        assert abs(first_value - 100.0) < 1e-6
    
    def test_date_filtering(self, sample_tract_indices, sample_weights):
        """Test date range filtering"""
        laspeyres = LaspeyresIndex()
        
        start_date = date(2020, 4, 1)
        end_date = date(2020, 9, 30)
        
        result = laspeyres.calculate_index(
            tract_indices=sample_tract_indices,
            weights=sample_weights,
            geography_id='C1',
            start_date=start_date,
            end_date=end_date
        )
        
        assert result is not None
        assert len(result.index_values) == 6  # April through September
        
        # Check date range
        df = result.index_values
        assert df['date'].min() >= start_date
        assert df['date'].max() <= end_date
    
    def test_supertract_handling(self, sample_tract_indices, sample_weights):
        """Test handling of supertracts"""
        # Add a supertract that contains T003
        supertract_indices = sample_tract_indices.copy()
        supertract_indices['super_T003'] = sample_tract_indices['T003'].copy()
        del supertract_indices['T003']  # Remove original tract
        
        laspeyres = LaspeyresIndex()
        result = laspeyres.calculate_index(
            tract_indices=supertract_indices,
            weights=sample_weights,
            geography_id='C1'
        )
        
        # Should still find T003 data in supertract
        assert result.coverage_rate == 1.0
        assert len(result.missing_tracts) == 0
    
    def test_weight_type_warning(self, sample_tract_indices, sample_weights, capsys):
        """Test warning for non-VALUE weight type"""
        # Change weight type
        sample_weights['C1_2020'].weight_type = WeightType.SAMPLE
        
        laspeyres = LaspeyresIndex()
        result = laspeyres.calculate_index(
            tract_indices=sample_tract_indices,
            weights=sample_weights,
            geography_id='C1'
        )
        
        # Check for warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert "SAMPLE" in captured.out
    
    def test_empty_data(self, sample_weights):
        """Test handling of empty data"""
        laspeyres = LaspeyresIndex()
        
        with pytest.raises(ValueError, match="No valid dates"):
            laspeyres.calculate_index(
                tract_indices={},
                weights=sample_weights,
                geography_id='C1'
            )
    
    def test_no_weights(self, sample_tract_indices):
        """Test handling of missing weights"""
        laspeyres = LaspeyresIndex()
        
        with pytest.raises(ValueError, match="No weights available"):
            laspeyres.calculate_index(
                tract_indices=sample_tract_indices,
                weights={},
                geography_id='C1'
            )
    
    def test_chain_index(self):
        """Test chaining of yearly indices"""
        # Create yearly indices
        yearly_indices = {
            2020: pd.DataFrame({
                'date': pd.date_range('2020-01-01', '2020-12-01', freq='MS'),
                'index_value': 100 * (1 + 0.01 * np.arange(12))
            }),
            2021: pd.DataFrame({
                'date': pd.date_range('2021-01-01', '2021-12-01', freq='MS'),
                'index_value': 100 * (1 + 0.015 * np.arange(12))
            })
        }
        
        # Create yearly weights (not used in chaining but required)
        yearly_weights = {
            2020: WeightSet('C1', 2020, WeightType.VALUE, {'T1': 1.0}),
            2021: WeightSet('C1', 2021, WeightType.VALUE, {'T1': 1.0})
        }
        
        laspeyres = LaspeyresIndex()
        chained = laspeyres.chain_index(yearly_indices, yearly_weights)
        
        assert len(chained) == 24  # Two years of monthly data
        assert 'chained_index' in chained.columns
        
        # Check continuity at year boundary
        dec_2020 = chained[chained['date'] == '2020-12-01']['chained_index'].iloc[0]
        jan_2021_base = chained[chained['date'] == '2021-01-01']['index_value'].iloc[0]
        jan_2021_chained = chained[chained['date'] == '2021-01-01']['chained_index'].iloc[0]
        
        # Chained value should be base value scaled by Dec 2020 factor
        expected = jan_2021_base * (dec_2020 / 100)
        assert abs(jan_2021_chained - expected) < 1e-6
    
    def test_weighted_aggregation_accuracy(self):
        """Test accuracy of weighted aggregation"""
        # Create simple test case with known answer
        indices = {
            'T1': pd.DataFrame({
                'date': [date(2020, 1, 1)],
                'index_value': [110.0]
            }),
            'T2': pd.DataFrame({
                'date': [date(2020, 1, 1)],
                'index_value': [120.0]
            })
        }
        
        weights = {
            'test_2020': WeightSet(
                geography_id='test',
                period=2020,
                weight_type=WeightType.VALUE,
                weights={'T1': 0.4, 'T2': 0.6}
            )
        }
        
        laspeyres = LaspeyresIndex()
        result = laspeyres.calculate_index(
            tract_indices=indices,
            weights=weights,
            geography_id='test'
        )
        
        # Expected: 0.4 * 110 + 0.6 * 120 = 44 + 72 = 116
        expected_value = 0.4 * 110 + 0.6 * 120
        actual_value = result.index_values.iloc[0]['index_value']
        assert abs(actual_value - expected_value) < 1e-6