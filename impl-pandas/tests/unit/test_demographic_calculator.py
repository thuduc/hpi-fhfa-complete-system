"""Unit tests for demographic weight calculator"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from hpi_fhfa.weighting import DemographicWeightCalculator
from hpi_fhfa.models.weights import WeightType, WeightSet, DemographicData


class TestDemographicWeightCalculator:
    """Test demographic weight calculator functionality"""
    
    @pytest.fixture
    def sample_demographic_data(self):
        """Create sample demographic data"""
        data = pd.DataFrame([
            {
                'tract_id': 'T001',
                'year': 2020,
                'cbsa_id': 'C1',
                'housing_units': 1000,
                'median_value': 250000,
                'college_share': 0.3,
                'non_white_share': 0.4,
                'upb_total': 150000000
            },
            {
                'tract_id': 'T002',
                'year': 2020,
                'cbsa_id': 'C1',
                'housing_units': 2000,
                'median_value': 300000,
                'college_share': 0.5,
                'non_white_share': 0.2,
                'upb_total': 400000000
            },
            {
                'tract_id': 'T003',
                'year': 2020,
                'cbsa_id': 'C2',
                'housing_units': 1500,
                'median_value': 200000,
                'college_share': 0.2,
                'non_white_share': 0.6,
                'upb_total': 200000000
            }
        ])
        return data
    
    def test_initialization(self, sample_demographic_data):
        """Test calculator initialization"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        assert calc.demographic_data is not None
        assert len(calc._data_by_year_geo) > 0
        
        # Check data organization
        assert (2020, 'C1') in calc._data_by_year_geo
        assert (2020, 'C2') in calc._data_by_year_geo
        assert (2020, 'national') in calc._data_by_year_geo
    
    def test_sample_weights(self, sample_demographic_data):
        """Test sample weight calculation (equal weights)"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.SAMPLE, 'C1', 2020)
        
        assert weights is not None
        assert weights.weight_type == WeightType.SAMPLE
        assert len(weights.weights) == 2  # Two tracts in C1
        assert weights.is_normalized
        
        # Equal weights
        for weight in weights.weights.values():
            assert abs(weight - 0.5) < 1e-6
    
    def test_value_weights(self, sample_demographic_data):
        """Test value weight calculation"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.VALUE, 'C1', 2020)
        
        assert weights is not None
        assert weights.weight_type == WeightType.VALUE
        assert weights.is_normalized
        
        # T002 should have higher weight (2000 * 300000 vs 1000 * 250000)
        assert weights.weights['T002'] > weights.weights['T001']
        
        # Check specific values
        total_value = 1000 * 250000 + 2000 * 300000
        expected_t001 = (1000 * 250000) / total_value
        expected_t002 = (2000 * 300000) / total_value
        
        assert abs(weights.weights['T001'] - expected_t001) < 1e-6
        assert abs(weights.weights['T002'] - expected_t002) < 1e-6
    
    def test_unit_weights(self, sample_demographic_data):
        """Test unit weight calculation"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.UNIT, 'C1', 2020)
        
        assert weights is not None
        assert weights.weight_type == WeightType.UNIT
        assert weights.is_normalized
        
        # Weights based on housing units
        total_units = 1000 + 2000
        assert abs(weights.weights['T001'] - 1000/total_units) < 1e-6
        assert abs(weights.weights['T002'] - 2000/total_units) < 1e-6
    
    def test_upb_weights(self, sample_demographic_data):
        """Test UPB weight calculation"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.UPB, 'C1', 2020)
        
        assert weights is not None
        assert weights.weight_type == WeightType.UPB
        assert weights.is_normalized
        
        # T002 has higher UPB
        assert weights.weights['T002'] > weights.weights['T001']
    
    def test_college_weights(self, sample_demographic_data):
        """Test college share weight calculation"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.COLLEGE, 'C1', 2020)
        
        assert weights is not None
        assert weights.weight_type == WeightType.COLLEGE
        assert weights.is_normalized
        
        # T002 has higher college share (0.5 vs 0.3)
        assert weights.weights['T002'] > weights.weights['T001']
        
        # Check proportions
        total_share = 0.3 + 0.5
        assert abs(weights.weights['T001'] - 0.3/total_share) < 1e-6
        assert abs(weights.weights['T002'] - 0.5/total_share) < 1e-6
    
    def test_non_white_weights(self, sample_demographic_data):
        """Test non-white share weight calculation"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.NON_WHITE, 'C1', 2020)
        
        assert weights is not None
        assert weights.weight_type == WeightType.NON_WHITE
        assert weights.is_normalized
        
        # T001 has higher non-white share (0.4 vs 0.2)
        assert weights.weights['T001'] > weights.weights['T002']
    
    def test_national_weights(self, sample_demographic_data):
        """Test national level weight calculation"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.VALUE, 'national', 2020)
        
        assert weights is not None
        assert len(weights.weights) == 3  # All three tracts
        assert weights.is_normalized
        assert weights.geography_id == 'national'
    
    def test_missing_data(self):
        """Test handling of missing data"""
        calc = DemographicWeightCalculator()
        
        weights = calc.calculate_weights(WeightType.SAMPLE, 'C1', 2020)
        assert weights is None
        
        # With empty data
        calc = DemographicWeightCalculator(pd.DataFrame())
        weights = calc.calculate_weights(WeightType.SAMPLE, 'C1', 2020)
        assert weights is None
    
    def test_add_demographic_data(self, sample_demographic_data):
        """Test adding new demographic data"""
        calc = DemographicWeightCalculator()
        
        # Initially no data
        assert calc.demographic_data is None
        
        # Add data
        calc.add_demographic_data(sample_demographic_data)
        
        # Now should have data
        weights = calc.calculate_weights(WeightType.SAMPLE, 'C1', 2020)
        assert weights is not None
        
        # Add more data
        new_data = pd.DataFrame([{
            'tract_id': 'T004',
            'year': 2021,
            'cbsa_id': 'C1',
            'housing_units': 1200,
            'median_value': 280000,
            'college_share': 0.4,
            'non_white_share': 0.3,
            'upb_total': 250000000
        }])
        calc.add_demographic_data(new_data)
        
        # Should have both years
        assert 2020 in calc.get_available_periods('C1')
        assert 2021 in calc.get_available_periods('C1')
    
    def test_available_periods(self, sample_demographic_data):
        """Test getting available periods"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        periods = calc.get_available_periods('C1')
        assert periods == [2020]
        
        periods = calc.get_available_periods('national')
        assert periods == [2020]
        
        periods = calc.get_available_periods('unknown')
        assert periods == []
    
    def test_available_geographies(self, sample_demographic_data):
        """Test getting available geographies"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        geos = calc.get_available_geographies(2020)
        assert 'C1' in geos
        assert 'C2' in geos
        assert 'national' in geos
        
        geos = calc.get_available_geographies(2021)
        assert len(geos) == 0
    
    def test_validate_weights(self, sample_demographic_data):
        """Test weight validation"""
        calc = DemographicWeightCalculator(sample_demographic_data)
        
        weights = calc.calculate_weights(WeightType.SAMPLE, 'C1', 2020)
        assert calc.validate_weights(weights)
        
        # Create invalid weights
        invalid_weights = WeightSet(
            geography_id='C1',
            period=2020,
            weight_type=WeightType.SAMPLE,
            weights={'T001': 0.6, 'T002': 0.6}  # Not normalized
        )
        assert not calc.validate_weights(invalid_weights)
        
        # Negative weights
        invalid_weights2 = WeightSet(
            geography_id='C1',
            period=2020,
            weight_type=WeightType.SAMPLE,
            weights={'T001': -0.5, 'T002': 1.5}
        )
        assert not calc.validate_weights(invalid_weights2)
    
    def test_synthetic_data_generation(self, sample_tract_gdf):
        """Test synthetic demographic data generation"""
        calc = DemographicWeightCalculator()
        
        years = [2018, 2019, 2020]
        synthetic_data = calc.generate_synthetic_demographic_data(
            sample_tract_gdf, years, seed=42
        )
        
        assert len(synthetic_data) == len(sample_tract_gdf) * len(years)
        assert all(col in synthetic_data.columns for col in [
            'tract_id', 'year', 'cbsa_id', 'housing_units',
            'median_value', 'college_share', 'non_white_share', 'upb_total'
        ])
        
        # Check value ranges
        assert (synthetic_data['housing_units'] >= 500).all()
        assert (synthetic_data['housing_units'] <= 5000).all()
        assert (synthetic_data['college_share'] >= 0).all()
        assert (synthetic_data['college_share'] <= 1).all()
        assert (synthetic_data['non_white_share'] >= 0).all()
        assert (synthetic_data['non_white_share'] <= 1).all()
    
    def test_demographic_data_class(self):
        """Test DemographicData class methods"""
        demo = DemographicData(
            tract_id='T001',
            year=2020,
            housing_units=1000,
            median_value=250000,
            college_share=0.3,
            non_white_share=0.4,
            upb_total=150000000
        )
        
        # Test weight value calculations
        assert demo.get_weight_value(WeightType.VALUE) == 1000 * 250000
        assert demo.get_weight_value(WeightType.UNIT) == 1000
        assert demo.get_weight_value(WeightType.UPB) == 150000000
        assert demo.get_weight_value(WeightType.COLLEGE) == 0.3
        assert demo.get_weight_value(WeightType.NON_WHITE) == 0.4
        
        # Test invalid weight type
        with pytest.raises(ValueError):
            demo.get_weight_value(WeightType.SAMPLE)