"""Unit tests for data validators"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
import geopandas as gpd
from shapely.geometry import Point, Polygon

from hpi_fhfa.models.validators import DataValidator
from hpi_fhfa.models.geography import Supertract
from hpi_fhfa.models.weights import WeightSet, WeightType


class TestDataValidator:
    """Test data validation functions"""
    
    def test_validate_transaction_batch(self, sample_pairs_df):
        """Test batch validation of transactions"""
        validated_df = DataValidator.validate_transaction_batch(sample_pairs_df)
        
        # Check added columns
        assert 'is_valid' in validated_df.columns
        assert 'rejection_reason' in validated_df.columns
        assert 'valid_time_diff' in validated_df.columns
        assert 'valid_cagr' in validated_df.columns
        
        # Check some transactions are valid
        assert validated_df['is_valid'].sum() > 0
        
    def test_validate_extreme_cases(self):
        """Test validation of extreme transaction cases"""
        # Create problematic transactions
        data = [
            # Same month transaction
            {
                'property_id': 'P001',
                'tract_id': '12345678901',
                'cbsa_id': '10420',
                'first_sale_date': pd.Timestamp('2020-01-15'),
                'first_sale_price': 200000,
                'second_sale_date': pd.Timestamp('2020-06-15'),
                'second_sale_price': 210000
            },
            # Extreme CAGR
            {
                'property_id': 'P002',
                'tract_id': '12345678901',
                'cbsa_id': '10420',
                'first_sale_date': pd.Timestamp('2020-01-15'),
                'first_sale_price': 200000,
                'second_sale_date': pd.Timestamp('2022-01-15'),
                'second_sale_price': 400000  # 100% in 2 years
            },
            # Negative price
            {
                'property_id': 'P003',
                'tract_id': '12345678901',
                'cbsa_id': '10420',
                'first_sale_date': pd.Timestamp('2020-01-15'),
                'first_sale_price': -100000,
                'second_sale_date': pd.Timestamp('2022-01-15'),
                'second_sale_price': 200000
            }
        ]
        
        df = pd.DataFrame(data)
        validated = DataValidator.validate_transaction_batch(df)
        
        # Check specific rejections
        assert not validated.iloc[0]['is_valid']  # Same period
        assert 'Same 12-month period' in validated.iloc[0]['rejection_reason']
        
        assert not validated.iloc[1]['is_valid']  # High CAGR
        assert 'CAGR' in validated.iloc[1]['rejection_reason']
        
        assert not validated.iloc[2]['is_valid']  # Negative price
        assert 'Invalid prices' in validated.iloc[2]['rejection_reason']
    
    def test_validate_tract_data(self):
        """Test tract data validation"""
        # Create valid tract data
        coords1 = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
        coords2 = [(1, 0), (2, 0), (2, 1), (1, 1), (1, 0)]
        
        data = [
            {
                'tract_id': '12345678901',
                'cbsa_id': '10420',
                'state': '12',
                'county': '345',
                'geometry': Polygon(coords1)
            },
            {
                'tract_id': '12345678902',
                'cbsa_id': '10420',
                'state': '12',
                'county': '345',
                'geometry': Polygon(coords2)
            }
        ]
        
        gdf = gpd.GeoDataFrame(data, geometry='geometry')
        gdf.set_index('tract_id', inplace=True)
        
        is_valid, errors = DataValidator.validate_tract_data(gdf)
        if not is_valid:
            print(f"Validation errors: {errors}")
        assert is_valid
        assert len(errors) == 0
        
        # Test with missing columns
        bad_gdf = gdf.drop(columns=['state'])
        is_valid, errors = DataValidator.validate_tract_data(bad_gdf)
        assert not is_valid
        assert any('Missing columns' in e for e in errors)
        
        # Test with invalid geometry
        bad_coords = [(0, 0), (1, 1), (0, 1), (1, 0), (0, 0)]  # Self-intersecting
        bad_gdf = gdf.copy()
        bad_gdf.loc['12345678901', 'geometry'] = Polygon(bad_coords)
        is_valid, errors = DataValidator.validate_tract_data(bad_gdf)
        assert not is_valid
        assert any('Invalid geometries' in e for e in errors)
    
    def test_validate_supertract(self):
        """Test supertract validation"""
        # Valid supertract
        year_pairs_count = {
            '12345678901': 20,
            '12345678902': 25
        }
        
        supertract = Supertract(
            supertract_id='super_12345678901',
            component_tract_ids=['12345678901', '12345678902'],
            year=2020,
            half_pairs_count=45
        )
        
        is_valid, error = DataValidator.validate_supertract(supertract, year_pairs_count)
        assert is_valid
        assert error is None
        
        # Insufficient half-pairs
        low_supertract = Supertract(
            supertract_id='super_12345678903',
            component_tract_ids=['12345678903'],
            year=2020,
            half_pairs_count=30
        )
        
        is_valid, error = DataValidator.validate_supertract(
            low_supertract, 
            {'12345678903': 30}
        )
        assert not is_valid
        assert 'Insufficient half-pairs' in error
        
        # Missing component tract
        is_valid, error = DataValidator.validate_supertract(supertract, {})
        assert not is_valid
        assert 'not found' in error
    
    def test_validate_weights(self):
        """Test weight validation"""
        # Valid normalized weights
        weights = WeightSet(
            geography_id='10420',
            period=2020,
            weight_type=WeightType.SAMPLE,
            weights={
                'tract1': 0.3,
                'tract2': 0.5,
                'tract3': 0.2
            }
        )
        
        is_valid, error = DataValidator.validate_weights(weights)
        assert is_valid
        assert error is None
        
        # Empty weights
        empty_weights = WeightSet(
            geography_id='10420',
            period=2020,
            weight_type=WeightType.SAMPLE,
            weights={}
        )
        
        is_valid, error = DataValidator.validate_weights(empty_weights)
        assert not is_valid
        assert 'Empty weight set' in error
        
        # Negative weights
        negative_weights = WeightSet(
            geography_id='10420',
            period=2020,
            weight_type=WeightType.SAMPLE,
            weights={
                'tract1': 0.5,
                'tract2': -0.2,
                'tract3': 0.7
            }
        )
        
        is_valid, error = DataValidator.validate_weights(negative_weights)
        assert not is_valid
        assert 'Negative weights' in error
        
        # Non-normalized weights
        non_norm_weights = WeightSet(
            geography_id='10420',
            period=2020,
            weight_type=WeightType.SAMPLE,
            weights={
                'tract1': 0.3,
                'tract2': 0.5,
                'tract3': 0.3  # Sum = 1.1
            }
        )
        
        is_valid, error = DataValidator.validate_weights(non_norm_weights)
        assert not is_valid
        assert 'not normalized' in error
    
    def test_generate_validation_report(self, sample_pairs_df):
        """Test validation report generation"""
        report = DataValidator.generate_validation_report(sample_pairs_df)
        
        # Check report structure
        assert 'total_transactions' in report
        assert 'valid_transactions' in report
        assert 'invalid_transactions' in report
        assert 'validation_rate' in report
        assert 'rejection_reasons' in report
        assert 'cagr_stats' in report
        assert 'time_diff_stats' in report
        
        # Check consistency
        assert report['total_transactions'] == len(sample_pairs_df)
        assert report['valid_transactions'] + report['invalid_transactions'] == report['total_transactions']
        assert 0 <= report['validation_rate'] <= 1
        
        # Check statistics
        assert 'mean' in report['cagr_stats']
        assert 'std' in report['cagr_stats']
        assert 'min' in report['cagr_stats']
        assert 'max' in report['cagr_stats']