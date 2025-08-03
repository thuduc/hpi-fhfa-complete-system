"""Integration tests for Phase 3 algorithms"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from hpi_fhfa.algorithms import (
    RepeatSalesRegression,
    SupertractConstructor,
    BMNIndexEstimator
)
from hpi_fhfa.data import SyntheticDataGenerator, GeographicDataGenerator
from hpi_fhfa.models.validators import DataValidator
from hpi_fhfa.models.weights import SampleWeightCalculator, WeightType


class TestPhase3Integration:
    """Integration tests for complete Phase 3 workflow"""
    
    def test_full_index_calculation_pipeline(self):
        """Test complete index calculation pipeline"""
        # Generate test data
        syn_gen = SyntheticDataGenerator(seed=42)
        geo_gen = GeographicDataGenerator(seed=42)
        
        # Generate smaller dataset for testing
        transaction_data = syn_gen.generate_complete_dataset(
            start_year=2018,
            end_year=2020,
            num_cbsas=3,
            num_tracts=30,
            num_properties=300,
            target_pairs=150
        )
        
        geo_data = geo_gen.generate_complete_geographic_data(
            num_tracts=30
        )
        
        # Validate transaction pairs
        validator = DataValidator()
        validated_df = validator.validate_transaction_batch(
            transaction_data['pairs']
        )
        valid_pairs = validated_df[validated_df['is_valid']]
        
        # Create weight calculator
        weight_calculator = SampleWeightCalculator()
        
        # Initialize estimator
        estimator = BMNIndexEstimator(
            min_half_pairs=20,
            weight_calculator=weight_calculator,
            adjacency_data=geo_data['adjacency']
        )
        
        # Estimate indices
        results = estimator.estimate_indices(
            valid_pairs,
            geo_data['tracts'],
            WeightType.SAMPLE,
            geography_level='cbsa',
            start_date=date(2018, 1, 1),
            end_date=date(2020, 12, 31)
        )
        
        # Verify results
        assert len(results.index_values) > 0
        assert len(results.regression_results) > 0
        assert results.coverage_stats['coverage_rate'] > 0.5
        
        # Check index continuity
        for cbsa_id in results.index_values['geography_id'].unique():
            cbsa_indices = results.index_values[
                results.index_values['geography_id'] == cbsa_id
            ].sort_values('date')
            
            # Should have continuous monthly data
            assert len(cbsa_indices) >= 24  # At least 2 years of monthly data
            
            # Index values should be reasonable
            assert all(cbsa_indices['index_value'] > 50)
            assert all(cbsa_indices['index_value'] < 200)
            
    def test_supertract_construction_integration(self):
        """Test supertract construction with real-like data"""
        # Generate data with varying density
        syn_gen = SyntheticDataGenerator(seed=42)
        geo_gen = GeographicDataGenerator(seed=42)
        
        # Create some tracts with insufficient data
        sparse_data = []
        
        # Dense tract (sufficient data)
        for i in range(30):
            sparse_data.append({
                'property_id': f'P1_{i}',
                'tract_id': 'T001',
                'cbsa_id': 'C1',
                'first_sale_date': pd.Timestamp('2019-01-01') + timedelta(days=i*10),
                'second_sale_date': pd.Timestamp('2020-01-01') + timedelta(days=i*10),
                'first_sale_price': 200000 + i * 1000,
                'second_sale_price': 220000 + i * 1000
            })
        
        # Sparse tracts (insufficient data)
        for tract_num in range(2, 5):
            for i in range(10):  # Only 10 pairs each
                sparse_data.append({
                    'property_id': f'P{tract_num}_{i}',
                    'tract_id': f'T00{tract_num}',
                    'cbsa_id': 'C1',
                    'first_sale_date': pd.Timestamp('2019-01-01') + timedelta(days=i*20),
                    'second_sale_date': pd.Timestamp('2020-01-01') + timedelta(days=i*20),
                    'first_sale_price': 200000 + i * 1000,
                    'second_sale_price': 220000 + i * 1000
                })
        
        pairs_df = pd.DataFrame(sparse_data)
        
        # Create adjacency
        adjacency = {
            'T001': ['T002'],
            'T002': ['T001', 'T003'],
            'T003': ['T002', 'T004'],
            'T004': ['T003']
        }
        
        # Construct supertracts
        constructor = SupertractConstructor(
            min_half_pairs=40,
            adjacency_data=adjacency
        )
        
        # Create minimal tract data
        tract_data = pd.DataFrame([
            {'tract_id': f'T00{i}', 'cbsa_id': 'C1', 'state': '01', 'county': '001'}
            for i in range(1, 5)
        ])
        
        supertracts = constructor.construct_supertracts(
            pairs_df,
            tract_data,
            2020
        )
        
        # Verify construction
        assert len(supertracts) > 0
        
        # T001 should be alone (60 half-pairs)
        # T002, T003, T004 should be combined (20 + 20 + 20 = 60 half-pairs)
        single_tract_found = False
        multi_tract_found = False
        
        for st_id, supertract in supertracts.items():
            if len(supertract.component_tract_ids) == 1:
                assert 'T001' in supertract.component_tract_ids
                single_tract_found = True
            elif len(supertract.component_tract_ids) > 1:
                multi_tract_found = True
                
        assert single_tract_found
        assert multi_tract_found
        
    def test_regression_with_real_patterns(self):
        """Test regression with realistic price patterns"""
        # Create data with known appreciation pattern
        data = []
        base_price = 200000
        monthly_appreciation = 0.005  # 0.5% per month
        
        for prop_id in range(50):
            # First sale random month in 2018
            first_month = np.random.randint(1, 12)
            first_date = pd.Timestamp(f'2018-{first_month:02d}-15')
            
            # Second sale 12-24 months later
            months_later = np.random.randint(12, 24)
            second_date = first_date + pd.DateOffset(months=months_later)
            
            # Calculate prices with known appreciation
            first_price = base_price * (1 + np.random.normal(0, 0.1))
            second_price = first_price * ((1 + monthly_appreciation) ** months_later)
            
            # Add some noise
            second_price *= (1 + np.random.normal(0, 0.05))
            
            data.append({
                'property_id': f'P{prop_id}',
                'tract_id': 'T001',
                'cbsa_id': 'C1',
                'first_sale_date': first_date,
                'first_sale_price': first_price,
                'second_sale_date': second_date,
                'second_sale_price': second_price
            })
        
        pairs_df = pd.DataFrame(data)
        
        # Run regression
        regression = RepeatSalesRegression()
        results = regression.fit(pairs_df)
        
        # Get index values
        index_df = regression.get_index_values()
        
        # Calculate monthly returns
        returns_df = regression.calculate_returns('monthly')
        
        # Average monthly return should be close to true appreciation
        avg_return = returns_df['return'].mean()
        assert abs(avg_return - monthly_appreciation) < 0.002  # Within 0.2%
        
        # R-squared should be decent
        assert results.r_squared > 0.5
        
    def test_multi_geography_aggregation(self):
        """Test aggregation across multiple geography levels"""
        # Generate multi-CBSA data
        syn_gen = SyntheticDataGenerator(seed=42)
        geo_gen = GeographicDataGenerator(seed=42)
        
        transaction_data = syn_gen.generate_complete_dataset(
            start_year=2019,
            end_year=2020,
            num_cbsas=5,
            num_tracts=50,
            num_properties=500,
            target_pairs=250
        )
        
        geo_data = geo_gen.generate_complete_geographic_data(
            num_tracts=50
        )
        
        # Validate pairs
        validator = DataValidator()
        validated_df = validator.validate_transaction_batch(
            transaction_data['pairs']
        )
        valid_pairs = validated_df[validated_df['is_valid']]
        
        # Estimate at different levels
        estimator = BMNIndexEstimator(min_half_pairs=20)
        
        # CBSA level
        cbsa_results = estimator.estimate_indices(
            valid_pairs,
            geo_data['tracts'],
            WeightType.SAMPLE,
            geography_level='cbsa'
        )
        
        # National level
        national_results = estimator.estimate_indices(
            valid_pairs,
            geo_data['tracts'],
            WeightType.SAMPLE,
            geography_level='national'
        )
        
        # Verify different aggregation levels
        assert len(cbsa_results.index_values['geography_id'].unique()) > 1
        assert len(national_results.index_values['geography_id'].unique()) == 1
        assert national_results.index_values['geography_id'].iloc[0] == 'USA'
        
        # National index should be weighted average of CBSA indices
        # (approximately, given the weighting scheme)
        national_mean = national_results.index_values['index_value'].mean()
        cbsa_mean = cbsa_results.index_values['index_value'].mean()
        
        # Should be in similar range
        assert abs(national_mean - cbsa_mean) / cbsa_mean < 0.2  # Within 20%