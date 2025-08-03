"""Unit tests for data models"""

import pytest
import numpy as np
from datetime import date

from hpi_fhfa.models.transaction import (
    TransactionPair, 
    validate_transaction_pair,
    create_transaction_pairs_df
)
from hpi_fhfa.models.geography import Tract, CBSA, Supertract
from hpi_fhfa.models.weights import WeightType, DemographicData, WeightSet


class TestTransactionPair:
    """Test TransactionPair model"""
    
    def test_transaction_pair_properties(self, sample_transaction_pair):
        """Test calculated properties"""
        pair = sample_transaction_pair
        
        # Test log price difference
        expected_log_diff = np.log(325000) - np.log(250000)
        assert abs(pair.log_price_diff - expected_log_diff) < 1e-6
        
        # Test time difference
        assert abs(pair.time_diff_years - 5.43) < 0.1
        
        # Test CAGR
        expected_cagr = (325000 / 250000) ** (1 / pair.time_diff_years) - 1
        assert abs(pair.cagr - expected_cagr) < 1e-6
    
    def test_transaction_validation_valid(self, sample_transaction_pair):
        """Test validation of valid transaction"""
        is_valid, error = validate_transaction_pair(sample_transaction_pair)
        assert is_valid
        assert error is None
    
    def test_transaction_validation_same_period(self):
        """Test rejection of same 12-month period"""
        pair = TransactionPair(
            property_id="P001",
            tract_id="12345678901", 
            cbsa_id="10420",
            first_sale_date=date(2010, 1, 15),
            first_sale_price=250000.0,
            second_sale_date=date(2010, 11, 20),
            second_sale_price=260000.0
        )
        is_valid, error = validate_transaction_pair(pair)
        assert not is_valid
        assert "same 12-month period" in error.lower()
    
    def test_transaction_validation_high_cagr(self):
        """Test rejection of high CAGR"""
        pair = TransactionPair(
            property_id="P001",
            tract_id="12345678901",
            cbsa_id="10420", 
            first_sale_date=date(2010, 1, 15),
            first_sale_price=250000.0,
            second_sale_date=date(2012, 1, 15),
            second_sale_price=500000.0  # 100% appreciation in 2 years
        )
        is_valid, error = validate_transaction_pair(pair)
        assert not is_valid
        assert "CAGR" in error
    
    def test_transaction_validation_extreme_appreciation(self):
        """Test rejection of extreme appreciation"""
        # Test > 10x appreciation
        pair = TransactionPair(
            property_id="P001",
            tract_id="12345678901",
            cbsa_id="10420",
            first_sale_date=date(2010, 1, 15),
            first_sale_price=100000.0,
            second_sale_date=date(2020, 1, 15),
            second_sale_price=1100000.0  # 11x
        )
        is_valid, error = validate_transaction_pair(pair)
        assert not is_valid
        assert "10x" in error
        
        # Test < 0.75x appreciation  
        pair = TransactionPair(
            property_id="P001",
            tract_id="12345678901",
            cbsa_id="10420",
            first_sale_date=date(2010, 1, 15),
            first_sale_price=100000.0,
            second_sale_date=date(2020, 1, 15),
            second_sale_price=70000.0  # 0.7x
        )
        is_valid, error = validate_transaction_pair(pair)
        assert not is_valid
        assert "0.75x" in error
    
    def test_create_pairs_dataframe(self):
        """Test creating DataFrame from transaction pairs"""
        pairs = []
        for i in range(5):
            pairs.append(TransactionPair(
                property_id=f"P{i:03d}",
                tract_id="12345678901",
                cbsa_id="10420",
                first_sale_date=date(2010 + i, 1, 1),
                first_sale_price=200000 + i * 10000,
                second_sale_date=date(2015 + i, 1, 1), 
                second_sale_price=250000 + i * 15000
            ))
        
        df = create_transaction_pairs_df(pairs)
        assert len(df) == 5
        assert 'log_price_diff' in df.columns
        assert 'cagr' in df.columns
        assert df['property_id'].tolist() == ['P000', 'P001', 'P002', 'P003', 'P004']


class TestGeographyModels:
    """Test geography models"""
    
    def test_tract_distance(self, sample_tract):
        """Test distance calculation between tracts"""
        # Create another tract
        from shapely.geometry import Point, Polygon
        coords = [(-79.5, 25.5), (-79.4, 25.5), (-79.4, 25.6), (-79.5, 25.6), (-79.5, 25.5)]
        other_tract = Tract(
            tract_id="12345678902",
            cbsa_id="10420",
            state="12",
            county="345",
            geometry=Polygon(coords),
            centroid=Point(-79.45, 25.55)
        )
        
        distance = sample_tract.distance_to(other_tract)
        # Should be approximately 0.707 degrees (diagonal distance)
        assert 0.6 < distance < 0.8
    
    def test_cbsa_properties(self):
        """Test CBSA model"""
        cbsa = CBSA(
            cbsa_id="10420",
            name="Miami-Fort Lauderdale-Pompano Beach, FL",
            state="FL",
            tract_ids=["12345678901", "12345678902", "12345678903"]
        )
        
        assert cbsa.tract_count == 3
        assert "12345678902" in cbsa.tract_ids
    
    def test_supertract_properties(self):
        """Test Supertract model"""
        supertract = Supertract(
            supertract_id="super_12345678901",
            component_tract_ids=["12345678901", "12345678902"],
            year=2020,
            half_pairs_count=45
        )
        
        assert not supertract.is_single_tract
        assert supertract.contains_tract("12345678901")
        assert not supertract.contains_tract("12345678903")


class TestWeightModels:
    """Test weight models"""
    
    def test_weight_types(self):
        """Test weight type enum"""
        assert WeightType.SAMPLE.value == "sample"
        assert WeightType.VALUE.value == "value"
        assert len(WeightType) == 6
    
    def test_demographic_data(self):
        """Test demographic data model"""
        demo = DemographicData(
            tract_id="12345678901",
            year=2020,
            housing_units=1000,
            median_value=300000.0,
            college_share=0.35,
            non_white_share=0.45,
            upb_total=150000000.0
        )
        
        # Test weight value calculations
        assert demo.get_weight_value(WeightType.VALUE) == 1000 * 300000
        assert demo.get_weight_value(WeightType.UNIT) == 1000
        assert demo.get_weight_value(WeightType.UPB) == 150000000.0
        assert demo.get_weight_value(WeightType.COLLEGE) == 0.35
        assert demo.get_weight_value(WeightType.NON_WHITE) == 0.45
    
    def test_weight_set_normalization(self):
        """Test weight set normalization"""
        weights = WeightSet(
            geography_id="10420",
            period=2020,
            weight_type=WeightType.SAMPLE,
            weights={
                "tract1": 0.3,
                "tract2": 0.5,
                "tract3": 0.2
            }
        )
        
        assert weights.is_normalized
        
        # Test non-normalized weights
        weights.weights["tract4"] = 0.1
        assert not weights.is_normalized
        
        # Normalize
        weights.normalize()
        assert weights.is_normalized
        assert abs(sum(weights.weights.values()) - 1.0) < 1e-10