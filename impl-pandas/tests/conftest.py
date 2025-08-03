"""Pytest configuration and fixtures"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
import tempfile
from pathlib import Path

from hpi_fhfa.data import SyntheticDataGenerator, GeographicDataGenerator
from hpi_fhfa.models.transaction import TransactionPair
from hpi_fhfa.models.geography import Tract
from shapely.geometry import Point, Polygon


@pytest.fixture
def sample_transaction_pair():
    """Create a sample transaction pair"""
    return TransactionPair(
        property_id="P001",
        tract_id="12345678901",
        cbsa_id="10420",
        first_sale_date=date(2010, 1, 15),
        first_sale_price=250000.0,
        second_sale_date=date(2015, 6, 20),
        second_sale_price=325000.0
    )


@pytest.fixture
def sample_tract():
    """Create a sample tract"""
    coords = [(-80.0, 25.0), (-79.9, 25.0), (-79.9, 25.1), (-80.0, 25.1), (-80.0, 25.0)]
    return Tract(
        tract_id="12345678901",
        cbsa_id="10420",
        state="12",
        county="345",
        geometry=Polygon(coords),
        centroid=Point(-79.95, 25.05)
    )


@pytest.fixture
def sample_pairs_df():
    """Create sample repeat-sales pairs DataFrame"""
    data = []
    base_date = date(2010, 1, 1)
    
    # Use consistent tract IDs matching geographic generator format
    state_codes = ['01', '02', '03', '04']
    county_codes = ['001', '002', '003', '004']
    
    for i in range(100):
        first_date = base_date + timedelta(days=np.random.randint(0, 1000))
        second_date = first_date + timedelta(days=np.random.randint(400, 2000))
        first_price = np.random.uniform(100000, 500000)
        appreciation = np.random.uniform(0.9, 1.5)
        
        # Generate tract ID in correct format: SSCCCTTTTTT (state+county+tract)
        state = state_codes[np.random.randint(0, len(state_codes))]
        county = county_codes[np.random.randint(0, len(county_codes))]
        tract = f"{np.random.randint(1, 6):06d}"  # 000001-000005
        tract_id = f"{state}{county}{tract}"
        
        # Match CBSA to state
        cbsa_id = str(10000 + int(state) * 20)
        
        data.append({
            'property_id': f"P{i:04d}",
            'tract_id': tract_id,
            'cbsa_id': cbsa_id,
            'first_sale_date': first_date,
            'first_sale_price': first_price,
            'second_sale_date': second_date,
            'second_sale_price': first_price * appreciation
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_generator():
    """Create synthetic data generator with fixed seed"""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def geographic_generator():
    """Create geographic data generator with fixed seed"""
    return GeographicDataGenerator(seed=42)


@pytest.fixture
def sample_tract_gdf(geographic_generator):
    """Sample tract GeoDataFrame for testing"""
    geo_data = geographic_generator.generate_complete_geographic_data(
        num_tracts=20
    )
    return geo_data['tracts']


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)