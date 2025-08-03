"""Unit tests for data generators"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from hpi_fhfa.data import SyntheticDataGenerator, GeographicDataGenerator


class TestSyntheticDataGenerator:
    """Test synthetic data generation"""
    
    def test_generator_initialization(self, synthetic_generator):
        """Test generator setup"""
        assert synthetic_generator.seed == 42
        assert len(synthetic_generator.market_profiles) == 3
        assert 'high_growth' in synthetic_generator.market_profiles
        
    def test_cbsa_assignments(self, synthetic_generator):
        """Test CBSA assignment generation"""
        assignments = synthetic_generator.generate_cbsa_assignments(
            num_tracts=100,
            num_cbsas=5
        )
        
        # Check all tracts assigned
        assert len(assignments) == 100
        
        # Check CBSA count
        unique_cbsas = set(assignments.values())
        assert len(unique_cbsas) == 5
        
        # Check each CBSA has at least some tracts
        cbsa_counts = {}
        for tract, cbsa in assignments.items():
            cbsa_counts[cbsa] = cbsa_counts.get(cbsa, 0) + 1
        
        for cbsa, count in cbsa_counts.items():
            assert count >= 5  # Minimum size per CBSA
    
    def test_property_base_data(self, synthetic_generator):
        """Test property base data generation"""
        assignments = {'tract1': 'cbsa1', 'tract2': 'cbsa2'}
        profiles = {'cbsa1': 'high_growth', 'cbsa2': 'low_growth'}
        
        properties_df = synthetic_generator.generate_property_base_data(
            num_properties=50,
            tract_assignments=assignments,
            cbsa_profiles=profiles
        )
        
        assert len(properties_df) == 50
        assert 'property_id' in properties_df.columns
        assert 'base_price' in properties_df.columns
        assert properties_df['base_price'].min() >= 50000
        
        # Check profile assignment
        high_growth = properties_df[properties_df['profile_type'] == 'high_growth']
        low_growth = properties_df[properties_df['profile_type'] == 'low_growth']
        
        # High growth should have higher average prices
        assert high_growth['base_price'].mean() > low_growth['base_price'].mean()
    
    def test_market_indices(self, synthetic_generator):
        """Test market index generation"""
        profiles = {
            'cbsa1': 'high_growth',
            'cbsa2': 'moderate_growth',
            'cbsa3': 'low_growth'
        }
        
        indices_df = synthetic_generator.generate_market_indices(
            start_year=2000,
            end_year=2023,
            cbsa_profiles=profiles
        )
        
        # Check structure
        assert len(indices_df) == 3 * 24  # 3 CBSAs × 24 years
        assert 'index_value' in indices_df.columns
        assert 'annual_return' in indices_df.columns
        
        # Check base year
        base_indices = indices_df[indices_df['year'] == 2000]
        assert all(base_indices['index_value'] == 1.0)
        
        # Check recession years have negative returns
        recession_2008 = indices_df[indices_df['year'] == 2008]
        assert recession_2008['annual_return'].mean() < 0
        
        # Check growth patterns
        for cbsa_id, profile_type in profiles.items():
            cbsa_data = indices_df[indices_df['cbsa_id'] == cbsa_id]
            final_index = cbsa_data[cbsa_data['year'] == 2023]['index_value'].iloc[0]
            
            if profile_type == 'high_growth':
                assert final_index > 2.0  # Should have doubled
            elif profile_type == 'low_growth':
                assert final_index < 2.0  # Less than doubled
    
    def test_transaction_generation(self, synthetic_generator):
        """Test transaction generation"""
        # Create minimal test data
        properties_df = pd.DataFrame([
            {'property_id': 'P001', 'tract_id': 't1', 'cbsa_id': 'c1', 
             'base_price': 200000, 'profile_type': 'moderate_growth'},
            {'property_id': 'P002', 'tract_id': 't1', 'cbsa_id': 'c1',
             'base_price': 300000, 'profile_type': 'moderate_growth'}
        ])
        
        market_indices = pd.DataFrame([
            {'cbsa_id': 'c1', 'year': 2010, 'index_value': 1.0},
            {'cbsa_id': 'c1', 'year': 2011, 'index_value': 1.05},
            {'cbsa_id': 'c1', 'year': 2012, 'index_value': 1.08},
            {'cbsa_id': 'c1', 'year': 2013, 'index_value': 1.12},
            {'cbsa_id': 'c1', 'year': 2014, 'index_value': 1.15},
            {'cbsa_id': 'c1', 'year': 2015, 'index_value': 1.20}
        ])
        
        transactions_df = synthetic_generator.generate_transactions(
            properties_df=properties_df,
            market_indices_df=market_indices,
            start_year=2010,
            end_year=2015,
            target_pairs=2
        )
        
        assert len(transactions_df) >= 4  # At least 2 pairs (4 transactions)
        assert 'sale_date' in transactions_df.columns
        assert 'sale_price' in transactions_df.columns
        
        # Check prices are reasonable
        assert transactions_df['sale_price'].min() >= 10000
        
    def test_create_repeat_sales_pairs(self, synthetic_generator):
        """Test pair creation from transactions"""
        transactions = pd.DataFrame([
            {'property_id': 'P001', 'tract_id': 't1', 'cbsa_id': 'c1',
             'sale_date': date(2010, 1, 15), 'sale_price': 200000},
            {'property_id': 'P001', 'tract_id': 't1', 'cbsa_id': 'c1',
             'sale_date': date(2013, 6, 20), 'sale_price': 230000},
            {'property_id': 'P001', 'tract_id': 't1', 'cbsa_id': 'c1',
             'sale_date': date(2018, 3, 10), 'sale_price': 280000},
            {'property_id': 'P002', 'tract_id': 't2', 'cbsa_id': 'c1',
             'sale_date': date(2011, 5, 1), 'sale_price': 150000},
            {'property_id': 'P002', 'tract_id': 't2', 'cbsa_id': 'c1',
             'sale_date': date(2015, 8, 15), 'sale_price': 180000}
        ])
        
        pairs_df = synthetic_generator.create_repeat_sales_pairs(transactions)
        
        assert len(pairs_df) == 3  # P001: 2 pairs, P002: 1 pair
        
        # Check first pair of P001
        p001_pairs = pairs_df[pairs_df['property_id'] == 'P001']
        assert len(p001_pairs) == 2
        assert p001_pairs.iloc[0]['first_sale_price'] == 200000
        assert p001_pairs.iloc[0]['second_sale_price'] == 230000
    
    def test_complete_dataset_generation(self, synthetic_generator):
        """Test complete dataset generation"""
        data = synthetic_generator.generate_complete_dataset(
            start_year=2010,
            end_year=2015,
            num_cbsas=3,
            num_tracts=20,
            num_properties=100,
            target_pairs=50
        )
        
        # Check all components present
        assert 'transactions' in data
        assert 'pairs' in data
        assert 'properties' in data
        assert 'tracts' in data
        assert 'market_indices' in data
        
        # Check data consistency
        assert len(data['properties']) == 100
        assert len(data['tracts']) == 20
        assert len(data['pairs']) >= 40  # May not reach exact target
        
        # Check tract-CBSA mapping
        cbsa_ids = data['tracts']['cbsa_id'].unique()
        assert len(cbsa_ids) == 3
    
    def test_demographic_data_generation(self, synthetic_generator):
        """Test demographic data generation"""
        tracts_df = pd.DataFrame([
            {'tract_id': 't1', 'cbsa_id': 'c1'},
            {'tract_id': 't2', 'cbsa_id': 'c1'},
            {'tract_id': 't3', 'cbsa_id': 'c2'}
        ])
        
        demo_df = synthetic_generator.add_demographic_data(
            tracts_df=tracts_df,
            years=[2010, 2015, 2020]
        )
        
        assert len(demo_df) == 9  # 3 tracts × 3 years
        assert 'housing_units' in demo_df.columns
        assert 'median_value' in demo_df.columns
        assert 'college_share' in demo_df.columns
        
        # Check trends over time
        t1_data = demo_df[demo_df['tract_id'] == 't1'].sort_values('year')
        assert t1_data.iloc[-1]['housing_units'] >= t1_data.iloc[0]['housing_units']
        assert t1_data.iloc[-1]['median_value'] >= t1_data.iloc[0]['median_value']


class TestGeographicDataGenerator:
    """Test geographic data generation"""
    
    def test_tract_geometry_generation(self, geographic_generator):
        """Test tract geometry creation"""
        tract_gdf = geographic_generator.generate_tract_geometries(
            num_tracts=50,
            bounds=(-80, 25, -75, 30)  # Small test area
        )
        
        assert len(tract_gdf) == 50
        assert 'geometry' in tract_gdf.columns
        assert 'centroid' in tract_gdf.columns
        assert 'tract_id' in tract_gdf.columns
        
        # Check geometries are valid
        assert all(tract_gdf.geometry.is_valid)
        
        # Check bounds
        bounds = tract_gdf.total_bounds
        assert bounds[0] >= -80.5  # Some buffer for tract size
        assert bounds[1] >= 24.5
        assert bounds[2] <= -74.5
        assert bounds[3] <= 30.5
        
        # Check clustering (CBSAs)
        cbsa_counts = tract_gdf['cbsa_id'].value_counts()
        assert len(cbsa_counts) > 1  # Multiple CBSAs
        assert cbsa_counts.max() > 1  # Each CBSA has multiple tracts
    
    def test_cbsa_data_generation(self, geographic_generator):
        """Test CBSA metadata generation"""
        # Create sample tract data
        tract_gdf = pd.DataFrame([
            {'tract_id': 't1', 'cbsa_id': 'c1', 'state': '12', 'population': 5000, 'area_sqmi': 10},
            {'tract_id': 't2', 'cbsa_id': 'c1', 'state': '12', 'population': 3000, 'area_sqmi': 8},
            {'tract_id': 't3', 'cbsa_id': 'c2', 'state': '13', 'population': 4000, 'area_sqmi': 12}
        ])
        
        cbsa_df = geographic_generator.generate_cbsa_data(tract_gdf)
        
        assert len(cbsa_df) == 2
        assert 'name' in cbsa_df.columns
        assert 'tract_count' in cbsa_df.columns
        
        # Check aggregations
        c1_data = cbsa_df[cbsa_df['cbsa_id'] == 'c1'].iloc[0]
        assert c1_data['tract_count'] == 2
        assert c1_data['total_population'] == 8000
        assert c1_data['total_area_sqmi'] == 18
    
    def test_tract_adjacency(self, geographic_generator):
        """Test adjacency relationship generation"""
        from shapely.geometry import Point
        
        # Create simple tract data with known positions
        tract_data = []
        for i in range(4):
            tract_data.append({
                'tract_id': f't{i}',
                'cbsa_id': 'c1',
                'centroid': Point(i, 0)  # Linear arrangement
            })
        
        tract_gdf = pd.DataFrame(tract_data)
        
        adjacency = geographic_generator.generate_tract_adjacency(
            tract_gdf,
            max_neighbors=2
        )
        
        # Check adjacency structure
        assert len(adjacency) == 4
        
        # t1 should have t0 and t2 as nearest neighbors
        assert 't0' in adjacency['t1']
        assert 't2' in adjacency['t1']
        
        # t0 should have t1 as nearest neighbor
        assert 't1' in adjacency['t0']
    
    def test_complete_geographic_generation(self, geographic_generator):
        """Test complete geographic data generation"""
        geo_data = geographic_generator.generate_complete_geographic_data(
            num_tracts=100
        )
        
        assert 'tracts' in geo_data
        assert 'cbsas' in geo_data  
        assert 'adjacency' in geo_data
        
        # Check consistency
        tract_ids = set(geo_data['tracts']['tract_id'])
        adjacency_ids = set(geo_data['adjacency'].keys())
        
        assert tract_ids == adjacency_ids
        
        # Check all adjacency lists are valid
        for tract_id, neighbors in geo_data['adjacency'].items():
            for neighbor in neighbors:
                assert neighbor in tract_ids
    
    def test_demographic_weights_generation(self, geographic_generator):
        """Test demographic weight data generation"""
        import geopandas as gpd
        from shapely.geometry import Point, Polygon
        
        # Create sample tract data
        tract_data = []
        for i in range(5):
            coords = [(i, 0), (i+1, 0), (i+1, 1), (i, 1), (i, 0)]
            tract_data.append({
                'tract_id': f't{i}',
                'population': 1000 * (i + 1),
                'geometry': Polygon(coords)
            })
        
        tract_gdf = gpd.GeoDataFrame(tract_data)
        
        demo_df = geographic_generator.create_sample_demographic_weights(tract_gdf)
        
        assert len(demo_df) == 5
        assert 'housing_units' in demo_df.columns
        assert 'college_share' in demo_df.columns
        
        # Check value ranges
        assert all(0 <= demo_df['college_share']) and all(demo_df['college_share'] <= 1)
        assert all(0 <= demo_df['non_white_share']) and all(demo_df['non_white_share'] <= 1)
        assert all(demo_df['median_value'] > 0)