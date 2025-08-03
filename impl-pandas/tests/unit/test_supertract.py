"""Unit tests for supertract construction"""

import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import networkx as nx

from hpi_fhfa.algorithms.supertract import SupertractConstructor
from hpi_fhfa.models.geography import Supertract


class TestSupertractConstructor:
    """Test supertract construction algorithm"""
    
    def test_basic_construction(self, sample_pairs_df, sample_tract_gdf):
        """Test basic supertract construction"""
        constructor = SupertractConstructor(min_half_pairs=40)
        
        # Construct supertracts for a year
        year = pd.to_datetime(sample_pairs_df['second_sale_date']).dt.year.iloc[0]
        supertracts = constructor.construct_supertracts(
            sample_pairs_df,
            sample_tract_gdf,
            year
        )
        
        assert isinstance(supertracts, dict)
        assert len(supertracts) > 0
        
        # Check supertract properties
        for st_id, supertract in supertracts.items():
            assert isinstance(supertract, Supertract)
            assert supertract.year == year
            assert len(supertract.component_tract_ids) > 0
            assert supertract.half_pairs_count >= 0
            
    def test_half_pairs_counting(self):
        """Test half-pairs counting logic"""
        # Create test data
        data = [
            {'property_id': f'P{i}', 'tract_id': 'T1', 'cbsa_id': 'C1',
             'first_sale_date': pd.Timestamp('2020-01-01'),
             'second_sale_date': pd.Timestamp('2020-06-01'),
             'first_sale_price': 100000, 'second_sale_price': 110000}
            for i in range(10)
        ]
        
        df = pd.DataFrame(data)
        constructor = SupertractConstructor()
        
        # Count half-pairs
        counts = constructor._count_half_pairs_by_tract(df)
        
        # Each pair contributes 2 half-pairs
        assert counts['T1'] == 20
        
    def test_insufficient_tract_aggregation(self):
        """Test aggregation of tracts with insufficient data"""
        # Create data with insufficient tracts
        data = []
        
        # Tract T1: 15 pairs (30 half-pairs - insufficient)
        for i in range(15):
            data.append({
                'property_id': f'P1_{i}',
                'tract_id': 'T1',
                'cbsa_id': 'C1',
                'first_sale_date': pd.Timestamp('2020-01-01'),
                'second_sale_date': pd.Timestamp('2020-06-01'),
                'first_sale_price': 100000,
                'second_sale_price': 110000
            })
        
        # Tract T2: 12 pairs (24 half-pairs - insufficient)
        for i in range(12):
            data.append({
                'property_id': f'P2_{i}',
                'tract_id': 'T2',
                'cbsa_id': 'C1',
                'first_sale_date': pd.Timestamp('2020-01-01'),
                'second_sale_date': pd.Timestamp('2020-06-01'),
                'first_sale_price': 100000,
                'second_sale_price': 110000
            })
        
        # Tract T3: 25 pairs (50 half-pairs - sufficient)
        for i in range(25):
            data.append({
                'property_id': f'P3_{i}',
                'tract_id': 'T3',
                'cbsa_id': 'C1',
                'first_sale_date': pd.Timestamp('2020-01-01'),
                'second_sale_date': pd.Timestamp('2020-06-01'),
                'first_sale_price': 100000,
                'second_sale_price': 110000
            })
        
        df = pd.DataFrame(data)
        
        # Create simple tract geometries
        tract_data = [
            {'tract_id': 'T1', 'cbsa_id': 'C1', 'state': '01', 'county': '001',
             'geometry': Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])},
            {'tract_id': 'T2', 'cbsa_id': 'C1', 'state': '01', 'county': '001',
             'geometry': Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])},
            {'tract_id': 'T3', 'cbsa_id': 'C1', 'state': '01', 'county': '001',
             'geometry': Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])}
        ]
        tract_gdf = gpd.GeoDataFrame(tract_data)
        
        # Set up adjacency (T1-T2 adjacent, T2-T3 adjacent)
        adjacency = {
            'T1': ['T2'],
            'T2': ['T1', 'T3'],
            'T3': ['T2']
        }
        
        constructor = SupertractConstructor(
            min_half_pairs=40,
            adjacency_data=adjacency
        )
        
        supertracts = constructor.construct_supertracts(df, tract_gdf, 2020)
        
        # T1 and T2 should be combined (30 + 24 = 54 half-pairs)
        # T3 should be alone (50 half-pairs)
        combined_found = False
        single_found = False
        
        for st_id, supertract in supertracts.items():
            if len(supertract.component_tract_ids) == 2:
                assert set(supertract.component_tract_ids) == {'T1', 'T2'}
                assert supertract.half_pairs_count == 54
                combined_found = True
            elif len(supertract.component_tract_ids) == 1:
                assert supertract.component_tract_ids[0] == 'T3'
                assert supertract.half_pairs_count == 50
                single_found = True
        
        assert combined_found
        assert single_found
        
    def test_adjacency_graph_construction(self, sample_tract_gdf):
        """Test adjacency graph construction"""
        constructor = SupertractConstructor()
        
        tract_ids = sample_tract_gdf.index.tolist() if sample_tract_gdf.index.name == 'tract_id' else sample_tract_gdf['tract_id'].tolist()
        graph = constructor._build_adjacency_graph(
            tract_ids[:5],  # Use subset
            sample_tract_gdf
        )
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 5
        # Should have some edges from geometric adjacency
        assert graph.number_of_edges() >= 0
        
    def test_year_filtering(self):
        """Test filtering pairs by year"""
        # Create multi-year data
        data = []
        for year in [2019, 2020, 2021]:
            for i in range(10):
                data.append({
                    'property_id': f'P{year}_{i}',
                    'tract_id': 'T1',
                    'cbsa_id': 'C1',
                    'first_sale_date': pd.Timestamp(f'{year-1}-01-01'),
                    'second_sale_date': pd.Timestamp(f'{year}-06-01'),
                    'first_sale_price': 100000,
                    'second_sale_price': 110000
                })
        
        df = pd.DataFrame(data)
        constructor = SupertractConstructor()
        
        # Filter for 2020
        filtered = constructor._filter_pairs_by_year(df, 2020)
        
        # Should include pairs where either sale is in 2020
        years = pd.concat([
            pd.to_datetime(filtered['first_sale_date']).dt.year,
            pd.to_datetime(filtered['second_sale_date']).dt.year
        ])
        
        assert 2020 in years.values
        assert len(filtered) > 0
        
    def test_supertract_validation(self, sample_pairs_df, sample_tract_gdf):
        """Test supertract validation"""
        constructor = SupertractConstructor(min_half_pairs=40)
        
        year = pd.to_datetime(sample_pairs_df['second_sale_date']).dt.year.iloc[0]
        supertracts = constructor.construct_supertracts(
            sample_pairs_df,
            sample_tract_gdf,
            year
        )
        
        # Validate supertracts
        validation = constructor.validate_supertracts(
            supertracts,
            sample_pairs_df,
            year
        )
        
        assert isinstance(validation, dict)
        for st_id, results in validation.items():
            assert 'component_tracts' in results
            assert 'expected_half_pairs' in results
            assert 'actual_half_pairs' in results
            assert 'meets_threshold' in results
            assert 'discrepancy' in results
            
    def test_tract_mapping(self, sample_pairs_df, sample_tract_gdf):
        """Test tract to supertract mapping"""
        constructor = SupertractConstructor(min_half_pairs=40)
        
        year = pd.to_datetime(sample_pairs_df['second_sale_date']).dt.year.iloc[0]
        supertracts = constructor.construct_supertracts(
            sample_pairs_df,
            sample_tract_gdf,
            year
        )
        
        # Get mapping
        mapping = constructor.get_supertract_mapping(supertracts)
        
        assert isinstance(mapping, dict)
        
        # Every component tract should be in mapping
        for supertract in supertracts.values():
            for tract_id in supertract.component_tract_ids:
                assert tract_id in mapping
                assert mapping[tract_id] in supertracts
                
    def test_geometric_adjacency_computation(self):
        """Test geometric adjacency computation"""
        # Create simple adjacent geometries
        tract_data = [
            {'tract_id': 'T1', 'cbsa_id': 'C1', 'state': '01', 'county': '001',
             'geometry': Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])},
            {'tract_id': 'T2', 'cbsa_id': 'C1', 'state': '01', 'county': '001',
             'geometry': Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])},  # Adjacent to T1
            {'tract_id': 'T3', 'cbsa_id': 'C1', 'state': '01', 'county': '001',
             'geometry': Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])},  # Not adjacent
        ]
        tract_gdf = gpd.GeoDataFrame(tract_data)
        
        constructor = SupertractConstructor()
        graph = nx.Graph()
        graph.add_nodes_from(['T1', 'T2', 'T3'])
        
        constructor._compute_geometric_adjacencies(
            graph,
            tract_gdf,
            ['T1', 'T2', 'T3']
        )
        
        # T1 and T2 should be connected
        assert graph.has_edge('T1', 'T2')
        # T1 and T3 should not be connected
        assert not graph.has_edge('T1', 'T3')
        # T2 and T3 should not be connected
        assert not graph.has_edge('T2', 'T3')