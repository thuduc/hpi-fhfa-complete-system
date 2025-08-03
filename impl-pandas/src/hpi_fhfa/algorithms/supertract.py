"""Supertract Construction Algorithm"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import networkx as nx

from ..models.geography import Supertract, Tract


class SupertractConstructor:
    """
    Construct supertracts by aggregating census tracts with insufficient data
    
    Supertracts are created when individual tracts don't have enough
    repeat-sales pairs (minimum 40 half-pairs) for reliable index estimation.
    """
    
    def __init__(self,
                 min_half_pairs: int = 40,
                 adjacency_data: Optional[Dict[str, List[str]]] = None):
        """
        Initialize supertract constructor
        
        Args:
            min_half_pairs: Minimum number of half-pairs required
            adjacency_data: Dict mapping tract IDs to list of adjacent tract IDs
        """
        self.min_half_pairs = min_half_pairs
        self.adjacency_data = adjacency_data or {}
        
    def construct_supertracts(self,
                            pairs_df: pd.DataFrame,
                            tract_gdf: pd.DataFrame,
                            year: int) -> Dict[str, Supertract]:
        """
        Construct supertracts for a given year
        
        Args:
            pairs_df: DataFrame of transaction pairs
            tract_gdf: GeoDataFrame of tract geometries  
            year: Year to construct supertracts for
            
        Returns:
            Dict mapping supertract ID to Supertract object
        """
        # Filter pairs for the specified year
        year_pairs = self._filter_pairs_by_year(pairs_df, year)
        
        # Count half-pairs by tract
        tract_counts = self._count_half_pairs_by_tract(year_pairs)
        
        # Identify tracts needing aggregation
        insufficient_tracts = {
            tract_id: count 
            for tract_id, count in tract_counts.items()
            if count < self.min_half_pairs
        }
        
        # Build adjacency graph
        graph = self._build_adjacency_graph(
            list(tract_counts.keys()), 
            tract_gdf
        )
        
        # Construct supertracts through aggregation
        supertracts = self._aggregate_tracts(
            insufficient_tracts,
            tract_counts,
            graph,
            year
        )
        
        # Add single-tract supertracts for sufficient tracts
        for tract_id, count in tract_counts.items():
            if count >= self.min_half_pairs:
                supertract_id = f"super_{tract_id}"
                if supertract_id not in supertracts:
                    supertracts[supertract_id] = Supertract(
                        supertract_id=supertract_id,
                        component_tract_ids=[tract_id],
                        year=year,
                        half_pairs_count=count
                    )
        
        return supertracts
    
    def _filter_pairs_by_year(self,
                            pairs_df: pd.DataFrame,
                            year: int) -> pd.DataFrame:
        """Filter pairs where either sale occurred in the specified year"""
        # Convert dates to datetime if needed
        df = pairs_df.copy()
        for col in ['first_sale_date', 'second_sale_date']:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        # Filter for pairs with sales in the target year
        mask = (
            (df['first_sale_date'].dt.year == year) |
            (df['second_sale_date'].dt.year == year)
        )
        
        return df[mask]
    
    def _count_half_pairs_by_tract(self, pairs_df: pd.DataFrame) -> Dict[str, int]:
        """
        Count half-pairs by tract
        
        A half-pair is one transaction in a repeat-sales pair.
        Each pair contributes 2 half-pairs (one for each sale).
        """
        tract_counts = defaultdict(int)
        
        for _, row in pairs_df.iterrows():
            tract_id = row['tract_id']
            # Each pair contributes 2 half-pairs
            tract_counts[tract_id] += 2
        
        return dict(tract_counts)
    
    def _build_adjacency_graph(self,
                             tract_ids: List[str],
                             tract_gdf: pd.DataFrame) -> nx.Graph:
        """Build graph of tract adjacencies"""
        G = nx.Graph()
        
        # Add all tracts as nodes
        G.add_nodes_from(tract_ids)
        
        # Add edges from provided adjacency data
        if self.adjacency_data:
            for tract_id, neighbors in self.adjacency_data.items():
                if tract_id in tract_ids:
                    for neighbor_id in neighbors:
                        if neighbor_id in tract_ids:
                            G.add_edge(tract_id, neighbor_id)
        else:
            # Compute adjacencies from geometries if not provided
            self._compute_geometric_adjacencies(G, tract_gdf, tract_ids)
        
        return G
    
    def _compute_geometric_adjacencies(self,
                                     graph: nx.Graph,
                                     tract_gdf: pd.DataFrame,
                                     tract_ids: List[str]):
        """Compute adjacencies based on geometric proximity"""
        # Get relevant tracts
        if 'tract_id' in tract_gdf.columns:
            relevant_gdf = tract_gdf[tract_gdf['tract_id'].isin(tract_ids)]
        else:
            # tract_id might be the index
            relevant_gdf = tract_gdf[tract_gdf.index.isin(tract_ids)]
        
        # For each tract, find neighbors within same CBSA
        for idx1, tract1 in relevant_gdf.iterrows():
            tract1_id = tract1.get('tract_id', idx1)
            cbsa1 = tract1['cbsa_id']
            
            # Find other tracts in same CBSA
            same_cbsa = relevant_gdf[relevant_gdf['cbsa_id'] == cbsa1]
            
            for idx2, tract2 in same_cbsa.iterrows():
                tract2_id = tract2.get('tract_id', idx2)
                
                if tract1_id != tract2_id:
                    # Check if geometries touch or are very close
                    if tract1['geometry'].touches(tract2['geometry']):
                        graph.add_edge(tract1_id, tract2_id)
                    elif tract1['geometry'].distance(tract2['geometry']) < 0.001:
                        # Very close (accounting for floating point precision)
                        graph.add_edge(tract1_id, tract2_id)
    
    def _aggregate_tracts(self,
                        insufficient_tracts: Dict[str, int],
                        all_counts: Dict[str, int],
                        graph: nx.Graph,
                        year: int) -> Dict[str, Supertract]:
        """
        Aggregate insufficient tracts into supertracts
        
        Uses greedy algorithm to combine adjacent tracts until
        minimum threshold is reached.
        """
        supertracts = {}
        assigned_tracts = set()
        
        # Sort insufficient tracts by count (ascending) to prioritize smallest
        sorted_tracts = sorted(
            insufficient_tracts.items(),
            key=lambda x: x[1]
        )
        
        for tract_id, count in sorted_tracts:
            if tract_id in assigned_tracts:
                continue
            
            # Start new supertract with this tract
            component_tracts = [tract_id]
            total_count = count
            assigned_tracts.add(tract_id)
            
            # Try to add adjacent tracts until threshold is met
            while total_count < self.min_half_pairs:
                best_neighbor = self._find_best_neighbor(
                    component_tracts,
                    assigned_tracts,
                    all_counts,
                    graph
                )
                
                if best_neighbor is None:
                    # No more neighbors available
                    break
                
                component_tracts.append(best_neighbor)
                total_count += all_counts.get(best_neighbor, 0)
                assigned_tracts.add(best_neighbor)
            
            # Create supertract if we have enough data
            if total_count >= self.min_half_pairs:
                supertract_id = f"super_{'_'.join(sorted(component_tracts)[:3])}"
                supertracts[supertract_id] = Supertract(
                    supertract_id=supertract_id,
                    component_tract_ids=sorted(component_tracts),
                    year=year,
                    half_pairs_count=total_count
                )
        
        return supertracts
    
    def _find_best_neighbor(self,
                          component_tracts: List[str],
                          assigned_tracts: Set[str],
                          all_counts: Dict[str, int],
                          graph: nx.Graph) -> Optional[str]:
        """
        Find best unassigned neighbor to add to supertract
        
        Prioritizes:
        1. Insufficient tracts (to help them reach threshold)
        2. Tracts with fewer half-pairs
        """
        candidates = []
        
        # Find all unassigned neighbors
        for tract_id in component_tracts:
            if tract_id in graph:
                for neighbor in graph.neighbors(tract_id):
                    if neighbor not in assigned_tracts:
                        count = all_counts.get(neighbor, 0)
                        is_insufficient = count < self.min_half_pairs
                        candidates.append((neighbor, count, is_insufficient))
        
        if not candidates:
            return None
        
        # Sort by: insufficient first, then by count (ascending)
        candidates.sort(key=lambda x: (not x[2], x[1]))
        
        return candidates[0][0]
    
    def validate_supertracts(self,
                           supertracts: Dict[str, Supertract],
                           pairs_df: pd.DataFrame,
                           year: int) -> Dict[str, Dict[str, any]]:
        """
        Validate constructed supertracts
        
        Returns validation report for each supertract
        """
        year_pairs = self._filter_pairs_by_year(pairs_df, year)
        validation_results = {}
        
        for supertract_id, supertract in supertracts.items():
            # Count actual half-pairs
            actual_count = 0
            for tract_id in supertract.component_tract_ids:
                tract_pairs = year_pairs[year_pairs['tract_id'] == tract_id]
                actual_count += len(tract_pairs) * 2
            
            validation_results[supertract_id] = {
                'component_tracts': len(supertract.component_tract_ids),
                'expected_half_pairs': supertract.half_pairs_count,
                'actual_half_pairs': actual_count,
                'meets_threshold': actual_count >= self.min_half_pairs,
                'discrepancy': abs(actual_count - supertract.half_pairs_count)
            }
        
        return validation_results
    
    def get_supertract_mapping(self,
                             supertracts: Dict[str, Supertract]) -> Dict[str, str]:
        """
        Get mapping from tract IDs to supertract IDs
        
        Returns:
            Dict mapping each tract ID to its supertract ID
        """
        mapping = {}
        
        for supertract_id, supertract in supertracts.items():
            for tract_id in supertract.component_tract_ids:
                mapping[tract_id] = supertract_id
        
        return mapping