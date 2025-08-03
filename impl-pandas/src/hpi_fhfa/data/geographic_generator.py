"""Generate synthetic geographic data for testing"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import List, Dict, Tuple
import random


class GeographicDataGenerator:
    """Generate synthetic geographic data for tracts and CBSAs"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_tract_geometries(self,
                                num_tracts: int,
                                bounds: Tuple[float, float, float, float] = (-125, 25, -65, 50)
                                ) -> gpd.GeoDataFrame:
        """
        Generate synthetic tract geometries
        
        Args:
            num_tracts: Number of tracts to generate
            bounds: (min_lon, min_lat, max_lon, max_lat) for USA
        """
        min_lon, min_lat, max_lon, max_lat = bounds
        
        # Generate tract data
        tracts = []
        
        # Create clusters to simulate CBSAs
        n_clusters = int(np.sqrt(num_tracts))
        cluster_centers = []
        
        for i in range(n_clusters):
            # Random cluster center
            center_lon = np.random.uniform(min_lon + 5, max_lon - 5)
            center_lat = np.random.uniform(min_lat + 5, max_lat - 5)
            cluster_centers.append((center_lon, center_lat))
        
        tract_id = 0
        for cluster_idx, (center_lon, center_lat) in enumerate(cluster_centers):
            # Number of tracts in this cluster
            if cluster_idx == len(cluster_centers) - 1:
                n_tracts_cluster = num_tracts - tract_id
            else:
                n_tracts_cluster = num_tracts // n_clusters
            
            # Generate tracts around cluster center
            for i in range(n_tracts_cluster):
                # Offset from center (simulate urban density)
                if i < n_tracts_cluster * 0.3:  # Urban core
                    radius = np.random.exponential(0.05)
                else:  # Suburban
                    radius = np.random.exponential(0.2)
                
                angle = np.random.uniform(0, 2 * np.pi)
                tract_lon = center_lon + radius * np.cos(angle)
                tract_lat = center_lat + radius * np.sin(angle)
                
                # Create a small polygon for the tract
                size = np.random.uniform(0.01, 0.05)
                polygon = self._create_tract_polygon(tract_lon, tract_lat, size)
                
                # Generate tract ID (state + county + tract)
                state_code = f"{(cluster_idx % 50) + 1:02d}"
                county_code = f"{(cluster_idx % 10) + 1:03d}"
                tract_code = f"{i + 1:06d}"
                
                tracts.append({
                    'tract_id': f"{state_code}{county_code}{tract_code}",
                    'state': state_code,
                    'county': county_code,
                    'cbsa_id': f"{10000 + cluster_idx * 20}",
                    'geometry': polygon,
                    'centroid': Point(tract_lon, tract_lat),
                    'population': np.random.randint(1000, 10000),
                    'area_sqmi': size * size * 1000  # Rough approximation
                })
                
                tract_id += 1
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(tracts, geometry='geometry')
        gdf['centroid_x'] = gdf['centroid'].apply(lambda p: p.x)
        gdf['centroid_y'] = gdf['centroid'].apply(lambda p: p.y)
        
        return gdf
    
    def _create_tract_polygon(self, 
                            center_lon: float,
                            center_lat: float,
                            size: float) -> Polygon:
        """Create a simple rectangular polygon for a tract"""
        # Create a rectangle around the center point
        coords = [
            (center_lon - size/2, center_lat - size/2),
            (center_lon + size/2, center_lat - size/2),
            (center_lon + size/2, center_lat + size/2),
            (center_lon - size/2, center_lat + size/2),
            (center_lon - size/2, center_lat - size/2)  # Close the polygon
        ]
        
        # Add some randomness to make it less regular
        perturbed_coords = []
        for i, (lon, lat) in enumerate(coords[:-1]):  # Don't perturb the closing point
            if i > 0:  # Don't perturb the first point
                lon += np.random.normal(0, size * 0.1)
                lat += np.random.normal(0, size * 0.1)
            perturbed_coords.append((lon, lat))
        perturbed_coords.append(perturbed_coords[0])  # Close the polygon
        
        return Polygon(perturbed_coords)
    
    def generate_cbsa_data(self, 
                         tract_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate CBSA metadata from tract data"""
        cbsa_data = []
        
        # Get unique CBSAs
        cbsa_groups = tract_gdf.groupby('cbsa_id')
        
        # Major US metro names for realistic data
        metro_names = [
            "New York-Newark-Jersey City, NY-NJ-PA",
            "Los Angeles-Long Beach-Anaheim, CA",
            "Chicago-Naperville-Elgin, IL-IN-WI",
            "Dallas-Fort Worth-Arlington, TX",
            "Houston-The Woodlands-Sugar Land, TX",
            "Washington-Arlington-Alexandria, DC-VA-MD-WV",
            "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD",
            "Miami-Fort Lauderdale-Pompano Beach, FL",
            "Atlanta-Sandy Springs-Alpharetta, GA",
            "Boston-Cambridge-Newton, MA-NH",
            "Phoenix-Mesa-Chandler, AZ",
            "San Francisco-Oakland-Berkeley, CA",
            "Riverside-San Bernardino-Ontario, CA",
            "Detroit-Warren-Dearborn, MI",
            "Seattle-Tacoma-Bellevue, WA",
            "Minneapolis-St. Paul-Bloomington, MN-WI",
            "San Diego-Chula Vista-Carlsbad, CA",
            "Tampa-St. Petersburg-Clearwater, FL",
            "Denver-Aurora-Lakewood, CO",
            "Portland-Vancouver-Hillsboro, OR-WA"
        ]
        
        for i, (cbsa_id, group) in enumerate(cbsa_groups):
            # Get state from most common state in CBSA
            primary_state = group['state'].mode()[0]
            
            # Use real metro name if available, otherwise generate
            if i < len(metro_names):
                name = metro_names[i]
            else:
                name = f"Metro Area {i + 1}, {primary_state}"
            
            cbsa_data.append({
                'cbsa_id': cbsa_id,
                'name': name,
                'state': primary_state,
                'tract_count': len(group),
                'total_population': group['population'].sum(),
                'total_area_sqmi': group['area_sqmi'].sum()
            })
        
        return pd.DataFrame(cbsa_data)
    
    def generate_tract_adjacency(self,
                               tract_gdf: gpd.GeoDataFrame,
                               max_neighbors: int = 10) -> Dict[str, List[str]]:
        """Generate tract adjacency relationships"""
        adjacency = {}
        
        # For each tract, find nearby tracts
        for idx, tract in tract_gdf.iterrows():
            tract_id = tract['tract_id']
            
            # Calculate distances to all other tracts in same CBSA
            same_cbsa = tract_gdf[tract_gdf['cbsa_id'] == tract['cbsa_id']]
            
            if len(same_cbsa) <= 1:
                adjacency[tract_id] = []
                continue
            
            # Calculate distances using centroids
            distances = same_cbsa['centroid'].apply(
                lambda x: tract['centroid'].distance(x)
            )
            
            # Sort by distance and get nearest neighbors (excluding self)
            sorted_indices = distances.sort_values().index
            neighbors = []
            
            for neighbor_idx in sorted_indices:
                if neighbor_idx != idx:
                    neighbors.append(same_cbsa.loc[neighbor_idx, 'tract_id'])
                if len(neighbors) >= max_neighbors:
                    break
            
            adjacency[tract_id] = neighbors
        
        return adjacency
    
    def generate_complete_geographic_data(self,
                                        num_tracts: int = 1000,
                                        bounds: Tuple[float, float, float, float] = (-125, 25, -65, 50)
                                        ) -> Dict[str, any]:
        """
        Generate complete geographic dataset
        
        Returns dict with:
        - 'tracts': GeoDataFrame of tract geometries
        - 'cbsas': DataFrame of CBSA metadata  
        - 'adjacency': Dict of tract adjacency relationships
        """
        print(f"Generating geographic data for {num_tracts} tracts...")
        
        # Generate tract geometries
        print("Creating tract geometries...")
        tract_gdf = self.generate_tract_geometries(num_tracts, bounds)
        
        # Generate CBSA data
        print("Creating CBSA metadata...")
        cbsa_df = self.generate_cbsa_data(tract_gdf)
        
        # Generate adjacency relationships
        print("Computing tract adjacencies...")
        adjacency = self.generate_tract_adjacency(tract_gdf)
        
        print(f"Generated {len(tract_gdf)} tracts in {len(cbsa_df)} CBSAs")
        
        return {
            'tracts': tract_gdf,
            'cbsas': cbsa_df,
            'adjacency': adjacency
        }
    
    def export_to_files(self,
                       geographic_data: Dict[str, any],
                       output_dir: str) -> None:
        """Export geographic data to files"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tract geometries as GeoJSON
        geographic_data['tracts'].to_file(
            os.path.join(output_dir, 'tracts.geojson'),
            driver='GeoJSON'
        )
        
        # Save CBSA data as CSV
        geographic_data['cbsas'].to_csv(
            os.path.join(output_dir, 'cbsas.csv'),
            index=False
        )
        
        # Save adjacency as JSON
        with open(os.path.join(output_dir, 'tract_adjacency.json'), 'w') as f:
            json.dump(geographic_data['adjacency'], f, indent=2)
        
        print(f"Geographic data exported to {output_dir}/")
    
    def create_sample_demographic_weights(self,
                                        tract_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Create sample demographic data for weight calculations"""
        demographic_data = []
        
        for _, tract in tract_gdf.iterrows():
            # Generate correlated demographic variables
            urbanicity = np.random.uniform(0, 1)  # 0 = rural, 1 = urban
            
            # Urban areas tend to have higher college education rates
            college_share = np.clip(
                np.random.beta(2 + urbanicity * 3, 5 - urbanicity * 2),
                0, 1
            )
            
            # Generate other demographics
            demographic_data.append({
                'tract_id': tract['tract_id'],
                'year': 2010,  # Static weights from 2010 Census
                'housing_units': int(tract['population'] / 2.5),
                'median_value': 150000 + 350000 * urbanicity + np.random.normal(0, 50000),
                'college_share': college_share,
                'non_white_share': np.clip(np.random.beta(2, 3), 0, 1),
                'upb_total': tract['population'] * np.random.uniform(50000, 200000)
            })
        
        return pd.DataFrame(demographic_data)