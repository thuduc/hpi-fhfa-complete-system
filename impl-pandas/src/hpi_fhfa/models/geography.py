"""Geographic data models"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import geopandas as gpd
from shapely.geometry import Point, Polygon


@dataclass
class Tract:
    """Census tract representation"""
    tract_id: str
    cbsa_id: str
    state: str
    county: str
    geometry: Polygon
    centroid: Point
    
    @property
    def fips_code(self) -> str:
        """Get full FIPS code for tract"""
        # Assuming tract_id format: state(2) + county(3) + tract(6)
        return self.tract_id
    
    def distance_to(self, other: 'Tract') -> float:
        """Calculate distance to another tract (in degrees)"""
        return self.centroid.distance(other.centroid)


@dataclass
class CBSA:
    """Core-Based Statistical Area"""
    cbsa_id: str
    name: str
    state: str
    tract_ids: List[str]
    
    @property
    def tract_count(self) -> int:
        return len(self.tract_ids)


@dataclass
class Supertract:
    """Aggregation of tracts for sufficient transaction volume"""
    supertract_id: str
    component_tract_ids: List[str]
    year: int
    half_pairs_count: int
    
    @property
    def is_single_tract(self) -> bool:
        return len(self.component_tract_ids) == 1
    
    def contains_tract(self, tract_id: str) -> bool:
        return tract_id in self.component_tract_ids


def create_tract_gdf(tracts: List[Tract]) -> gpd.GeoDataFrame:
    """Create GeoDataFrame from list of Tract objects"""
    data = []
    for tract in tracts:
        data.append({
            'tract_id': tract.tract_id,
            'cbsa_id': tract.cbsa_id,
            'state': tract.state,
            'county': tract.county,
            'geometry': tract.geometry,
            'centroid_x': tract.centroid.x,
            'centroid_y': tract.centroid.y
        })
    
    gdf = gpd.GeoDataFrame(data, geometry='geometry')
    gdf.set_index('tract_id', inplace=True)
    return gdf


def create_cbsa_mapping(cbsas: List[CBSA]) -> dict:
    """Create tract to CBSA mapping"""
    mapping = {}
    for cbsa in cbsas:
        for tract_id in cbsa.tract_ids:
            mapping[tract_id] = cbsa.cbsa_id
    return mapping