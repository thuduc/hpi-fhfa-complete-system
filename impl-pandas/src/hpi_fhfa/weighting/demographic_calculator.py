"""Demographic-based weight calculators for all 6 weight types"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from collections import defaultdict

from ..models.weights import (
    WeightCalculator, WeightSet, WeightType, DemographicData
)


class DemographicWeightCalculator(WeightCalculator):
    """
    Calculate weights based on demographic data for all weight types
    
    This calculator handles all 6 FHFA weight types:
    - Sample: Equal weights (1/n)
    - Value: Housing units × median value (Laspeyres)
    - Unit: Number of housing units
    - UPB: Unpaid principal balance
    - College: Share of college-educated population
    - Non-White: Share of non-white population
    """
    
    def __init__(self, demographic_data: Optional[pd.DataFrame] = None):
        """
        Initialize calculator with demographic data
        
        Args:
            demographic_data: DataFrame with columns:
                - tract_id: Census tract ID
                - year: Year of data
                - cbsa_id: CBSA ID
                - housing_units: Number of housing units
                - median_value: Median home value
                - college_share: Share with college degree (0-1)
                - non_white_share: Share of non-white population (0-1)
                - upb_total: Total unpaid principal balance
        """
        self.demographic_data = demographic_data
        self._data_by_year_geo = self._organize_data() if demographic_data is not None else {}
        
    def _organize_data(self) -> Dict[tuple, Dict[str, DemographicData]]:
        """Organize demographic data by (year, geography) for efficient lookup"""
        organized = defaultdict(dict)
        
        if self.demographic_data is None:
            return organized
            
        for _, row in self.demographic_data.iterrows():
            demo = DemographicData(
                tract_id=row['tract_id'],
                year=row['year'],
                housing_units=row['housing_units'],
                median_value=row['median_value'],
                college_share=row['college_share'],
                non_white_share=row['non_white_share'],
                upb_total=row['upb_total']
            )
            
            # Store by CBSA
            key = (row['year'], row['cbsa_id'])
            organized[key][row['tract_id']] = demo
            
            # Also store by national
            national_key = (row['year'], 'national')
            organized[national_key][row['tract_id']] = demo
            
        return dict(organized)
    
    def calculate_weights(self,
                        weight_type: WeightType,
                        geography_id: str,
                        period: int) -> Optional[WeightSet]:
        """
        Calculate weights for given geography and period
        
        Args:
            weight_type: Type of weights to calculate
            geography_id: CBSA ID or 'national'
            period: Year for which to calculate weights
            
        Returns:
            WeightSet or None if no data available
        """
        key = (period, geography_id)
        if key not in self._data_by_year_geo:
            return None
            
        tract_data = self._data_by_year_geo[key]
        if not tract_data:
            return None
        
        # Calculate raw weights based on type
        if weight_type == WeightType.SAMPLE:
            # Equal weights
            raw_weights = {tract_id: 1.0 for tract_id in tract_data}
        else:
            # Demographic-based weights
            raw_weights = {}
            for tract_id, demo in tract_data.items():
                weight = demo.get_weight_value(weight_type)
                raw_weights[tract_id] = weight
        
        # Normalize weights to sum to 1
        total = sum(raw_weights.values())
        if total <= 0:
            return None
            
        normalized_weights = {k: v/total for k, v in raw_weights.items()}
        
        return WeightSet(
            geography_id=geography_id,
            period=period,
            weight_type=weight_type,
            weights=normalized_weights
        )
    
    def set_demographic_data(self, demographic_data: pd.DataFrame) -> None:
        """Update demographic data"""
        self.demographic_data = demographic_data
        self._data_by_year_geo = self._organize_data()
    
    def add_demographic_data(self, new_data: pd.DataFrame) -> None:
        """Add additional demographic data"""
        if self.demographic_data is None:
            self.demographic_data = new_data
        else:
            self.demographic_data = pd.concat([self.demographic_data, new_data], ignore_index=True)
        self._data_by_year_geo = self._organize_data()
    
    def get_available_periods(self, geography_id: str) -> List[int]:
        """Get list of years with data for given geography"""
        periods = []
        for (year, geo_id), _ in self._data_by_year_geo.items():
            if geo_id == geography_id:
                periods.append(year)
        return sorted(periods)
    
    def get_available_geographies(self, period: int) -> List[str]:
        """Get list of geographies with data for given period"""
        geographies = []
        for (year, geo_id), _ in self._data_by_year_geo.items():
            if year == period:
                geographies.append(geo_id)
        return sorted(set(geographies))
    
    def validate_weights(self, weight_set: WeightSet) -> bool:
        """
        Validate that weights are properly normalized and complete
        
        Args:
            weight_set: WeightSet to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check normalization
        if not weight_set.is_normalized:
            return False
            
        # Check all weights are non-negative
        if any(w < 0 for w in weight_set.weights.values()):
            return False
            
        # Check we have weights for all tracts in geography
        key = (weight_set.period, weight_set.geography_id)
        if key in self._data_by_year_geo:
            expected_tracts = set(self._data_by_year_geo[key].keys())
            actual_tracts = set(weight_set.weights.keys())
            if expected_tracts != actual_tracts:
                return False
                
        return True
    
    def generate_synthetic_demographic_data(self,
                                          tract_gdf: pd.DataFrame,
                                          years: List[int],
                                          seed: int = 42) -> pd.DataFrame:
        """
        Generate synthetic demographic data for testing
        
        Args:
            tract_gdf: GeoDataFrame with tract geometries
            years: List of years to generate data for
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with synthetic demographic data
        """
        np.random.seed(seed)
        
        data = []
        for year in years:
            for _, tract in tract_gdf.iterrows():
                tract_id = tract.name if tract_gdf.index.name == 'tract_id' else tract['tract_id']
                
                # Generate realistic synthetic values
                housing_units = np.random.randint(500, 5000)
                median_value = np.random.uniform(100000, 800000)
                college_share = np.random.beta(2, 5)  # Skewed towards lower values
                non_white_share = np.random.beta(3, 7)  # Varies by tract
                upb_total = housing_units * median_value * np.random.uniform(0.4, 0.8)
                
                data.append({
                    'tract_id': tract_id,
                    'year': year,
                    'cbsa_id': tract['cbsa_id'],
                    'housing_units': housing_units,
                    'median_value': median_value,
                    'college_share': college_share,
                    'non_white_share': non_white_share,
                    'upb_total': upb_total
                })
        
        return pd.DataFrame(data)