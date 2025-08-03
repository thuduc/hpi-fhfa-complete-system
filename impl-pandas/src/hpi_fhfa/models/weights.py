"""Weight type definitions"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, List
from abc import ABC, abstractmethod


class WeightType(Enum):
    """Available weight types for index aggregation"""
    SAMPLE = "sample"
    VALUE = "value"
    UNIT = "unit"
    UPB = "upb"
    COLLEGE = "college"
    NON_WHITE = "non_white"


@dataclass
class DemographicData:
    """Demographic data for weight calculations"""
    tract_id: str
    year: int
    housing_units: int
    median_value: float
    college_share: float  # Share of population with college degree
    non_white_share: float  # Share of non-white population
    upb_total: float  # Total unpaid principal balance
    
    def get_weight_value(self, weight_type: WeightType) -> float:
        """Get value for specific weight type"""
        if weight_type == WeightType.VALUE:
            return self.housing_units * self.median_value
        elif weight_type == WeightType.UNIT:
            return self.housing_units
        elif weight_type == WeightType.UPB:
            return self.upb_total
        elif weight_type == WeightType.COLLEGE:
            return self.college_share
        elif weight_type == WeightType.NON_WHITE:
            return self.non_white_share
        else:
            raise ValueError(f"Cannot get demographic value for {weight_type}")


@dataclass
class WeightSet:
    """Collection of weights for a geographic area"""
    geography_id: str
    period: int
    weight_type: WeightType
    weights: Dict[str, float]  # tract_id -> weight
    
    @property
    def is_normalized(self) -> bool:
        """Check if weights sum to 1"""
        total = sum(self.weights.values())
        return abs(total - 1.0) < 1e-6
    
    def normalize(self) -> None:
        """Normalize weights to sum to 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}


class WeightCalculator(ABC):
    """Abstract base class for weight calculators"""
    
    @abstractmethod
    def calculate_weights(self,
                        weight_type: WeightType,
                        geography_id: str,
                        period: int) -> Optional[WeightSet]:
        """Calculate weights for given geography and period"""
        pass


class SampleWeightCalculator(WeightCalculator):
    """Simple calculator that assigns equal weights"""
    
    def __init__(self, tract_ids_by_geography: Optional[Dict[str, List[str]]] = None):
        """
        Initialize calculator
        
        Args:
            tract_ids_by_geography: Dict mapping geography IDs to list of tract IDs
        """
        self.tract_ids_by_geography = tract_ids_by_geography or {}
    
    def calculate_weights(self,
                        weight_type: WeightType,
                        geography_id: str,
                        period: int) -> Optional[WeightSet]:
        """Calculate equal weights for all tracts in geography"""
        if geography_id not in self.tract_ids_by_geography:
            return None
        
        tract_ids = self.tract_ids_by_geography[geography_id]
        if not tract_ids:
            return None
        
        # Equal weights
        weight = 1.0 / len(tract_ids)
        weights = {tract_id: weight for tract_id in tract_ids}
        
        return WeightSet(
            geography_id=geography_id,
            period=period,
            weight_type=weight_type,
            weights=weights
        )