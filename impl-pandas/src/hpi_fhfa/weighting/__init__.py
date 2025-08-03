"""Weighting and aggregation module"""

from .demographic_calculator import DemographicWeightCalculator
from .laspeyres import LaspeyresIndex
from .aggregator import GeographicAggregator
from .normalizer import WeightNormalizer

__all__ = [
    'DemographicWeightCalculator',
    'LaspeyresIndex',
    'GeographicAggregator',
    'WeightNormalizer'
]