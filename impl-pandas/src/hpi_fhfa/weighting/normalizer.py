"""Weight normalization and validation utilities"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from ..models.weights import WeightSet, WeightType


class WeightNormalizer:
    """
    Normalize and validate weight sets for index aggregation
    
    This class ensures weights are properly normalized, handles missing
    data, and validates weight consistency across time periods.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize normalizer
        
        Args:
            tolerance: Tolerance for normalization checks
        """
        self.tolerance = tolerance
        
    def normalize_weights(self, weight_set: WeightSet) -> WeightSet:
        """
        Normalize weights to sum to 1
        
        Args:
            weight_set: WeightSet to normalize
            
        Returns:
            Normalized WeightSet (modifies in place and returns)
        """
        weight_set.normalize()
        return weight_set
    
    def validate_weight_set(self, weight_set: WeightSet) -> Tuple[bool, List[str]]:
        """
        Validate a single weight set
        
        Args:
            weight_set: WeightSet to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check if normalized
        if not weight_set.is_normalized:
            total = sum(weight_set.weights.values())
            issues.append(f"Weights not normalized: sum = {total:.6f}")
        
        # Check for negative weights
        negative_weights = [tid for tid, w in weight_set.weights.items() if w < 0]
        if negative_weights:
            issues.append(f"Negative weights found for tracts: {negative_weights[:5]}")
        
        # Check for empty weights
        if not weight_set.weights:
            issues.append("Weight set is empty")
        
        # Check for zero total weight
        total = sum(weight_set.weights.values())
        if abs(total) < self.tolerance:
            issues.append("Total weight is zero or near zero")
        
        # Check weight range
        if weight_set.weights:
            min_weight = min(weight_set.weights.values())
            max_weight = max(weight_set.weights.values())
            if max_weight > 0.5:  # No single tract should dominate
                dominant_tracts = [tid for tid, w in weight_set.weights.items() if w > 0.5]
                issues.append(f"Dominant weights found: {dominant_tracts[0]} = {max_weight:.3f}")
        
        return len(issues) == 0, issues
    
    def validate_weight_consistency(self,
                                  weights_by_period: Dict[int, WeightSet]) -> Tuple[bool, List[str]]:
        """
        Validate consistency of weights across time periods
        
        Args:
            weights_by_period: Dict mapping years to WeightSets
            
        Returns:
            Tuple of (is_consistent, list_of_issues)
        """
        issues = []
        
        if not weights_by_period:
            issues.append("No weights provided")
            return False, issues
        
        # Check tract coverage consistency
        periods = sorted(weights_by_period.keys())
        tract_sets = [set(weights_by_period[p].weights.keys()) for p in periods]
        
        # Find tracts that appear and disappear
        all_tracts = set().union(*tract_sets)
        
        for i, period in enumerate(periods):
            current_tracts = tract_sets[i]
            missing = all_tracts - current_tracts
            
            if missing and i > 0:  # Allow missing in first period
                issues.append(f"Year {period}: Missing weights for {len(missing)} tracts")
        
        # Check weight stability
        for i in range(1, len(periods)):
            prev_period = periods[i-1]
            curr_period = periods[i]
            
            prev_weights = weights_by_period[prev_period].weights
            curr_weights = weights_by_period[curr_period].weights
            
            # Compare weights for common tracts
            common_tracts = set(prev_weights.keys()) & set(curr_weights.keys())
            
            if common_tracts:
                # Calculate maximum weight change
                max_change = 0.0
                max_change_tract = None
                
                for tract in common_tracts:
                    change = abs(curr_weights[tract] - prev_weights[tract])
                    if change > max_change:
                        max_change = change
                        max_change_tract = tract
                
                # Flag large changes (>50% relative change)
                if max_change > 0.5 * prev_weights.get(max_change_tract, 1.0):
                    issues.append(
                        f"Large weight change {prev_period}->{curr_period}: "
                        f"{max_change_tract} changed by {max_change:.3f}"
                    )
        
        return len(issues) == 0, issues
    
    def fill_missing_weights(self,
                           weight_set: WeightSet,
                           all_tract_ids: List[str],
                           fill_value: float = 0.0) -> WeightSet:
        """
        Fill missing tract weights with specified value
        
        Args:
            weight_set: Original weight set
            all_tract_ids: Complete list of tract IDs
            fill_value: Value to use for missing tracts
            
        Returns:
            New WeightSet with filled values
        """
        # Copy existing weights
        filled_weights = weight_set.weights.copy()
        
        # Add missing tracts
        for tract_id in all_tract_ids:
            if tract_id not in filled_weights:
                filled_weights[tract_id] = fill_value
        
        # Create new weight set
        filled_set = WeightSet(
            geography_id=weight_set.geography_id,
            period=weight_set.period,
            weight_type=weight_set.weight_type,
            weights=filled_weights
        )
        
        # Normalize if needed
        if fill_value > 0:
            filled_set.normalize()
        
        return filled_set
    
    def redistribute_missing_weights(self,
                                   weight_set: WeightSet,
                                   available_tract_ids: Set[str]) -> Optional[WeightSet]:
        """
        Redistribute weights from missing tracts to available ones
        
        Args:
            weight_set: Original weight set
            available_tract_ids: Set of tract IDs with data
            
        Returns:
            New WeightSet with redistributed weights, or None if no overlap
        """
        # Find available tracts with weights
        available_weights = {
            tid: w for tid, w in weight_set.weights.items()
            if tid in available_tract_ids
        }
        
        if not available_weights:
            return None
        
        # Create new normalized weight set
        redistributed = WeightSet(
            geography_id=weight_set.geography_id,
            period=weight_set.period,
            weight_type=weight_set.weight_type,
            weights=available_weights
        )
        
        redistributed.normalize()
        return redistributed
    
    def merge_weight_sets(self,
                        weight_sets: List[WeightSet],
                        merge_strategy: str = 'average') -> WeightSet:
        """
        Merge multiple weight sets into one
        
        Args:
            weight_sets: List of WeightSets to merge
            merge_strategy: 'average', 'sum', or 'first'
            
        Returns:
            Merged WeightSet
        """
        if not weight_sets:
            raise ValueError("No weight sets to merge")
        
        if len(weight_sets) == 1:
            return weight_sets[0]
        
        # Collect all weights by tract
        weights_by_tract = defaultdict(list)
        for ws in weight_sets:
            for tract_id, weight in ws.weights.items():
                weights_by_tract[tract_id].append(weight)
        
        # Merge based on strategy
        merged_weights = {}
        
        if merge_strategy == 'average':
            for tract_id, weight_list in weights_by_tract.items():
                merged_weights[tract_id] = np.mean(weight_list)
                
        elif merge_strategy == 'sum':
            for tract_id, weight_list in weights_by_tract.items():
                merged_weights[tract_id] = sum(weight_list)
                
        elif merge_strategy == 'first':
            merged_weights = weight_sets[0].weights.copy()
            
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Create merged weight set
        merged = WeightSet(
            geography_id=weight_sets[0].geography_id,
            period=weight_sets[0].period,
            weight_type=weight_sets[0].weight_type,
            weights=merged_weights
        )
        
        # Normalize
        merged.normalize()
        return merged
    
    def calculate_effective_coverage(self,
                                   weight_set: WeightSet,
                                   available_tract_ids: Set[str]) -> float:
        """
        Calculate effective weight coverage
        
        Args:
            weight_set: WeightSet to analyze
            available_tract_ids: Set of tract IDs with data
            
        Returns:
            Fraction of total weight covered by available tracts
        """
        if not weight_set.weights:
            return 0.0
        
        covered_weight = sum(
            w for tid, w in weight_set.weights.items()
            if tid in available_tract_ids
        )
        
        total_weight = sum(weight_set.weights.values())
        
        if total_weight > 0:
            return covered_weight / total_weight
        else:
            return 0.0
    
    def create_adjustment_report(self,
                               original_weights: Dict[str, WeightSet],
                               adjusted_weights: Dict[str, WeightSet]) -> pd.DataFrame:
        """
        Create report showing weight adjustments
        
        Args:
            original_weights: Original weight sets
            adjusted_weights: Adjusted weight sets
            
        Returns:
            DataFrame with adjustment details
        """
        report_data = []
        
        for geo_id in original_weights:
            if geo_id not in adjusted_weights:
                continue
                
            orig = original_weights[geo_id]
            adj = adjusted_weights[geo_id]
            
            # Calculate changes
            orig_total = sum(orig.weights.values())
            adj_total = sum(adj.weights.values())
            
            orig_count = len(orig.weights)
            adj_count = len(adj.weights)
            
            # Find dropped tracts
            dropped = set(orig.weights.keys()) - set(adj.weights.keys())
            
            report_data.append({
                'geography_id': geo_id,
                'period': orig.period,
                'weight_type': orig.weight_type.value,
                'original_tract_count': orig_count,
                'adjusted_tract_count': adj_count,
                'tracts_dropped': len(dropped),
                'original_total_weight': orig_total,
                'adjusted_total_weight': adj_total,
                'normalized': adj.is_normalized
            })
        
        return pd.DataFrame(report_data)