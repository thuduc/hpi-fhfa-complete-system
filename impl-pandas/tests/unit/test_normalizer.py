"""Unit tests for weight normalizer"""

import pytest
import pandas as pd
import numpy as np

from hpi_fhfa.weighting import WeightNormalizer
from hpi_fhfa.models.weights import WeightSet, WeightType


class TestWeightNormalizer:
    """Test weight normalization and validation"""
    
    @pytest.fixture
    def sample_weight_set(self):
        """Create sample weight set"""
        return WeightSet(
            geography_id='C1',
            period=2020,
            weight_type=WeightType.VALUE,
            weights={
                'T001': 0.25,
                'T002': 0.50,
                'T003': 0.25
            }
        )
    
    @pytest.fixture
    def unnormalized_weight_set(self):
        """Create unnormalized weight set"""
        return WeightSet(
            geography_id='C1',
            period=2020,
            weight_type=WeightType.VALUE,
            weights={
                'T001': 1.0,
                'T002': 2.0,
                'T003': 1.0
            }
        )
    
    @pytest.fixture
    def weights_by_period(self):
        """Create weights across multiple periods"""
        return {
            2019: WeightSet(
                geography_id='C1',
                period=2019,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.30,
                    'T002': 0.45,
                    'T003': 0.25
                }
            ),
            2020: WeightSet(
                geography_id='C1',
                period=2020,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.25,
                    'T002': 0.50,
                    'T003': 0.25
                }
            ),
            2021: WeightSet(
                geography_id='C1',
                period=2021,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.20,
                    'T002': 0.55,
                    'T003': 0.25
                }
            )
        }
    
    def test_normalize_weights(self, unnormalized_weight_set):
        """Test weight normalization"""
        normalizer = WeightNormalizer()
        
        # Check unnormalized
        assert not unnormalized_weight_set.is_normalized
        
        # Normalize
        normalized = normalizer.normalize_weights(unnormalized_weight_set)
        
        # Check normalized
        assert normalized.is_normalized
        assert abs(sum(normalized.weights.values()) - 1.0) < 1e-6
        
        # Check proportions maintained
        assert normalized.weights['T002'] == 0.5  # 2/4
        assert normalized.weights['T001'] == 0.25  # 1/4
        assert normalized.weights['T003'] == 0.25  # 1/4
    
    def test_validate_valid_weight_set(self, sample_weight_set):
        """Test validation of valid weight set"""
        normalizer = WeightNormalizer()
        
        is_valid, issues = normalizer.validate_weight_set(sample_weight_set)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_unnormalized(self, unnormalized_weight_set):
        """Test validation catches unnormalized weights"""
        normalizer = WeightNormalizer()
        
        is_valid, issues = normalizer.validate_weight_set(unnormalized_weight_set)
        
        assert not is_valid
        assert any("not normalized" in issue for issue in issues)
    
    def test_validate_negative_weights(self):
        """Test validation catches negative weights"""
        weight_set = WeightSet(
            geography_id='C1',
            period=2020,
            weight_type=WeightType.VALUE,
            weights={
                'T001': -0.5,
                'T002': 1.5
            }
        )
        
        normalizer = WeightNormalizer()
        is_valid, issues = normalizer.validate_weight_set(weight_set)
        
        assert not is_valid
        assert any("Negative weights" in issue for issue in issues)
    
    def test_validate_empty_weights(self):
        """Test validation catches empty weights"""
        weight_set = WeightSet(
            geography_id='C1',
            period=2020,
            weight_type=WeightType.VALUE,
            weights={}
        )
        
        normalizer = WeightNormalizer()
        is_valid, issues = normalizer.validate_weight_set(weight_set)
        
        assert not is_valid
        assert any("empty" in issue for issue in issues)
    
    def test_validate_dominant_weight(self):
        """Test validation catches dominant weights"""
        weight_set = WeightSet(
            geography_id='C1',
            period=2020,
            weight_type=WeightType.VALUE,
            weights={
                'T001': 0.85,
                'T002': 0.10,
                'T003': 0.05
            }
        )
        
        normalizer = WeightNormalizer()
        is_valid, issues = normalizer.validate_weight_set(weight_set)
        
        assert not is_valid
        assert any("Dominant weights" in issue for issue in issues)
    
    def test_validate_consistency(self, weights_by_period):
        """Test consistency validation across periods"""
        normalizer = WeightNormalizer()
        
        is_consistent, issues = normalizer.validate_weight_consistency(weights_by_period)
        
        assert is_consistent
        assert len(issues) == 0
    
    def test_validate_missing_tracts(self):
        """Test detection of missing tracts across periods"""
        weights = {
            2019: WeightSet(
                geography_id='C1',
                period=2019,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.5,
                    'T002': 0.5
                }
            ),
            2020: WeightSet(
                geography_id='C1',
                period=2020,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.3,
                    'T002': 0.4,
                    'T003': 0.3  # New tract appears
                }
            ),
            2021: WeightSet(
                geography_id='C1',
                period=2021,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.4,
                    'T002': 0.6
                    # T003 disappears
                }
            )
        }
        
        normalizer = WeightNormalizer()
        is_consistent, issues = normalizer.validate_weight_consistency(weights)
        
        # Should find issue with missing T003 in 2021
        assert not is_consistent
        assert any("Missing weights" in issue for issue in issues)
    
    def test_validate_large_weight_changes(self):
        """Test detection of large weight changes"""
        weights = {
            2019: WeightSet(
                geography_id='C1',
                period=2019,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.8,
                    'T002': 0.2
                }
            ),
            2020: WeightSet(
                geography_id='C1',
                period=2020,
                weight_type=WeightType.VALUE,
                weights={
                    'T001': 0.2,  # Large decrease
                    'T002': 0.8   # Large increase
                }
            )
        }
        
        normalizer = WeightNormalizer()
        is_consistent, issues = normalizer.validate_weight_consistency(weights)
        
        assert not is_consistent
        assert any("Large weight change" in issue for issue in issues)
    
    def test_fill_missing_weights(self, sample_weight_set):
        """Test filling missing tract weights"""
        normalizer = WeightNormalizer()
        
        all_tracts = ['T001', 'T002', 'T003', 'T004', 'T005']
        
        filled = normalizer.fill_missing_weights(
            sample_weight_set,
            all_tracts,
            fill_value=0.0
        )
        
        # Should have all tracts
        assert len(filled.weights) == 5
        assert 'T004' in filled.weights
        assert 'T005' in filled.weights
        assert filled.weights['T004'] == 0.0
        assert filled.weights['T005'] == 0.0
        
        # Original weights unchanged
        assert filled.weights['T001'] == sample_weight_set.weights['T001']
    
    def test_fill_and_normalize(self, sample_weight_set):
        """Test filling and normalizing weights"""
        normalizer = WeightNormalizer()
        
        all_tracts = ['T001', 'T002', 'T003', 'T004']
        
        filled = normalizer.fill_missing_weights(
            sample_weight_set,
            all_tracts,
            fill_value=0.1  # Non-zero fill
        )
        
        # Should be normalized
        assert filled.is_normalized
        assert abs(sum(filled.weights.values()) - 1.0) < 1e-6
    
    def test_redistribute_missing_weights(self, sample_weight_set):
        """Test redistributing weights to available tracts"""
        normalizer = WeightNormalizer()
        
        # Only T001 and T002 have data
        available_tracts = {'T001', 'T002'}
        
        redistributed = normalizer.redistribute_missing_weights(
            sample_weight_set,
            available_tracts
        )
        
        assert redistributed is not None
        assert len(redistributed.weights) == 2
        assert 'T003' not in redistributed.weights
        assert redistributed.is_normalized
        
        # Weights should be rescaled
        # Original: T001=0.25, T002=0.50, total=0.75
        # New: T001=0.25/0.75=1/3, T002=0.50/0.75=2/3
        assert abs(redistributed.weights['T001'] - 1/3) < 1e-6
        assert abs(redistributed.weights['T002'] - 2/3) < 1e-6
    
    def test_redistribute_no_overlap(self, sample_weight_set):
        """Test redistribution with no overlapping tracts"""
        normalizer = WeightNormalizer()
        
        # No overlap
        available_tracts = {'T004', 'T005'}
        
        redistributed = normalizer.redistribute_missing_weights(
            sample_weight_set,
            available_tracts
        )
        
        assert redistributed is None
    
    def test_merge_weights_average(self):
        """Test merging weights with average strategy"""
        weights = [
            WeightSet('C1', 2020, WeightType.VALUE, {'T1': 0.6, 'T2': 0.4}),
            WeightSet('C1', 2020, WeightType.VALUE, {'T1': 0.4, 'T2': 0.6}),
            WeightSet('C1', 2020, WeightType.VALUE, {'T1': 0.5, 'T2': 0.5})
        ]
        
        normalizer = WeightNormalizer()
        merged = normalizer.merge_weight_sets(weights, 'average')
        
        assert merged.is_normalized
        assert abs(merged.weights['T1'] - 0.5) < 1e-6  # Average of 0.6, 0.4, 0.5
        assert abs(merged.weights['T2'] - 0.5) < 1e-6
    
    def test_merge_weights_sum(self):
        """Test merging weights with sum strategy"""
        weights = [
            WeightSet('C1', 2020, WeightType.VALUE, {'T1': 0.3, 'T2': 0.2}),
            WeightSet('C1', 2020, WeightType.VALUE, {'T1': 0.2, 'T2': 0.3})
        ]
        
        normalizer = WeightNormalizer()
        merged = normalizer.merge_weight_sets(weights, 'sum')
        
        assert merged.is_normalized
        # Sum then normalize: T1=0.5, T2=0.5, total=1.0
        assert abs(merged.weights['T1'] - 0.5) < 1e-6
        assert abs(merged.weights['T2'] - 0.5) < 1e-6
    
    def test_merge_weights_first(self):
        """Test merging weights with first strategy"""
        weights = [
            WeightSet('C1', 2020, WeightType.VALUE, {'T1': 0.7, 'T2': 0.3}),
            WeightSet('C1', 2020, WeightType.VALUE, {'T1': 0.4, 'T2': 0.6})
        ]
        
        normalizer = WeightNormalizer()
        merged = normalizer.merge_weight_sets(weights, 'first')
        
        # Should use first weight set
        assert merged.weights['T1'] == 0.7
        assert merged.weights['T2'] == 0.3
    
    def test_calculate_coverage(self, sample_weight_set):
        """Test effective coverage calculation"""
        normalizer = WeightNormalizer()
        
        # All tracts available
        all_available = {'T001', 'T002', 'T003'}
        coverage = normalizer.calculate_effective_coverage(sample_weight_set, all_available)
        assert coverage == 1.0
        
        # Only T001 and T002 available (75% of weight)
        partial_available = {'T001', 'T002'}
        coverage = normalizer.calculate_effective_coverage(sample_weight_set, partial_available)
        assert abs(coverage - 0.75) < 1e-6
        
        # No tracts available
        coverage = normalizer.calculate_effective_coverage(sample_weight_set, set())
        assert coverage == 0.0
    
    def test_adjustment_report(self):
        """Test creation of adjustment report"""
        original = {
            'C1': WeightSet('C1', 2020, WeightType.VALUE, 
                          {'T1': 0.3, 'T2': 0.4, 'T3': 0.3}),
            'C2': WeightSet('C2', 2020, WeightType.VALUE,
                          {'T4': 0.5, 'T5': 0.5})
        }
        
        adjusted = {
            'C1': WeightSet('C1', 2020, WeightType.VALUE,
                          {'T1': 0.5, 'T2': 0.5}),  # T3 dropped
            'C2': WeightSet('C2', 2020, WeightType.VALUE,
                          {'T4': 0.6, 'T5': 0.4})   # Reweighted
        }
        
        normalizer = WeightNormalizer()
        report = normalizer.create_adjustment_report(original, adjusted)
        
        assert len(report) == 2
        assert 'geography_id' in report.columns
        assert 'tracts_dropped' in report.columns
        
        # Check C1 had 1 tract dropped
        c1_report = report[report['geography_id'] == 'C1']
        assert c1_report['tracts_dropped'].iloc[0] == 1
        assert c1_report['original_tract_count'].iloc[0] == 3
        assert c1_report['adjusted_tract_count'].iloc[0] == 2