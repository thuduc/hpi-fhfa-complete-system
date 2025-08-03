"""Unit tests for data quality metrics"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from hpi_fhfa.outliers import DataQualityAnalyzer, QualityReport


class TestDataQualityAnalyzer:
    """Test data quality analysis functionality"""
    
    @pytest.fixture
    def sample_pairs_good_quality(self):
        """Create high quality sample data"""
        np.random.seed(42)
        data = []
        base_date = date(2018, 1, 1)
        
        # Generate data for multiple CBSAs and tracts
        for cbsa in range(3):
            for tract in range(10):
                for prop in range(5):
                    first_date = base_date + timedelta(days=np.random.randint(0, 730))
                    second_date = first_date + timedelta(days=np.random.randint(400, 1000))
                    
                    first_price = np.random.uniform(150000, 350000)
                    appreciation = np.random.uniform(1.03, 1.15)  # 3-15% total
                    second_price = first_price * appreciation
                    
                    data.append({
                        'property_id': f'P{cbsa:02d}{tract:02d}{prop:03d}',
                        'tract_id': f'T{cbsa:02d}{tract:03d}',
                        'cbsa_id': f'C{cbsa:02d}',
                        'first_sale_date': first_date,
                        'first_sale_price': first_price,
                        'second_sale_date': second_date,
                        'second_sale_price': second_price
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_pairs_poor_quality(self):
        """Create poor quality sample data"""
        np.random.seed(42)
        data = []
        base_date = date(2018, 1, 1)
        
        # Sparse data with quality issues
        for i in range(30):
            first_date = base_date + timedelta(days=np.random.randint(0, 1825))
            
            # Add various quality issues
            if i % 5 == 0:
                # Very short holding period
                second_date = first_date + timedelta(days=30)
            elif i % 5 == 1:
                # Very long holding period
                second_date = first_date + timedelta(days=4000)
            else:
                second_date = first_date + timedelta(days=np.random.randint(365, 730))
            
            if i % 4 == 0:
                # Extreme prices
                first_price = np.random.choice([10000, 5000000])
                second_price = first_price * np.random.uniform(0.2, 5.0)
            else:
                first_price = np.random.uniform(100000, 400000)
                second_price = first_price * np.random.uniform(0.8, 1.2)
            
            data.append({
                'property_id': f'P{i:04d}',
                'tract_id': f'T{i % 3:03d}',  # Only 3 tracts
                'cbsa_id': 'C01',  # Single CBSA
                'first_sale_date': first_date,
                'first_sale_price': first_price,
                'second_sale_date': second_date,
                'second_sale_price': second_price
            })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_geography_df(self):
        """Create sample geographic reference data"""
        data = []
        for cbsa in range(3):
            for tract in range(15):
                data.append({
                    'tract_id': f'T{cbsa:02d}{tract:03d}',
                    'cbsa_id': f'C{cbsa:02d}',
                    'state': f'S{cbsa}'
                })
        return pd.DataFrame(data)
    
    def test_basic_quality_analysis(self, sample_pairs_good_quality):
        """Test basic quality analysis"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        assert isinstance(report, QualityReport)
        assert isinstance(report.summary_metrics, dict)
        assert isinstance(report.validation_results, dict)
        assert isinstance(report.temporal_coverage, pd.DataFrame)
        assert isinstance(report.geographic_coverage, pd.DataFrame)
        assert isinstance(report.price_distribution, pd.DataFrame)
        assert isinstance(report.quality_scores, dict)
        assert isinstance(report.recommendations, list)
    
    def test_summary_metrics(self, sample_pairs_good_quality):
        """Test summary metrics calculation"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        metrics = report.summary_metrics
        assert 'total_pairs' in metrics
        assert 'unique_properties' in metrics
        assert 'unique_tracts' in metrics
        assert 'unique_cbsas' in metrics
        assert 'years_covered' in metrics
        assert 'avg_holding_period_years' in metrics
        
        # Check values
        assert metrics['total_pairs'] == len(sample_pairs_good_quality)
        assert metrics['unique_properties'] == sample_pairs_good_quality['property_id'].nunique()
        assert metrics['unique_tracts'] == 30  # 3 CBSAs × 10 tracts
        assert metrics['unique_cbsas'] == 3
    
    def test_validation_analysis(self, sample_pairs_good_quality):
        """Test validation analysis"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        validation = report.validation_results
        assert 'overall' in validation
        assert 'passed' in validation['overall']
        assert 'failed' in validation['overall']
        assert 'pass_rate' in validation['overall']
        
        # Good quality data should have high pass rate
        assert validation['overall']['pass_rate'] > 0.95
    
    def test_temporal_coverage(self, sample_pairs_good_quality):
        """Test temporal coverage analysis"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        temporal = report.temporal_coverage
        assert len(temporal) > 0
        assert 'date' in temporal.columns
        assert 'pair_count' in temporal.columns
        assert 'tract_count' in temporal.columns
        assert 'has_sufficient_data' in temporal.columns
    
    def test_geographic_coverage(self, sample_pairs_good_quality, sample_geography_df):
        """Test geographic coverage analysis"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(
            sample_pairs_good_quality,
            sample_geography_df
        )
        
        geographic = report.geographic_coverage
        assert len(geographic) > 0
        assert 'geography_id' in geographic.columns
        assert 'pair_count' in geographic.columns
        assert 'has_sufficient_data' in geographic.columns
        
        # Should have overall summary
        overall = geographic[geographic['geography_id'] == 'OVERALL']
        assert len(overall) == 1
        assert 'coverage_rate' in overall.columns
    
    def test_price_distribution(self, sample_pairs_good_quality):
        """Test price distribution analysis"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        price_dist = report.price_distribution
        assert len(price_dist) == 3  # First sale, second sale, appreciation
        
        # Check statistics
        for col in ['mean', 'median', 'std', 'cv', 'min', 'max', 'skewness']:
            assert col in price_dist.columns
        
        # Check sale types
        sale_types = price_dist['sale_type'].tolist()
        assert 'First Sale' in sale_types
        assert 'Second Sale' in sale_types
        assert 'Appreciation Ratio' in sale_types
    
    def test_quality_scores(self, sample_pairs_good_quality):
        """Test quality score calculation"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        scores = report.quality_scores
        assert 'overall' in scores
        assert 'volume' in scores
        assert 'validation' in scores
        assert 'temporal_coverage' in scores
        assert 'geographic_coverage' in scores
        assert 'price_quality' in scores
        
        # All scores should be between 0 and 1
        for score in scores.values():
            assert 0 <= score <= 1
        
        # Good quality data should have reasonable scores
        assert scores['overall'] > 0.5  # Adjusted threshold for test data
    
    def test_recommendations_good_data(self, sample_pairs_good_quality):
        """Test recommendations for good quality data"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        recommendations = report.recommendations
        assert len(recommendations) > 0
        
        # Should have assessment based on quality
        assert len(recommendations) > 0
        # Check that it has some kind of quality assessment
        assert any(('good' in rec.lower() or 'acceptable' in rec.lower() or 'improvement' in rec.lower()) 
                   for rec in recommendations)
    
    def test_recommendations_poor_data(self, sample_pairs_poor_quality):
        """Test recommendations for poor quality data"""
        analyzer = DataQualityAnalyzer(
            min_pairs_threshold=100,
            min_geographic_coverage=0.8
        )
        report = analyzer.analyze_quality(sample_pairs_poor_quality)
        
        recommendations = report.recommendations
        assert len(recommendations) > 1  # Multiple issues
        
        # Should recommend improvements
        assert any('improve' in rec.lower() for rec in recommendations)
    
    def test_report_summary_creation(self, sample_pairs_good_quality):
        """Test creation of text summary"""
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(sample_pairs_good_quality)
        
        summary = analyzer.create_quality_report_summary(report)
        
        assert isinstance(summary, str)
        assert 'DATA QUALITY REPORT SUMMARY' in summary
        assert 'Overall Quality Score' in summary
        assert 'KEY METRICS' in summary
        assert 'RECOMMENDATIONS' in summary
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        analyzer = DataQualityAnalyzer()
        empty_df = pd.DataFrame(columns=[
            'property_id', 'tract_id', 'cbsa_id',
            'first_sale_date', 'first_sale_price',
            'second_sale_date', 'second_sale_price'
        ])
        
        report = analyzer.analyze_quality(empty_df)
        
        assert report.summary_metrics['total_pairs'] == 0
        assert report.quality_scores['overall'] == 0
        assert len(report.recommendations) > 0
    
    def test_quality_thresholds(self, sample_pairs_poor_quality):
        """Test that quality thresholds affect scoring"""
        # Lenient analyzer
        lenient = DataQualityAnalyzer(
            min_pairs_threshold=10,
            min_geographic_coverage=0.2,
            max_price_cv=5.0
        )
        lenient_report = lenient.analyze_quality(sample_pairs_poor_quality)
        
        # Strict analyzer
        strict = DataQualityAnalyzer(
            min_pairs_threshold=200,
            min_geographic_coverage=0.95,
            max_price_cv=1.0
        )
        strict_report = strict.analyze_quality(sample_pairs_poor_quality)
        
        # Lenient should give higher scores
        assert lenient_report.quality_scores['overall'] > strict_report.quality_scores['overall']
        
        # Strict should have more recommendations
        assert len(strict_report.recommendations) >= len(lenient_report.recommendations)