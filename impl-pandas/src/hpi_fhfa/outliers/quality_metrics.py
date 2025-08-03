"""Data quality metrics and reporting for repeat-sales data"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date, timedelta

from ..models.transaction import TransactionPair
from ..models.validators import DataValidator


@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    summary_metrics: Dict[str, float]
    validation_results: Dict[str, Dict[str, int]]
    temporal_coverage: pd.DataFrame
    geographic_coverage: pd.DataFrame
    price_distribution: pd.DataFrame
    quality_scores: Dict[str, float]
    recommendations: List[str]


class DataQualityAnalyzer:
    """
    Analyze data quality for repeat-sales datasets
    
    This class provides comprehensive quality metrics including:
    - Validation pass rates
    - Temporal and geographic coverage
    - Price distribution analysis
    - Data completeness scores
    - Quality recommendations
    """
    
    def __init__(self,
                 min_pairs_threshold: int = 100,
                 min_geographic_coverage: float = 0.8,
                 max_price_cv: float = 2.0):
        """
        Initialize quality analyzer
        
        Args:
            min_pairs_threshold: Minimum pairs for adequate sample
            min_geographic_coverage: Minimum fraction of geographies with data
            max_price_cv: Maximum coefficient of variation for prices
        """
        self.min_pairs_threshold = min_pairs_threshold
        self.min_geographic_coverage = min_geographic_coverage
        self.max_price_cv = max_price_cv
        self.validator = DataValidator()
        
    def analyze_quality(self,
                       pairs_df: pd.DataFrame,
                       geography_df: Optional[pd.DataFrame] = None) -> QualityReport:
        """
        Perform comprehensive quality analysis
        
        Args:
            pairs_df: DataFrame of transaction pairs
            geography_df: Optional geographic reference data
            
        Returns:
            QualityReport with detailed metrics
        """
        # Summary metrics
        summary_metrics = self._calculate_summary_metrics(pairs_df)
        
        # Validation results
        validation_results = self._analyze_validation(pairs_df)
        
        # Temporal coverage
        temporal_coverage = self._analyze_temporal_coverage(pairs_df)
        
        # Geographic coverage
        geographic_coverage = self._analyze_geographic_coverage(
            pairs_df, geography_df
        )
        
        # Price distribution
        price_distribution = self._analyze_price_distribution(pairs_df)
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(
            summary_metrics,
            validation_results,
            temporal_coverage,
            geographic_coverage,
            price_distribution
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            summary_metrics,
            quality_scores,
            validation_results
        )
        
        return QualityReport(
            summary_metrics=summary_metrics,
            validation_results=validation_results,
            temporal_coverage=temporal_coverage,
            geographic_coverage=geographic_coverage,
            price_distribution=price_distribution,
            quality_scores=quality_scores,
            recommendations=recommendations
        )
    
    def _calculate_summary_metrics(self, pairs_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic summary metrics"""
        metrics = {}
        
        # Basic counts
        metrics['total_pairs'] = len(pairs_df)
        metrics['unique_properties'] = pairs_df['property_id'].nunique()
        metrics['unique_tracts'] = pairs_df['tract_id'].nunique()
        metrics['unique_cbsas'] = pairs_df['cbsa_id'].nunique()
        
        # Temporal span
        if len(pairs_df) > 0:
            all_dates = pd.concat([
                pd.to_datetime(pairs_df['first_sale_date']),
                pd.to_datetime(pairs_df['second_sale_date'])
            ])
            metrics['start_date'] = all_dates.min().strftime('%Y-%m-%d')
            metrics['end_date'] = all_dates.max().strftime('%Y-%m-%d')
            metrics['years_covered'] = (all_dates.max() - all_dates.min()).days / 365.25
        else:
            metrics['start_date'] = 'N/A'
            metrics['end_date'] = 'N/A'
            metrics['years_covered'] = 0
        
        # Average statistics
        if len(pairs_df) > 0:
            # Time between sales
            time_diffs = (pd.to_datetime(pairs_df['second_sale_date']) - 
                         pd.to_datetime(pairs_df['first_sale_date'])).dt.days / 365.25
            metrics['avg_holding_period_years'] = time_diffs.mean()
            metrics['median_holding_period_years'] = time_diffs.median()
            
            # Price statistics
            metrics['avg_first_price'] = pairs_df['first_sale_price'].mean()
            metrics['avg_second_price'] = pairs_df['second_sale_price'].mean()
            metrics['avg_appreciation'] = (
                pairs_df['second_sale_price'] / pairs_df['first_sale_price']
            ).mean() - 1
        
        return metrics
    
    def _analyze_validation(self, pairs_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Analyze validation pass/fail rates"""
        results = {}
        
        # Overall validation using DataValidator
        validated_df = self.validator.validate_transaction_batch(pairs_df)
        overall_valid = validated_df['is_valid']
        
        results['overall'] = {
            'passed': overall_valid.sum(),
            'failed': (~overall_valid).sum(),
            'pass_rate': overall_valid.mean()
        }
        
        # Individual validation rules
        validation_rules = [
            ('valid_prices', 'positive_prices'),
            ('valid_dates', 'date_order'),
            ('valid_time_diff', 'minimum_time_gap'),
            ('valid_cagr', 'cagr_bounds'),
            ('valid_appreciation', 'appreciation_bounds')
        ]
        
        for col_name, rule_name in validation_rules:
            if col_name in validated_df.columns:
                results[rule_name] = {
                    'passed': validated_df[col_name].sum(),
                    'failed': (~validated_df[col_name]).sum(),
                    'pass_rate': validated_df[col_name].mean()
                }
        
        return results
    
    def _analyze_temporal_coverage(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze temporal coverage and gaps"""
        if len(pairs_df) == 0:
            return pd.DataFrame()
        
        # Convert to datetime
        pairs_df = pairs_df.copy()
        pairs_df['second_sale_date'] = pd.to_datetime(pairs_df['second_sale_date'])
        
        # Group by year-month
        pairs_df['year_month'] = pairs_df['second_sale_date'].dt.to_period('M')
        
        temporal_stats = pairs_df.groupby('year_month').agg({
            'property_id': 'count',
            'tract_id': 'nunique',
            'second_sale_price': ['mean', 'std', 'median']
        }).reset_index()
        
        temporal_stats.columns = [
            'year_month', 'pair_count', 'tract_count',
            'avg_price', 'std_price', 'median_price'
        ]
        
        # Convert period back to timestamp for easier handling
        temporal_stats['date'] = temporal_stats['year_month'].dt.to_timestamp()
        temporal_stats = temporal_stats.drop('year_month', axis=1)
        
        # Add coverage metrics
        temporal_stats['has_sufficient_data'] = (
            temporal_stats['pair_count'] >= self.min_pairs_threshold / 12
        )
        
        return temporal_stats
    
    def _analyze_geographic_coverage(self,
                                   pairs_df: pd.DataFrame,
                                   geography_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Analyze geographic coverage"""
        # Group by geography
        geo_stats = []
        
        # CBSA level
        cbsa_stats = pairs_df.groupby('cbsa_id').agg({
            'property_id': 'count',
            'tract_id': 'nunique',
            'second_sale_price': ['mean', 'std']
        }).reset_index()
        
        cbsa_stats.columns = [
            'geography_id', 'pair_count', 'tract_count',
            'avg_price', 'std_price'
        ]
        cbsa_stats['geography_type'] = 'cbsa'
        
        # Tract level
        tract_stats = pairs_df.groupby('tract_id').agg({
            'property_id': 'count',
            'second_sale_price': ['mean', 'std']
        }).reset_index()
        
        tract_stats.columns = [
            'geography_id', 'pair_count',
            'avg_price', 'std_price'
        ]
        tract_stats['tract_count'] = 1
        tract_stats['geography_type'] = 'tract'
        
        # Combine
        geo_stats = pd.concat([cbsa_stats, tract_stats], ignore_index=True)
        
        # Add coverage flags
        geo_stats['has_sufficient_data'] = geo_stats['pair_count'] >= 10
        geo_stats['price_cv'] = geo_stats['std_price'] / geo_stats['avg_price']
        
        # If we have reference geography data, calculate coverage rate
        if geography_df is not None:
            total_tracts = len(geography_df)
            covered_tracts = pairs_df['tract_id'].nunique()
            coverage_rate = covered_tracts / total_tracts
            
            # Add to summary
            summary_row = pd.DataFrame([{
                'geography_id': 'OVERALL',
                'geography_type': 'summary',
                'pair_count': len(pairs_df),
                'tract_count': covered_tracts,
                'avg_price': pairs_df['second_sale_price'].mean(),
                'std_price': pairs_df['second_sale_price'].std(),
                'has_sufficient_data': len(pairs_df) >= self.min_pairs_threshold,
                'price_cv': pairs_df['second_sale_price'].std() / pairs_df['second_sale_price'].mean(),
                'coverage_rate': coverage_rate
            }])
            
            geo_stats = pd.concat([summary_row, geo_stats], ignore_index=True)
        
        return geo_stats
    
    def _analyze_price_distribution(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze price distribution and outliers"""
        if len(pairs_df) == 0:
            return pd.DataFrame()
        
        # Calculate price metrics
        price_data = []
        
        for price_col, sale_type in [
            ('first_sale_price', 'First Sale'),
            ('second_sale_price', 'Second Sale')
        ]:
            prices = pairs_df[price_col]
            
            price_data.append({
                'sale_type': sale_type,
                'mean': prices.mean(),
                'median': prices.median(),
                'std': prices.std(),
                'cv': prices.std() / prices.mean(),
                'min': prices.min(),
                'p05': prices.quantile(0.05),
                'p25': prices.quantile(0.25),
                'p75': prices.quantile(0.75),
                'p95': prices.quantile(0.95),
                'max': prices.max(),
                'iqr': prices.quantile(0.75) - prices.quantile(0.25),
                'skewness': prices.skew(),
                'kurtosis': prices.kurtosis()
            })
        
        # Add appreciation statistics
        appreciation = pairs_df['second_sale_price'] / pairs_df['first_sale_price']
        price_data.append({
            'sale_type': 'Appreciation Ratio',
            'mean': appreciation.mean(),
            'median': appreciation.median(),
            'std': appreciation.std(),
            'cv': appreciation.std() / appreciation.mean(),
            'min': appreciation.min(),
            'p05': appreciation.quantile(0.05),
            'p25': appreciation.quantile(0.25),
            'p75': appreciation.quantile(0.75),
            'p95': appreciation.quantile(0.95),
            'max': appreciation.max(),
            'iqr': appreciation.quantile(0.75) - appreciation.quantile(0.25),
            'skewness': appreciation.skew(),
            'kurtosis': appreciation.kurtosis()
        })
        
        return pd.DataFrame(price_data)
    
    def _calculate_quality_scores(self,
                                summary_metrics: Dict[str, float],
                                validation_results: Dict[str, Dict[str, int]],
                                temporal_coverage: pd.DataFrame,
                                geographic_coverage: pd.DataFrame,
                                price_distribution: pd.DataFrame) -> Dict[str, float]:
        """Calculate quality scores for different aspects"""
        scores = {}
        
        # Data volume score (0-1)
        volume_score = min(1.0, summary_metrics['total_pairs'] / self.min_pairs_threshold)
        scores['volume'] = volume_score
        
        # Validation score (0-1)
        validation_score = validation_results['overall']['pass_rate']
        scores['validation'] = validation_score
        
        # Temporal coverage score (0-1)
        if len(temporal_coverage) > 0:
            sufficient_months = temporal_coverage['has_sufficient_data'].sum()
            total_months = len(temporal_coverage)
            temporal_score = sufficient_months / total_months if total_months > 0 else 0
        else:
            temporal_score = 0
        scores['temporal_coverage'] = temporal_score
        
        # Geographic coverage score (0-1)
        if len(geographic_coverage) > 0:
            overall_row = geographic_coverage[geographic_coverage['geography_id'] == 'OVERALL']
            if len(overall_row) > 0 and 'coverage_rate' in overall_row.columns:
                geo_score = overall_row['coverage_rate'].iloc[0]
            else:
                covered_geos = geographic_coverage['has_sufficient_data'].sum()
                total_geos = len(geographic_coverage)
                geo_score = covered_geos / total_geos if total_geos > 0 else 0
        else:
            geo_score = 0
        scores['geographic_coverage'] = geo_score
        
        # Price quality score (0-1)
        if len(price_distribution) > 0:
            # Check coefficient of variation
            max_cv = price_distribution['cv'].max()
            cv_score = 1.0 - min(1.0, max_cv / self.max_price_cv)
            
            # Check for extreme skewness
            max_skew = price_distribution['skewness'].abs().max()
            skew_score = 1.0 - min(1.0, max_skew / 3.0)  # Penalty for |skew| > 3
            
            price_score = (cv_score + skew_score) / 2
        else:
            price_score = 0
        scores['price_quality'] = price_score
        
        # Overall score (weighted average)
        weights = {
            'volume': 0.2,
            'validation': 0.3,
            'temporal_coverage': 0.2,
            'geographic_coverage': 0.2,
            'price_quality': 0.1
        }
        
        # Handle NaN values
        valid_scores = {k: v for k, v in scores.items() if k in weights and not np.isnan(v)}
        if valid_scores:
            total_weight = sum(weights[k] for k in valid_scores)
            overall_score = sum(valid_scores[k] * weights[k] for k in valid_scores) / total_weight
        else:
            overall_score = 0.0
            
        scores['overall'] = overall_score
        
        return scores
    
    def _generate_recommendations(self,
                                summary_metrics: Dict[str, float],
                                quality_scores: Dict[str, float],
                                validation_results: Dict[str, Dict[str, int]]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Volume recommendations
        if summary_metrics['total_pairs'] < self.min_pairs_threshold:
            recommendations.append(
                f"Increase data volume: Currently have {summary_metrics['total_pairs']} pairs, "
                f"recommend at least {self.min_pairs_threshold}"
            )
        
        # Validation recommendations
        if quality_scores['validation'] < 0.9:
            failed_rules = []
            for rule, results in validation_results.items():
                if rule != 'overall' and results['pass_rate'] < 0.9:
                    failed_rules.append(f"{rule} ({results['pass_rate']:.1%})")
            
            if failed_rules:
                recommendations.append(
                    f"Improve data quality for rules: {', '.join(failed_rules)}"
                )
        
        # Coverage recommendations
        if quality_scores['temporal_coverage'] < 0.8:
            recommendations.append(
                "Improve temporal coverage: Many months have insufficient data"
            )
        
        if quality_scores['geographic_coverage'] < self.min_geographic_coverage:
            recommendations.append(
                f"Improve geographic coverage: Currently at "
                f"{quality_scores['geographic_coverage']:.1%}, "
                f"recommend at least {self.min_geographic_coverage:.1%}"
            )
        
        # Price quality recommendations
        if quality_scores['price_quality'] < 0.7:
            recommendations.append(
                "Review price distributions: High variability or skewness detected"
            )
        
        # Overall assessment
        if quality_scores['overall'] >= 0.7:
            recommendations.insert(0, "✓ Data quality is good overall")
        elif quality_scores['overall'] >= 0.5:
            recommendations.insert(0, "⚠ Data quality is acceptable but could be improved")
        else:
            recommendations.insert(0, "✗ Data quality needs significant improvement")
        
        return recommendations
    
    def create_quality_report_summary(self, quality_report: QualityReport) -> str:
        """Create a text summary of the quality report"""
        lines = []
        
        lines.append("=" * 60)
        lines.append("DATA QUALITY REPORT SUMMARY")
        lines.append("=" * 60)
        
        # Overall assessment
        lines.append(f"\nOverall Quality Score: {quality_report.quality_scores['overall']:.2%}")
        lines.append("")
        
        # Key metrics
        lines.append("KEY METRICS:")
        for metric, value in quality_report.summary_metrics.items():
            if isinstance(value, (int, float)):
                if 'rate' in metric or 'appreciation' in metric:
                    lines.append(f"  {metric}: {value:.2%}")
                elif 'price' in metric:
                    lines.append(f"  {metric}: ${value:,.0f}")
                else:
                    lines.append(f"  {metric}: {value:,.1f}")
            else:
                lines.append(f"  {metric}: {value}")
        
        # Quality scores
        lines.append("\nQUALITY SCORES:")
        for aspect, score in quality_report.quality_scores.items():
            if aspect != 'overall':
                lines.append(f"  {aspect}: {score:.2%}")
        
        # Validation summary
        lines.append("\nVALIDATION SUMMARY:")
        overall_valid = quality_report.validation_results['overall']
        lines.append(f"  Pass Rate: {overall_valid['pass_rate']:.1%}")
        lines.append(f"  Passed: {overall_valid['passed']:,}")
        lines.append(f"  Failed: {overall_valid['failed']:,}")
        
        # Recommendations
        lines.append("\nRECOMMENDATIONS:")
        for i, rec in enumerate(quality_report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)