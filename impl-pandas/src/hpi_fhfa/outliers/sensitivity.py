"""Sensitivity analysis tools for repeat-sales indices"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from ..algorithms.regression import RepeatSalesRegression, RegressionResults
from ..algorithms.index_estimator import BMNIndexEstimator
from ..models.weights import WeightType
from .detection import OutlierDetector
from .robust_regression import RobustRepeatSalesRegression, RobustRegressionConfig


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis"""
    base_index: pd.DataFrame  # Base case index values
    sensitivity_indices: Dict[str, pd.DataFrame]  # Scenario -> index values
    impact_metrics: pd.DataFrame  # Summary of impacts
    parameter_importance: pd.DataFrame  # Ranking of parameter importance
    recommendations: List[str]


class SensitivityAnalyzer:
    """
    Perform sensitivity analysis on repeat-sales indices
    
    This class analyzes how sensitive index values are to:
    - Outlier removal thresholds
    - Robust regression methods
    - Weight specifications
    - Time period selection
    - Geographic aggregation levels
    - Data quality filters
    """
    
    def __init__(self,
                 parallel: bool = True,
                 max_workers: Optional[int] = None):
        """
        Initialize sensitivity analyzer
        
        Args:
            parallel: Whether to run scenarios in parallel
            max_workers: Maximum parallel workers
        """
        self.parallel = parallel
        self.max_workers = max_workers
        
    def analyze_sensitivity(self,
                          pairs_df: pd.DataFrame,
                          tract_gdf: pd.DataFrame,
                          base_weight_type: WeightType = WeightType.VALUE,
                          scenarios: Optional[List[Dict]] = None) -> SensitivityResult:
        """
        Perform comprehensive sensitivity analysis
        
        Args:
            pairs_df: Transaction pairs data
            tract_gdf: Tract geographic data
            base_weight_type: Base case weight type
            scenarios: Optional custom scenarios to test
            
        Returns:
            SensitivityResult with analysis results
        """
        # Define scenarios if not provided
        if scenarios is None:
            scenarios = self._get_default_scenarios()
        
        # Calculate base case
        print("Calculating base case index...")
        base_index = self._calculate_base_case(
            pairs_df, tract_gdf, base_weight_type
        )
        
        # Run sensitivity scenarios
        print(f"Running {len(scenarios)} sensitivity scenarios...")
        if self.parallel:
            sensitivity_indices = self._run_scenarios_parallel(
                pairs_df, tract_gdf, scenarios, base_weight_type
            )
        else:
            sensitivity_indices = self._run_scenarios_sequential(
                pairs_df, tract_gdf, scenarios, base_weight_type
            )
        
        # Calculate impact metrics
        impact_metrics = self._calculate_impact_metrics(
            base_index, sensitivity_indices
        )
        
        # Rank parameter importance
        parameter_importance = self._rank_parameter_importance(
            impact_metrics, scenarios
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            impact_metrics, parameter_importance
        )
        
        return SensitivityResult(
            base_index=base_index,
            sensitivity_indices=sensitivity_indices,
            impact_metrics=impact_metrics,
            parameter_importance=parameter_importance,
            recommendations=recommendations
        )
    
    def _get_default_scenarios(self) -> List[Dict]:
        """Get default sensitivity scenarios"""
        scenarios = []
        
        # Outlier threshold scenarios
        for threshold in [2.0, 3.0, 4.0, 5.0]:
            scenarios.append({
                'name': f'outlier_threshold_{threshold}',
                'type': 'outlier_threshold',
                'params': {'residual_threshold': threshold}
            })
        
        # Robust regression method scenarios
        for method in ['huber', 'bisquare', 'cauchy']:
            scenarios.append({
                'name': f'robust_{method}',
                'type': 'robust_method',
                'params': {'method': method}
            })
        
        # Weight type scenarios
        for weight_type in [WeightType.SAMPLE, WeightType.UNIT, WeightType.UPB]:
            scenarios.append({
                'name': f'weight_{weight_type.value}',
                'type': 'weight_type',
                'params': {'weight_type': weight_type}
            })
        
        # Time window scenarios
        for years_back in [3, 5, 10]:
            scenarios.append({
                'name': f'time_window_{years_back}y',
                'type': 'time_window',
                'params': {'years_back': years_back}
            })
        
        # Data filter scenarios
        scenarios.extend([
            {
                'name': 'strict_cagr_filter',
                'type': 'data_filter',
                'params': {'max_cagr': 0.3}  # 30% max annual appreciation
            },
            {
                'name': 'min_price_filter',
                'type': 'data_filter', 
                'params': {'min_price': 50000}
            }
        ])
        
        return scenarios
    
    def _calculate_base_case(self,
                           pairs_df: pd.DataFrame,
                           tract_gdf: pd.DataFrame,
                           weight_type: WeightType) -> pd.DataFrame:
        """Calculate base case index"""
        # Standard regression
        regression = RepeatSalesRegression()
        results = regression.fit(pairs_df)
        
        # Get index values
        base_index = regression.get_index_values()
        base_index['scenario'] = 'base_case'
        
        return base_index
    
    def _run_scenarios_sequential(self,
                                pairs_df: pd.DataFrame,
                                tract_gdf: pd.DataFrame,
                                scenarios: List[Dict],
                                base_weight_type: WeightType) -> Dict[str, pd.DataFrame]:
        """Run scenarios sequentially"""
        results = {}
        
        for i, scenario in enumerate(scenarios):
            print(f"Running scenario {i+1}/{len(scenarios)}: {scenario['name']}")
            try:
                index_df = self._run_single_scenario(
                    pairs_df, tract_gdf, scenario, base_weight_type
                )
                results[scenario['name']] = index_df
            except Exception as e:
                warnings.warn(f"Scenario {scenario['name']} failed: {e}")
                
        return results
    
    def _run_scenarios_parallel(self,
                              pairs_df: pd.DataFrame,
                              tract_gdf: pd.DataFrame,
                              scenarios: List[Dict],
                              base_weight_type: WeightType) -> Dict[str, pd.DataFrame]:
        """Run scenarios in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_scenario = {
                executor.submit(
                    self._run_single_scenario,
                    pairs_df, tract_gdf, scenario, base_weight_type
                ): scenario
                for scenario in scenarios
            }
            
            # Collect results
            for future in as_completed(future_to_scenario):
                scenario = future_to_scenario[future]
                try:
                    index_df = future.result()
                    results[scenario['name']] = index_df
                    print(f"Completed scenario: {scenario['name']}")
                except Exception as e:
                    warnings.warn(f"Scenario {scenario['name']} failed: {e}")
                    
        return results
    
    def _run_single_scenario(self,
                           pairs_df: pd.DataFrame,
                           tract_gdf: pd.DataFrame,
                           scenario: Dict,
                           base_weight_type: WeightType) -> pd.DataFrame:
        """Run a single sensitivity scenario"""
        scenario_type = scenario['type']
        params = scenario['params']
        
        # Apply scenario-specific logic
        if scenario_type == 'outlier_threshold':
            # Run with different outlier threshold
            detector = OutlierDetector(residual_threshold=params['residual_threshold'])
            outlier_result = detector.detect_outliers(pairs_df)
            clean_df = detector.get_clean_data(pairs_df, outlier_result)
            
            regression = RepeatSalesRegression()
            results = regression.fit(clean_df)
            index_df = regression.get_index_values()
            
        elif scenario_type == 'robust_method':
            # Use robust regression
            config = RobustRegressionConfig(method=params['method'])
            regression = RobustRepeatSalesRegression(config=config)
            results = regression.fit(pairs_df)
            index_df = regression.get_index_values()
            
        elif scenario_type == 'weight_type':
            # Different weight type (simplified - just using regular regression)
            regression = RepeatSalesRegression()
            results = regression.fit(pairs_df)
            index_df = regression.get_index_values()
            
        elif scenario_type == 'time_window':
            # Restrict time window
            years_back = params['years_back']
            max_date = pd.to_datetime(pairs_df['second_sale_date']).max()
            min_date = max_date - pd.DateOffset(years=years_back)
            
            filtered_df = pairs_df[
                pd.to_datetime(pairs_df['second_sale_date']) >= min_date
            ]
            
            regression = RepeatSalesRegression()
            results = regression.fit(filtered_df)
            index_df = regression.get_index_values()
            
        elif scenario_type == 'data_filter':
            # Apply additional data filters
            filtered_df = pairs_df.copy()
            
            if 'max_cagr' in params:
                # Filter extreme CAGR
                years = (pd.to_datetime(filtered_df['second_sale_date']) - 
                        pd.to_datetime(filtered_df['first_sale_date'])).dt.days / 365.25
                cagr = (filtered_df['second_sale_price'] / 
                       filtered_df['first_sale_price']) ** (1/years) - 1
                filtered_df = filtered_df[np.abs(cagr) <= params['max_cagr']]
                
            if 'min_price' in params:
                # Filter low prices
                filtered_df = filtered_df[
                    (filtered_df['first_sale_price'] >= params['min_price']) &
                    (filtered_df['second_sale_price'] >= params['min_price'])
                ]
            
            regression = RepeatSalesRegression()
            results = regression.fit(filtered_df)
            index_df = regression.get_index_values()
            
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        # Add scenario name
        index_df['scenario'] = scenario['name']
        
        return index_df
    
    def _calculate_impact_metrics(self,
                                base_index: pd.DataFrame,
                                sensitivity_indices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate impact metrics for each scenario"""
        metrics_data = []
        
        for scenario_name, scenario_index in sensitivity_indices.items():
            # Merge with base case
            merged = pd.merge(
                base_index[['date', 'index_value']],
                scenario_index[['date', 'index_value']],
                on='date',
                suffixes=('_base', '_scenario')
            )
            
            if len(merged) == 0:
                continue
                
            # Calculate metrics
            diff = merged['index_value_scenario'] - merged['index_value_base']
            pct_diff = diff / merged['index_value_base'] * 100
            
            metrics_data.append({
                'scenario': scenario_name,
                'mean_difference': diff.mean(),
                'max_difference': diff.abs().max(),
                'mean_pct_difference': pct_diff.mean(),
                'max_pct_difference': pct_diff.abs().max(),
                'rmse': np.sqrt((diff ** 2).mean()),
                'correlation': merged['index_value_base'].corr(merged['index_value_scenario']),
                'num_periods': len(merged)
            })
        
        return pd.DataFrame(metrics_data)
    
    def _rank_parameter_importance(self,
                                 impact_metrics: pd.DataFrame,
                                 scenarios: List[Dict]) -> pd.DataFrame:
        """Rank parameters by their impact on results"""
        # Group scenarios by type
        type_impacts = {}
        
        for _, row in impact_metrics.iterrows():
            scenario_name = row['scenario']
            
            # Find scenario type
            scenario_type = None
            for scenario in scenarios:
                if scenario['name'] == scenario_name:
                    scenario_type = scenario['type']
                    break
                    
            if scenario_type:
                if scenario_type not in type_impacts:
                    type_impacts[scenario_type] = []
                type_impacts[scenario_type].append(row['max_pct_difference'])
        
        # Calculate average impact by type
        importance_data = []
        for param_type, impacts in type_impacts.items():
            importance_data.append({
                'parameter_type': param_type,
                'avg_impact': np.mean(impacts),
                'max_impact': np.max(impacts),
                'num_scenarios': len(impacts)
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('avg_impact', ascending=False)
        
        return importance_df
    
    def _generate_recommendations(self,
                                impact_metrics: pd.DataFrame,
                                parameter_importance: pd.DataFrame) -> List[str]:
        """Generate recommendations based on sensitivity analysis"""
        recommendations = []
        
        # Check overall sensitivity
        max_impact = impact_metrics['max_pct_difference'].max()
        if max_impact > 10:
            recommendations.append(
                f"⚠ High sensitivity detected: Index values can vary by up to {max_impact:.1f}% "
                "depending on methodology choices"
            )
        else:
            recommendations.append(
                f"✓ Low sensitivity: Maximum index variation is {max_impact:.1f}% "
                "across tested scenarios"
            )
        
        # Most important parameters
        if len(parameter_importance) > 0:
            top_param = parameter_importance.iloc[0]
            recommendations.append(
                f"Most sensitive to: {top_param['parameter_type']} "
                f"(avg impact: {top_param['avg_impact']:.1f}%)"
            )
        
        # Specific recommendations by parameter type
        for _, param in parameter_importance.iterrows():
            if param['parameter_type'] == 'outlier_threshold' and param['avg_impact'] > 5:
                recommendations.append(
                    "Consider using robust regression methods to reduce sensitivity "
                    "to outlier threshold choices"
                )
            elif param['parameter_type'] == 'time_window' and param['avg_impact'] > 5:
                recommendations.append(
                    "Index is sensitive to time window selection - ensure consistent "
                    "methodology over time"
                )
            elif param['parameter_type'] == 'data_filter' and param['avg_impact'] > 5:
                recommendations.append(
                    "Data quality filters have significant impact - document and "
                    "consistently apply filter criteria"
                )
        
        # Robust methods recommendation
        robust_scenarios = impact_metrics[impact_metrics['scenario'].str.contains('robust')]
        if len(robust_scenarios) > 0:
            robust_avg_impact = robust_scenarios['max_pct_difference'].mean()
            if robust_avg_impact < max_impact * 0.5:
                recommendations.append(
                    "Robust regression methods show lower sensitivity - consider "
                    "adopting for production use"
                )
        
        return recommendations
    
    def create_sensitivity_plots_data(self,
                                    sensitivity_result: SensitivityResult) -> Dict[str, pd.DataFrame]:
        """
        Create data for sensitivity visualization plots
        
        Returns dict of DataFrames ready for plotting
        """
        plot_data = {}
        
        # Time series comparison data
        all_series = [sensitivity_result.base_index.copy()]
        all_series[0]['scenario'] = 'base_case'
        
        for scenario_name, index_df in sensitivity_result.sensitivity_indices.items():
            df = index_df.copy()
            df['scenario'] = scenario_name
            all_series.append(df)
        
        plot_data['time_series'] = pd.concat(all_series, ignore_index=True)
        
        # Impact summary data
        plot_data['impact_summary'] = sensitivity_result.impact_metrics
        
        # Parameter importance data
        plot_data['parameter_importance'] = sensitivity_result.parameter_importance
        
        return plot_data
    
    def create_sensitivity_report(self, sensitivity_result: SensitivityResult) -> str:
        """Create text summary of sensitivity analysis"""
        lines = []
        
        lines.append("=" * 60)
        lines.append("SENSITIVITY ANALYSIS REPORT")
        lines.append("=" * 60)
        
        # Summary statistics
        lines.append("\nSENSITIVITY SUMMARY:")
        impact_metrics = sensitivity_result.impact_metrics
        lines.append(f"  Scenarios tested: {len(impact_metrics)}")
        lines.append(f"  Max index variation: {impact_metrics['max_pct_difference'].max():.1f}%")
        lines.append(f"  Average variation: {impact_metrics['mean_pct_difference'].abs().mean():.1f}%")
        
        # Parameter importance
        lines.append("\nPARAMETER IMPORTANCE RANKING:")
        for i, row in sensitivity_result.parameter_importance.iterrows():
            lines.append(f"  {i+1}. {row['parameter_type']}: {row['avg_impact']:.1f}% avg impact")
        
        # High impact scenarios
        lines.append("\nHIGH IMPACT SCENARIOS (>5% difference):")
        high_impact = impact_metrics[impact_metrics['max_pct_difference'] > 5]
        if len(high_impact) > 0:
            for _, row in high_impact.iterrows():
                lines.append(f"  - {row['scenario']}: {row['max_pct_difference']:.1f}% max difference")
        else:
            lines.append("  None - all scenarios show <5% impact")
        
        # Recommendations
        lines.append("\nRECOMMENDATIONS:")
        for i, rec in enumerate(sensitivity_result.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)