"""Pipeline orchestration for end-to-end HPI processing"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, date
from pathlib import Path
import logging
import json
import pandas as pd
from enum import Enum

from ..data import DataLoader
from ..models import SampleWeightCalculator, WeightType
from ..models.validators import DataValidator
from ..algorithms import BMNIndexEstimator, RepeatSalesRegression
from ..outliers import (
    DataQualityAnalyzer, OutlierDetector,
    RobustRepeatSalesRegression, RobustRegressionConfig
)
from ..weighting import GeographicAggregator


class StepStatus(str, Enum):
    """Pipeline step status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStep:
    """Individual pipeline step"""
    name: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    optional: bool = False
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[int] = None


@dataclass
class StepResult:
    """Result from a pipeline step"""
    step_name: str
    status: StepStatus
    start_time: datetime
    end_time: datetime
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Overall pipeline execution result"""
    pipeline_id: str
    start_time: datetime
    end_time: datetime
    status: str  # "success", "partial_success", "failed"
    step_results: Dict[str, StepResult]
    final_output: Optional[Any] = None
    error_summary: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """Generic pipeline orchestrator"""
    
    def __init__(self, name: str):
        """Initialize pipeline
        
        Args:
            name: Pipeline name
        """
        self.name = name
        self.steps: Dict[str, PipelineStep] = {}
        self.logger = logging.getLogger(f"pipeline.{name}")
        
    def add_step(self, step: PipelineStep):
        """Add step to pipeline
        
        Args:
            step: Pipeline step to add
        """
        self.steps[step.name] = step
        
    def execute(self, context: Dict[str, Any]) -> PipelineResult:
        """Execute pipeline
        
        Args:
            context: Execution context
            
        Returns:
            PipelineResult with execution details
        """
        pipeline_id = context.get('pipeline_id', str(datetime.now().timestamp()))
        start_time = datetime.now()
        step_results = {}
        
        self.logger.info(f"Starting pipeline execution: {pipeline_id}")
        
        # Determine execution order
        execution_order = self._determine_execution_order()
        
        # Execute steps
        for step_name in execution_order:
            step = self.steps[step_name]
            
            # Check dependencies
            if not self._check_dependencies(step, step_results):
                if step.optional:
                    step_results[step_name] = StepResult(
                        step_name=step_name,
                        status=StepStatus.SKIPPED,
                        start_time=datetime.now(),
                        end_time=datetime.now()
                    )
                    continue
                else:
                    # Required step with failed dependencies
                    step_results[step_name] = StepResult(
                        step_name=step_name,
                        status=StepStatus.FAILED,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error="Dependencies not met"
                    )
                    break
            
            # Execute step
            step_result = self._execute_step(step, context, step_results)
            step_results[step_name] = step_result
            
            # Update context with step output
            if step_result.output is not None:
                context[f"{step_name}_output"] = step_result.output
            
            # Stop on failure of required step
            if step_result.status == StepStatus.FAILED and not step.optional:
                self.logger.error(f"Required step {step_name} failed, stopping pipeline")
                break
        
        # Determine overall status
        failed_steps = [r for r in step_results.values() if r.status == StepStatus.FAILED]
        if not failed_steps:
            status = "success"
        elif all(self.steps[r.step_name].optional for r in failed_steps):
            status = "partial_success"
        else:
            status = "failed"
        
        # Get final output
        final_step = execution_order[-1] if execution_order else None
        final_output = None
        if final_step and final_step in step_results:
            final_output = step_results[final_step].output
        
        # Compile errors
        error_summary = [
            f"{r.step_name}: {r.error}"
            for r in step_results.values()
            if r.status == StepStatus.FAILED and r.error
        ]
        
        # Calculate metrics
        total_duration = (datetime.now() - start_time).total_seconds()
        metrics = {
            'total_duration_seconds': total_duration,
            'steps_executed': len(step_results),
            'steps_succeeded': sum(1 for r in step_results.values() if r.status == StepStatus.COMPLETED),
            'steps_failed': len(failed_steps)
        }
        
        return PipelineResult(
            pipeline_id=pipeline_id,
            start_time=start_time,
            end_time=datetime.now(),
            status=status,
            step_results=step_results,
            final_output=final_output,
            error_summary=error_summary,
            metrics=metrics
        )
    
    def _determine_execution_order(self) -> List[str]:
        """Determine step execution order based on dependencies"""
        # Simple topological sort
        order = []
        visited = set()
        
        def visit(step_name: str):
            if step_name in visited:
                return
            visited.add(step_name)
            
            step = self.steps.get(step_name)
            if step:
                for dep in step.dependencies:
                    if dep in self.steps:
                        visit(dep)
                order.append(step_name)
        
        for step_name in self.steps:
            visit(step_name)
        
        return order
    
    def _check_dependencies(self, step: PipelineStep, results: Dict[str, StepResult]) -> bool:
        """Check if step dependencies are met"""
        for dep in step.dependencies:
            if dep not in results:
                return False
            if results[dep].status not in [StepStatus.COMPLETED, StepStatus.SKIPPED]:
                return False
        return True
    
    def _execute_step(self, step: PipelineStep, context: Dict[str, Any],
                     previous_results: Dict[str, StepResult]) -> StepResult:
        """Execute a single pipeline step"""
        start_time = datetime.now()
        self.logger.info(f"Executing step: {step.name}")
        
        for attempt in range(step.max_retries + 1):
            try:
                # Execute step function
                output = step.function(context, previous_results)
                
                # Success
                return StepResult(
                    step_name=step.name,
                    status=StepStatus.COMPLETED,
                    start_time=start_time,
                    end_time=datetime.now(),
                    output=output
                )
                
            except Exception as e:
                self.logger.error(f"Step {step.name} failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt == step.max_retries:
                    # Final failure
                    return StepResult(
                        step_name=step.name,
                        status=StepStatus.FAILED,
                        start_time=start_time,
                        end_time=datetime.now(),
                        error=str(e)
                    )
                
                # Retry
                step.retry_count = attempt + 1


class HPIPipeline(Pipeline):
    """Specialized pipeline for HPI calculation"""
    
    def __init__(self, data_path: Path):
        """Initialize HPI pipeline
        
        Args:
            data_path: Path to data storage
        """
        super().__init__("hpi_calculation")
        self.data_path = data_path
        self._setup_steps()
        
    def _setup_steps(self):
        """Set up HPI calculation steps"""
        # Step 1: Load data
        self.add_step(PipelineStep(
            name="load_data",
            function=self._load_data_step
        ))
        
        # Step 2: Validate data
        self.add_step(PipelineStep(
            name="validate_data",
            function=self._validate_data_step,
            dependencies=["load_data"]
        ))
        
        # Step 3: Quality analysis (optional)
        self.add_step(PipelineStep(
            name="quality_analysis",
            function=self._quality_analysis_step,
            dependencies=["validate_data"],
            optional=True
        ))
        
        # Step 4: Outlier detection
        self.add_step(PipelineStep(
            name="outlier_detection",
            function=self._outlier_detection_step,
            dependencies=["validate_data"]
        ))
        
        # Step 5: Calculate indices
        self.add_step(PipelineStep(
            name="calculate_indices",
            function=self._calculate_indices_step,
            dependencies=["outlier_detection"]
        ))
        
        # Step 6: Aggregate results
        self.add_step(PipelineStep(
            name="aggregate_results",
            function=self._aggregate_results_step,
            dependencies=["calculate_indices"]
        ))
        
        # Step 7: Generate reports
        self.add_step(PipelineStep(
            name="generate_reports",
            function=self._generate_reports_step,
            dependencies=["aggregate_results", "quality_analysis"],
            optional=True
        ))
    
    def _load_data_step(self, context: Dict[str, Any], previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Load data step"""
        loader = DataLoader(self.data_path)
        
        start_date = context.get('start_date')
        end_date = context.get('end_date')
        
        pairs_df = loader.load_transaction_pairs(start_date, end_date)
        geo_df = loader.load_geography_data()
        adjacency = loader.load_adjacency_data()
        
        return {
            'pairs_df': pairs_df,
            'geo_df': geo_df,
            'adjacency': adjacency,
            'record_count': len(pairs_df)
        }
    
    def _validate_data_step(self, context: Dict[str, Any], previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Validate data step"""
        data = previous_results['load_data'].output
        pairs_df = data['pairs_df']
        
        validator = DataValidator()
        validated_df = validator.validate_transaction_batch(pairs_df)
        valid_pairs = validated_df[validated_df['is_valid']]
        
        return {
            'valid_pairs': valid_pairs,
            'validation_rate': len(valid_pairs) / len(pairs_df) if len(pairs_df) > 0 else 0,
            'invalid_count': len(pairs_df) - len(valid_pairs)
        }
    
    def _quality_analysis_step(self, context: Dict[str, Any], previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Quality analysis step"""
        valid_pairs = previous_results['validate_data'].output['valid_pairs']
        geo_df = previous_results['load_data'].output['geo_df']
        
        analyzer = DataQualityAnalyzer()
        report = analyzer.analyze_quality(valid_pairs, geo_df)
        
        return {
            'quality_report': report,
            'overall_score': report.quality_scores['overall']
        }
    
    def _outlier_detection_step(self, context: Dict[str, Any], previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Outlier detection step"""
        valid_pairs = previous_results['validate_data'].output['valid_pairs']
        
        detector = OutlierDetector()
        outlier_result = detector.detect_outliers(valid_pairs)
        clean_pairs = detector.get_clean_data(valid_pairs, outlier_result)
        
        return {
            'clean_pairs': clean_pairs,
            'outlier_result': outlier_result,
            'outlier_rate': outlier_result.statistics['outlier_rate']
        }
    
    def _calculate_indices_step(self, context: Dict[str, Any], previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Calculate indices step"""
        clean_pairs = previous_results['outlier_detection'].output['clean_pairs']
        geo_df = previous_results['load_data'].output['geo_df']
        adjacency = previous_results['load_data'].output['adjacency']
        
        # Get configuration from context
        geography_level = context.get('geography_level', 'cbsa')
        weight_type = context.get('weight_type', WeightType.SAMPLE)
        robust = context.get('robust_regression', False)
        
        # Initialize estimator
        if robust:
            config = RobustRegressionConfig(method='huber')
            estimator = BMNIndexEstimator(
                regression_class=RobustRepeatSalesRegression,
                regression_kwargs={'config': config},
                adjacency_data=adjacency
            )
        else:
            estimator = BMNIndexEstimator(adjacency_data=adjacency)
        
        # Calculate indices
        results = estimator.estimate_indices(
            clean_pairs,
            geo_df,
            weight_type,
            geography_level=geography_level,
            start_date=context.get('start_date'),
            end_date=context.get('end_date')
        )
        
        return {
            'index_results': results,
            'coverage_rate': results.coverage_stats['coverage_rate']
        }
    
    def _aggregate_results_step(self, context: Dict[str, Any], previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Aggregate results step"""
        index_results = previous_results['calculate_indices'].output['index_results']
        
        # Additional aggregation if needed
        aggregator = IndexAggregator()
        
        # Get national aggregate if not already calculated
        if context.get('include_national_aggregate', True):
            # Would aggregate to national level here
            pass
        
        return {
            'final_indices': index_results.index_values,
            'regression_results': index_results.regression_results,
            'coverage_stats': index_results.coverage_stats
        }
    
    def _generate_reports_step(self, context: Dict[str, Any], previous_results: Dict[str, StepResult]) -> Dict[str, Any]:
        """Generate reports step"""
        output_path = context.get('output_path', self.data_path / 'reports')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save index results
        final_indices = previous_results['aggregate_results'].output['final_indices']
        final_indices.to_csv(output_path / 'index_values.csv', index=False)
        
        # Save quality report if available
        if 'quality_analysis' in previous_results:
            quality_report = previous_results['quality_analysis'].output['quality_report']
            with open(output_path / 'quality_report.json', 'w') as f:
                json.dump({
                    'overall_score': quality_report.quality_scores['overall'],
                    'scores': quality_report.quality_scores,
                    'metrics': quality_report.summary_metrics,
                    'recommendations': quality_report.recommendations
                }, f, indent=2)
        
        # Save outlier report
        outlier_result = previous_results['outlier_detection'].output['outlier_result']
        with open(output_path / 'outlier_report.json', 'w') as f:
            json.dump({
                'statistics': outlier_result.statistics,
                'thresholds': outlier_result.thresholds
            }, f, indent=2)
        
        return {
            'report_path': str(output_path),
            'files_generated': ['index_values.csv', 'quality_report.json', 'outlier_report.json']
        }