"""Unit tests for pipeline orchestrator"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile

from hpi_fhfa.pipeline.orchestrator import (
    Pipeline, PipelineStep, PipelineResult, StepResult,
    StepStatus, HPIPipeline
)


class TestPipelineOrchestrator:
    """Test pipeline orchestration functionality"""
    
    def test_pipeline_step_creation(self):
        """Test creating pipeline steps"""
        def dummy_function(context, results):
            return {"result": "success"}
            
        step = PipelineStep(
            name="test_step",
            function=dummy_function,
            dependencies=["dep1", "dep2"],
            optional=False,
            max_retries=3
        )
        
        assert step.name == "test_step"
        assert step.function == dummy_function
        assert len(step.dependencies) == 2
        assert step.optional is False
        assert step.max_retries == 3
        
    def test_simple_pipeline_execution(self):
        """Test simple pipeline execution"""
        pipeline = Pipeline("test_pipeline")
        
        # Add steps
        def step1_func(context, results):
            return {"value": 10}
            
        def step2_func(context, results):
            step1_result = results["step1"].output
            return {"value": step1_result["value"] * 2}
            
        pipeline.add_step(PipelineStep("step1", step1_func))
        pipeline.add_step(PipelineStep("step2", step2_func, dependencies=["step1"]))
        
        # Execute
        context = {"pipeline_id": "test123"}
        result = pipeline.execute(context)
        
        assert result.status == "success"
        assert len(result.step_results) == 2
        assert result.step_results["step1"].status == StepStatus.COMPLETED
        assert result.step_results["step2"].status == StepStatus.COMPLETED
        assert result.final_output["value"] == 20
        
    def test_pipeline_with_failure(self):
        """Test pipeline with failing step"""
        pipeline = Pipeline("test_pipeline")
        
        def failing_step(context, results):
            raise ValueError("Step failed")
            
        def dependent_step(context, results):
            return {"result": "should not execute"}
            
        pipeline.add_step(PipelineStep("failing", failing_step, max_retries=0))
        pipeline.add_step(PipelineStep("dependent", dependent_step, dependencies=["failing"]))
        
        result = pipeline.execute({})
        
        assert result.status == "failed"
        assert result.step_results["failing"].status == StepStatus.FAILED
        assert "dependent" not in result.step_results or \
               result.step_results["dependent"].status == StepStatus.FAILED
        assert len(result.error_summary) > 0
        
    def test_pipeline_with_optional_step(self):
        """Test pipeline with optional failing step"""
        pipeline = Pipeline("test_pipeline")
        
        def optional_failing(context, results):
            raise ValueError("Optional step failed")
            
        def required_step(context, results):
            return {"result": "success"}
            
        pipeline.add_step(PipelineStep("optional", optional_failing, optional=True, max_retries=0))
        pipeline.add_step(PipelineStep("required", required_step))
        
        result = pipeline.execute({})
        
        assert result.status == "partial_success"
        assert result.step_results["optional"].status == StepStatus.FAILED
        assert result.step_results["required"].status == StepStatus.COMPLETED
        
    def test_pipeline_execution_order(self):
        """Test correct execution order with dependencies"""
        pipeline = Pipeline("test_pipeline")
        execution_order = []
        
        def make_step_func(name):
            def step_func(context, results):
                execution_order.append(name)
                return {"name": name}
            return step_func
            
        # Create dependency chain: A -> B -> C, D -> C
        pipeline.add_step(PipelineStep("C", make_step_func("C"), dependencies=["B", "D"]))
        pipeline.add_step(PipelineStep("B", make_step_func("B"), dependencies=["A"]))
        pipeline.add_step(PipelineStep("A", make_step_func("A")))
        pipeline.add_step(PipelineStep("D", make_step_func("D")))
        
        result = pipeline.execute({})
        
        assert result.status == "success"
        assert execution_order.index("A") < execution_order.index("B")
        assert execution_order.index("B") < execution_order.index("C")
        assert execution_order.index("D") < execution_order.index("C")
        
    def test_pipeline_retry_mechanism(self):
        """Test step retry on failure"""
        pipeline = Pipeline("test_pipeline")
        attempt_count = 0
        
        def flaky_step(context, results):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError(f"Attempt {attempt_count} failed")
            return {"attempts": attempt_count}
            
        pipeline.add_step(PipelineStep("flaky", flaky_step, max_retries=3))
        
        result = pipeline.execute({})
        
        assert result.status == "success"
        assert result.step_results["flaky"].status == StepStatus.COMPLETED
        assert result.final_output["attempts"] == 3
        
    def test_hpi_pipeline_initialization(self):
        """Test HPIPipeline initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            pipeline = HPIPipeline(data_path)
            
            # Check that all expected steps are added
            expected_steps = [
                "load_data",
                "validate_data",
                "quality_analysis",
                "outlier_detection",
                "calculate_indices",
                "aggregate_results",
                "generate_reports"
            ]
            
            for step_name in expected_steps:
                assert step_name in pipeline.steps
                
            # Check dependencies
            assert "load_data" in pipeline.steps["validate_data"].dependencies
            assert "validate_data" in pipeline.steps["outlier_detection"].dependencies
            assert "outlier_detection" in pipeline.steps["calculate_indices"].dependencies
            
    def test_pipeline_context_updates(self):
        """Test that pipeline context is updated with step outputs"""
        pipeline = Pipeline("test_pipeline")
        
        def step1(context, results):
            return {"data": [1, 2, 3]}
            
        def step2(context, results):
            # Should be able to access step1 output from context
            step1_data = context.get("step1_output", {}).get("data", [])
            return {"sum": sum(step1_data)}
            
        pipeline.add_step(PipelineStep("step1", step1))
        pipeline.add_step(PipelineStep("step2", step2, dependencies=["step1"]))
        
        result = pipeline.execute({})
        
        assert result.status == "success"
        assert result.step_results["step2"].output["sum"] == 6
        
    def test_step_result_metrics(self):
        """Test step result metrics tracking"""
        pipeline = Pipeline("test_pipeline")
        
        def slow_step(context, results):
            import time
            time.sleep(0.1)
            return {"result": "done"}
            
        pipeline.add_step(PipelineStep("slow", slow_step))
        
        result = pipeline.execute({})
        
        assert result.status == "success"
        step_result = result.step_results["slow"]
        assert step_result.start_time < step_result.end_time
        duration = (step_result.end_time - step_result.start_time).total_seconds()
        assert duration >= 0.1
        
    def test_pipeline_result_serialization(self):
        """Test that pipeline results can be serialized"""
        pipeline = Pipeline("test_pipeline")
        
        def simple_step(context, results):
            return {"value": 42, "text": "test"}
            
        pipeline.add_step(PipelineStep("simple", simple_step))
        
        result = pipeline.execute({"pipeline_id": "test123"})
        
        # Check that result attributes are serializable
        assert isinstance(result.pipeline_id, str)
        assert isinstance(result.start_time, datetime)
        assert isinstance(result.end_time, datetime)
        assert isinstance(result.status, str)
        assert isinstance(result.metrics, dict)