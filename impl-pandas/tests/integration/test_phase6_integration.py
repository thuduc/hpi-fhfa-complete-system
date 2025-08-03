"""Integration tests for Phase 6 - API and Pipeline features"""

import pytest
from pathlib import Path
import tempfile
import json
import time
from datetime import date, datetime, timedelta
import pandas as pd
import numpy as np

from hpi_fhfa.api import create_app, HPIEndpoints, DataEndpoints
from hpi_fhfa.api.models import (
    IndexRequest, DataUploadRequest,
    GeographyLevel, WeightingScheme, OutputFormat
)
from hpi_fhfa.pipeline import (
    HPIPipeline, BatchProcessor, BatchJob, JobPriority,
    PipelineMonitor, ResultCache
)
from hpi_fhfa.pipeline.cache import MemoryCacheBackend
from hpi_fhfa.data import SyntheticDataGenerator, GeographicDataGenerator


class TestPhase6Integration:
    """Integration tests for Phase 6 functionality"""
    
    @pytest.fixture
    def test_data_path(self):
        """Create temporary data directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            
            # Generate test data
            syn_gen = SyntheticDataGenerator(seed=42)
            geo_gen = GeographicDataGenerator(seed=42)
            
            # Create smaller dataset for testing
            transaction_data = syn_gen.generate_complete_dataset(
                start_year=2019,
                end_year=2020,
                num_cbsas=3,
                num_tracts=30,
                num_properties=300,
                target_pairs=200
            )
            
            geo_data = geo_gen.generate_complete_geographic_data(
                num_tracts=30
            )
            
            # Save data
            transactions_dir = data_path / "transactions"
            transactions_dir.mkdir(parents=True)
            transaction_data['pairs'].to_parquet(
                transactions_dir / "test_pairs.parquet"
            )
            
            geography_dir = data_path / "geography"
            geography_dir.mkdir(parents=True)
            geo_data['tracts'].to_parquet(
                geography_dir / "tracts.parquet"
            )
            
            # Save adjacency
            adjacency_file = geography_dir / "adjacency.json"
            with open(adjacency_file, 'w') as f:
                json.dump(geo_data['adjacency'], f)
            
            yield data_path
            
    @pytest.fixture
    def flask_app(self, test_data_path):
        """Create Flask app for testing"""
        app = create_app(test_data_path)
        app.config['TESTING'] = True
        return app
        
    def test_api_health_check(self, flask_app):
        """Test API health check endpoint"""
        with flask_app.test_client() as client:
            response = client.get('/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'healthy'
            assert 'timestamp' in data
            assert 'version' in data
            
    def test_api_index_calculation(self, flask_app):
        """Test index calculation via API"""
        with flask_app.test_client() as client:
            request_data = {
                'start_date': '2019-01-01',
                'end_date': '2020-12-31',
                'geography_level': 'cbsa',
                'weighting_scheme': 'sample',
                'include_quality_metrics': True,
                'include_outlier_analysis': True
            }
            
            response = client.post(
                '/api/v1/index/calculate',
                json=request_data,
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['status'] == 'success'
            assert 'index_values' in data
            assert len(data['index_values']) > 0
            assert 'quality_metrics' in data
            assert 'coverage_stats' in data
            
    def test_api_data_upload(self, flask_app):
        """Test data upload via API"""
        with flask_app.test_client() as client:
            # Create test CSV data
            test_df = pd.DataFrame({
                'property_id': ['P001', 'P002'],
                'tract_id': ['T001', 'T001'],
                'cbsa_id': ['C01', 'C01'],
                'first_sale_date': ['2019-01-01', '2019-02-01'],
                'first_sale_price': [200000, 250000],
                'second_sale_date': ['2020-01-01', '2020-02-01'],
                'second_sale_price': [220000, 270000]
            })
            
            csv_data = test_df.to_csv(index=False).encode('utf-8')
            
            response = client.post(
                '/api/v1/data/upload',
                data={
                    'data_type': 'transactions',
                    'file_format': 'csv',
                    'validate_data': 'true',
                    'metadata': json.dumps({'source': 'test'})
                },
                files={'file': ('test.csv', csv_data, 'text/csv')}
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['status'] == 'success'
            assert data['records_processed'] == 2
            assert data['records_valid'] == 2
            assert data['records_invalid'] == 0
            
    def test_api_quality_report(self, flask_app):
        """Test quality report generation via API"""
        with flask_app.test_client() as client:
            response = client.get('/api/v1/data/quality')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert 'overall_quality_score' in data
            assert 'quality_scores' in data
            assert 'summary_metrics' in data
            assert 'recommendations' in data
            
    def test_api_csv_output_format(self, flask_app):
        """Test CSV output format"""
        with flask_app.test_client() as client:
            request_data = {
                'start_date': '2019-01-01',
                'end_date': '2019-12-31',
                'geography_level': 'cbsa',
                'weighting_scheme': 'sample',
                'output_format': 'csv'
            }
            
            response = client.post(
                '/api/v1/index/calculate',
                json=request_data,
                content_type='application/json'
            )
            
            assert response.status_code == 200
            assert response.content_type.startswith('text/csv')
            
            # Parse CSV response
            import io
            csv_data = io.StringIO(response.data.decode('utf-8'))
            df = pd.read_csv(csv_data)
            
            assert len(df) > 0
            assert 'date' in df.columns
            assert 'index_value' in df.columns
            
    def test_pipeline_end_to_end(self, test_data_path):
        """Test complete pipeline execution"""
        pipeline = HPIPipeline(test_data_path)
        
        context = {
            'start_date': date(2019, 1, 1),
            'end_date': date(2020, 12, 31),
            'geography_level': 'cbsa',
            'weight_type': 'sample',
            'robust_regression': False,
            'output_path': test_data_path / 'output'
        }
        
        result = pipeline.execute(context)
        
        assert result.status == 'success'
        assert len(result.step_results) >= 6  # At least 6 main steps
        
        # Check key steps completed
        assert 'load_data' in result.step_results
        assert 'validate_data' in result.step_results
        assert 'calculate_indices' in result.step_results
        assert 'aggregate_results' in result.step_results
        
        # Check output files were created
        output_path = test_data_path / 'output'
        assert (output_path / 'index_values.csv').exists()
        
    def test_batch_processing_integration(self, test_data_path):
        """Test batch processing of multiple jobs"""
        processor = BatchProcessor(
            max_workers=2,
            result_path=test_data_path / 'batch_results'
        )
        
        # Register pipeline
        pipeline = HPIPipeline(test_data_path)
        processor.register_pipeline('hpi_pipeline', pipeline)
        
        # Create multiple jobs for different periods
        jobs = []
        for year in [2019, 2020]:
            for quarter in range(1, 5):
                start_date = date(year, (quarter-1)*3 + 1, 1)
                if quarter == 4:
                    end_date = date(year, 12, 31)
                else:
                    end_date = date(year, quarter*3, 1) - timedelta(days=1)
                    
                job = BatchJob(
                    job_id=f"hpi_{year}_q{quarter}",
                    name=f"HPI {year} Q{quarter}",
                    pipeline='hpi_pipeline',
                    context={
                        'start_date': start_date,
                        'end_date': end_date,
                        'geography_level': 'cbsa',
                        'weight_type': 'sample'
                    },
                    priority=JobPriority.NORMAL
                )
                jobs.append(job)
                
        # Submit jobs
        job_ids = processor.submit_batch(jobs)
        assert len(job_ids) == 8  # 2 years × 4 quarters
        
        # Wait for processing
        time.sleep(5)
        
        # Check results
        completed = 0
        for job_id in job_ids:
            status = processor.queue.get_job_status(job_id)
            if status and status.status.value == 'completed':
                completed += 1
                
        assert completed > 0  # At least some should complete
        
        # Check result files
        result_files = list((test_data_path / 'batch_results').glob('*.json'))
        assert len(result_files) > 0
        
        processor.stop()
        
    def test_monitoring_integration(self, test_data_path):
        """Test monitoring system integration"""
        monitor = PipelineMonitor()
        
        # Add health checks
        monitor.add_health_check('data_available', lambda: True)
        monitor.add_health_check('disk_space', lambda: True)
        
        # Start monitoring
        monitor.start()
        
        # Simulate pipeline executions
        for i in range(5):
            success = i < 4  # Last one fails
            monitor.record_pipeline_execution(
                pipeline_name='test_pipeline',
                duration_seconds=np.random.uniform(1, 5),
                success=success,
                error="Test error" if not success else None
            )
            
        # Simulate API requests
        for i in range(10):
            monitor.record_api_request(
                endpoint='/api/v1/index/calculate',
                method='POST',
                duration_seconds=np.random.uniform(0.1, 2),
                status_code=200 if i < 9 else 500
            )
            
        # Get metrics summary
        metrics = monitor.metrics.get_metrics_summary()
        assert len(metrics) > 0
        
        # Check alerts
        alerts = monitor.alerts.get_active_alerts()
        assert any(a.message.startswith("Pipeline 'test_pipeline' failed") for a in alerts)
        
        # Check health status
        health = monitor.get_health_status()
        assert health.status == 'healthy'
        assert all(status == 'healthy' for status in health.components.values())
        
        monitor.stop()
        
    def test_caching_integration(self, test_data_path):
        """Test caching with API endpoints"""
        cache = ResultCache(backend=MemoryCacheBackend())
        endpoints = HPIEndpoints(test_data_path)
        
        # Monkey patch to use cache
        original_calculate = endpoints.calculate_index
        
        def cached_calculate(request):
            # Check cache first
            cached = cache.get_index_results(
                request.start_date,
                request.end_date,
                request.geography_level.value,
                request.weighting_scheme.value,
                request.filters
            )
            
            if cached is not None:
                # Return cached result
                from hpi_fhfa.api.models import IndexResponse
                return IndexResponse(
                    request_id="cached",
                    status="success",
                    index_values=cached.to_dict('records')
                )
                
            # Calculate and cache
            result = original_calculate(request)
            
            if result.status == "success" and result.index_values:
                df = pd.DataFrame(result.index_values)
                cache.set_index_results(
                    df,
                    request.start_date,
                    request.end_date,
                    request.geography_level.value,
                    request.weighting_scheme.value,
                    request.filters
                )
                
            return result
            
        endpoints.calculate_index = cached_calculate
        
        # First request - should calculate
        request = IndexRequest(
            start_date=date(2019, 1, 1),
            end_date=date(2019, 12, 31),
            geography_level=GeographyLevel.CBSA,
            weighting_scheme=WeightingScheme.SAMPLE
        )
        
        result1 = endpoints.calculate_index(request)
        assert result1.request_id != "cached"
        
        # Second request - should use cache
        result2 = endpoints.calculate_index(request)
        assert result2.request_id == "cached"
        
        # Check cache stats
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['hit_rate'] > 0
        
    def test_api_error_handling(self, flask_app):
        """Test API error handling"""
        with flask_app.test_client() as client:
            # Invalid date format
            response = client.post(
                '/api/v1/index/calculate',
                json={
                    'start_date': 'invalid-date',
                    'end_date': '2020-12-31'
                }
            )
            
            assert response.status_code == 500
            data = json.loads(response.data)
            assert data['status'] == 'error'
            assert 'error_message' in data
            
            # Missing required fields
            response = client.post(
                '/api/v1/index/calculate',
                json={'geography_level': 'cbsa'}
            )
            
            assert response.status_code == 500
            
            # Invalid file upload
            response = client.post(
                '/api/v1/data/upload',
                data={'data_type': 'transactions'}
                # Missing file
            )
            
            assert response.status_code == 400
            
    def test_recurring_batch_job(self, test_data_path):
        """Test recurring batch job functionality"""
        processor = BatchProcessor(
            max_workers=1,
            result_path=test_data_path / 'batch_results'
        )
        
        # Simple pipeline that tracks executions
        execution_count = 0
        
        from hpi_fhfa.pipeline.orchestrator import Pipeline, PipelineStep
        
        def counting_step(context, results):
            nonlocal execution_count
            execution_count += 1
            return {"count": execution_count}
            
        pipeline = Pipeline("recurring_test")
        pipeline.add_step(PipelineStep("count", counting_step))
        processor.register_pipeline("recurring_test", pipeline)
        
        # Create recurring job
        def context_generator():
            return {"timestamp": datetime.now().isoformat()}
            
        recurring_id = processor.create_recurring_job(
            name="Daily HPI",
            pipeline="recurring_test",
            context_generator=context_generator,
            schedule=timedelta(seconds=1),  # Run every second for testing
            priority=JobPriority.HIGH
        )
        
        # Wait for multiple executions
        time.sleep(3.5)
        
        # Should have executed multiple times
        assert execution_count >= 3
        
        processor.stop()