"""Unit tests for API models"""

import pytest
from datetime import date
from hpi_fhfa.api.models import (
    IndexRequest, IndexResponse, DataUploadRequest, DataUploadResponse,
    QualityReportResponse, SensitivityAnalysisRequest, SensitivityAnalysisResponse,
    GeographyLevel, WeightingScheme, OutputFormat
)


class TestAPIModels:
    """Test API model classes"""
    
    def test_index_request_creation(self):
        """Test IndexRequest creation"""
        request = IndexRequest(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 12, 31),
            geography_level=GeographyLevel.CBSA,
            weighting_scheme=WeightingScheme.SAMPLE
        )
        
        assert request.start_date == date(2020, 1, 1)
        assert request.end_date == date(2021, 12, 31)
        assert request.geography_level == GeographyLevel.CBSA
        assert request.weighting_scheme == WeightingScheme.SAMPLE
        assert request.include_quality_metrics is True  # Default
        assert request.include_outlier_analysis is True  # Default
        assert request.robust_regression is False  # Default
        assert request.output_format == OutputFormat.JSON  # Default
        
    def test_index_request_with_filters(self):
        """Test IndexRequest with filters"""
        filters = {"cbsa_id": ["12420", "12580"], "state": ["CA"]}
        request = IndexRequest(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 12, 31),
            filters=filters
        )
        
        assert request.filters == filters
        assert "cbsa_id" in request.filters
        assert len(request.filters["cbsa_id"]) == 2
        
    def test_index_response_success(self):
        """Test successful IndexResponse"""
        response = IndexResponse(
            request_id="test-123",
            status="success",
            index_values=[{"date": "2020-01-01", "value": 100.0}],
            quality_metrics={"overall_score": 0.85},
            computation_time_seconds=5.2
        )
        
        assert response.request_id == "test-123"
        assert response.status == "success"
        assert len(response.index_values) == 1
        assert response.quality_metrics["overall_score"] == 0.85
        assert response.error_message is None
        
    def test_index_response_error(self):
        """Test error IndexResponse"""
        response = IndexResponse(
            request_id="test-456",
            status="error",
            error_message="Database connection failed"
        )
        
        assert response.status == "error"
        assert response.error_message == "Database connection failed"
        assert response.index_values is None
        
    def test_data_upload_request(self):
        """Test DataUploadRequest"""
        request = DataUploadRequest(
            data_type="transactions",
            file_format="csv",
            validate_data=True,
            replace_existing=False,
            metadata={"source": "test", "version": "1.0"}
        )
        
        assert request.data_type == "transactions"
        assert request.file_format == "csv"
        assert request.validate_data is True
        assert request.replace_existing is False
        assert request.metadata["source"] == "test"
        
    def test_data_upload_response(self):
        """Test DataUploadResponse"""
        response = DataUploadResponse(
            upload_id="upload-789",
            status="success",
            records_processed=1000,
            records_valid=950,
            records_invalid=50,
            validation_errors=[{"row": 1, "error": "Invalid price"}]
        )
        
        assert response.upload_id == "upload-789"
        assert response.status == "success"
        assert response.records_processed == 1000
        assert response.records_valid == 950
        assert response.records_invalid == 50
        assert len(response.validation_errors) == 1
        
    def test_quality_report_response(self):
        """Test QualityReportResponse"""
        response = QualityReportResponse(
            report_id="report-111",
            generated_at=date.today(),
            overall_quality_score=0.92,
            quality_scores={"validation": 0.95, "coverage": 0.89},
            summary_metrics={"total_pairs": 10000},
            temporal_coverage=[{"date": "2020-01", "coverage": 0.85}],
            geographic_coverage=[{"cbsa": "12420", "coverage": 0.90}],
            price_distribution=[{"metric": "mean", "value": 250000}],
            recommendations=["Improve temporal coverage for Q1 2020"]
        )
        
        assert response.report_id == "report-111"
        assert response.overall_quality_score == 0.92
        assert response.quality_scores["validation"] == 0.95
        assert len(response.recommendations) == 1
        
    def test_sensitivity_analysis_request(self):
        """Test SensitivityAnalysisRequest"""
        base_request = IndexRequest(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 12, 31)
        )
        
        request = SensitivityAnalysisRequest(
            base_request=base_request,
            parameters_to_vary=["min_pairs", "outlier_threshold"],
            parameter_ranges={
                "min_pairs": [20, 30, 40],
                "outlier_threshold": [2.0, 3.0, 4.0]
            },
            include_visualizations=True
        )
        
        assert request.base_request.start_date == date(2020, 1, 1)
        assert len(request.parameters_to_vary) == 2
        assert "min_pairs" in request.parameters_to_vary
        assert len(request.parameter_ranges["min_pairs"]) == 3
        assert request.include_visualizations is True
        
    def test_geography_level_enum(self):
        """Test GeographyLevel enum"""
        assert GeographyLevel.TRACT.value == "tract"
        assert GeographyLevel.CBSA.value == "cbsa"
        assert GeographyLevel.STATE.value == "state"
        assert GeographyLevel.NATIONAL.value == "national"
        
        # Test enum creation from string
        level = GeographyLevel("cbsa")
        assert level == GeographyLevel.CBSA
        
    def test_weighting_scheme_enum(self):
        """Test WeightingScheme enum"""
        assert WeightingScheme.SAMPLE.value == "sample"
        assert WeightingScheme.VALUE.value == "value"
        assert WeightingScheme.STOCK.value == "stock"
        assert WeightingScheme.DEMOGRAPHIC.value == "demographic"
        
    def test_output_format_enum(self):
        """Test OutputFormat enum"""
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.CSV.value == "csv"
        assert OutputFormat.PARQUET.value == "parquet"