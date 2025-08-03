"""API request/response models for HPI system"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import date
from enum import Enum


class GeographyLevel(str, Enum):
    """Geographic aggregation levels"""
    TRACT = "tract"
    CBSA = "cbsa"
    STATE = "state"
    NATIONAL = "national"


class WeightingScheme(str, Enum):
    """Available weighting schemes"""
    SAMPLE = "sample"
    VALUE = "value"
    STOCK = "stock"
    DEMOGRAPHIC = "demographic"


class OutputFormat(str, Enum):
    """Output format options"""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


@dataclass
class IndexRequest:
    """Request model for index calculation"""
    start_date: date
    end_date: date
    geography_level: GeographyLevel = GeographyLevel.CBSA
    weighting_scheme: WeightingScheme = WeightingScheme.SAMPLE
    include_quality_metrics: bool = True
    include_outlier_analysis: bool = True
    robust_regression: bool = False
    output_format: OutputFormat = OutputFormat.JSON
    filters: Optional[Dict[str, List[str]]] = None  # e.g., {"cbsa_id": ["12420", "12580"]}


@dataclass
class IndexResponse:
    """Response model for index calculation"""
    request_id: str
    status: str  # "success", "error", "processing"
    index_values: Optional[List[Dict[str, Any]]] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    outlier_summary: Optional[Dict[str, Any]] = None
    coverage_stats: Optional[Dict[str, Any]] = None
    computation_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataUploadRequest:
    """Request model for data upload"""
    data_type: str  # "transactions", "geography", "demographics"
    file_format: str  # "csv", "parquet", "json"
    validate_data: bool = True
    replace_existing: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataUploadResponse:
    """Response model for data upload"""
    upload_id: str
    status: str  # "success", "error", "validation_failed"
    records_processed: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    validation_errors: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class QualityReportResponse:
    """Response model for data quality report"""
    report_id: str
    generated_at: date
    overall_quality_score: float
    quality_scores: Dict[str, float]
    summary_metrics: Dict[str, Any]
    temporal_coverage: List[Dict[str, Any]]
    geographic_coverage: List[Dict[str, Any]]
    price_distribution: List[Dict[str, Any]]
    recommendations: List[str]
    detailed_report_url: Optional[str] = None


@dataclass
class SensitivityAnalysisRequest:
    """Request model for sensitivity analysis"""
    base_request: IndexRequest
    parameters_to_vary: List[str]  # e.g., ["min_pairs", "outlier_threshold"]
    parameter_ranges: Dict[str, List[Any]]  # e.g., {"min_pairs": [20, 30, 40]}
    include_visualizations: bool = True


@dataclass
class SensitivityAnalysisResponse:
    """Response model for sensitivity analysis"""
    analysis_id: str
    base_results: IndexResponse
    sensitivity_results: List[Dict[str, Any]]
    parameter_impacts: Dict[str, float]  # Parameter -> impact score
    visualizations: Optional[Dict[str, str]] = None  # Type -> URL/base64
    recommendations: List[str] = field(default_factory=list)


@dataclass
class BatchProcessRequest:
    """Request model for batch processing"""
    job_name: str
    requests: List[IndexRequest]
    priority: str = "normal"  # "low", "normal", "high"
    notification_email: Optional[str] = None
    parallel_jobs: int = 1


@dataclass
class BatchProcessResponse:
    """Response model for batch processing"""
    batch_id: str
    status: str  # "queued", "processing", "completed", "failed"
    total_jobs: int
    completed_jobs: int = 0
    failed_jobs: int = 0
    estimated_completion_time: Optional[str] = None
    results_url: Optional[str] = None


@dataclass
class MonitoringStats:
    """System monitoring statistics"""
    total_requests_processed: int
    average_response_time_seconds: float
    cache_hit_rate: float
    active_batch_jobs: int
    system_health: str  # "healthy", "degraded", "unhealthy"
    last_update: date
    resource_usage: Dict[str, float]  # CPU, memory, disk
    error_rate_24h: float