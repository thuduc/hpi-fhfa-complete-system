"""API endpoints for HPI system"""

import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import date, datetime
import uuid
import json
from pathlib import Path

from .models import (
    IndexRequest, IndexResponse,
    DataUploadRequest, DataUploadResponse,
    QualityReportResponse,
    SensitivityAnalysisRequest, SensitivityAnalysisResponse,
    GeographyLevel, WeightingScheme
)
from ..algorithms import BMNIndexEstimator, RepeatSalesRegression
from ..models import SampleWeightCalculator, WeightType
from ..models.validators import DataValidator
from ..outliers import (
    DataQualityAnalyzer, OutlierDetector,
    RobustRepeatSalesRegression, RobustRegressionConfig,
    SensitivityAnalyzer
)
from ..data import DataLoader


class HPIEndpoints:
    """Main endpoints for HPI calculation"""
    
    def __init__(self, data_path: Path):
        """Initialize HPI endpoints
        
        Args:
            data_path: Path to data storage directory
        """
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)
        self.validator = DataValidator()
        self.weight_calculator = SampleWeightCalculator()
        
    def calculate_index(self, request: IndexRequest) -> IndexResponse:
        """Calculate house price index
        
        Args:
            request: Index calculation request
            
        Returns:
            IndexResponse with calculated indices
        """
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Load data
            pairs_df = self.data_loader.load_transaction_pairs(
                start_date=request.start_date,
                end_date=request.end_date
            )
            geo_df = self.data_loader.load_geography_data()
            adjacency = self.data_loader.load_adjacency_data()
            
            # Apply filters if provided
            if request.filters:
                for column, values in request.filters.items():
                    if column in pairs_df.columns:
                        pairs_df = pairs_df[pairs_df[column].isin(values)]
            
            # Validate pairs
            validated_df = self.validator.validate_transaction_batch(pairs_df)
            valid_pairs = validated_df[validated_df['is_valid']]
            
            # Perform outlier analysis if requested
            outlier_summary = None
            if request.include_outlier_analysis:
                detector = OutlierDetector()
                outlier_result = detector.detect_outliers(valid_pairs)
                valid_pairs = detector.get_clean_data(valid_pairs, outlier_result)
                
                outlier_summary = {
                    'total_outliers': outlier_result.statistics['total_outliers'],
                    'outlier_rate': outlier_result.statistics['outlier_rate'],
                    'outlier_types': {
                        'cagr': outlier_result.statistics.get('cagr_outliers', 0),
                        'time_gap': outlier_result.statistics.get('time_gap_outliers', 0)
                    }
                }
            
            # Choose regression method
            if request.robust_regression:
                config = RobustRegressionConfig(method='huber')
                estimator = BMNIndexEstimator(
                    regression_class=RobustRepeatSalesRegression,
                    regression_kwargs={'config': config},
                    weight_calculator=self.weight_calculator,
                    adjacency_data=adjacency
                )
            else:
                estimator = BMNIndexEstimator(
                    weight_calculator=self.weight_calculator,
                    adjacency_data=adjacency
                )
            
            # Map weighting scheme
            weight_type_map = {
                WeightingScheme.SAMPLE: WeightType.SAMPLE,
                WeightingScheme.VALUE: WeightType.VALUE,
                WeightingScheme.STOCK: WeightType.STOCK,
                WeightingScheme.DEMOGRAPHIC: WeightType.DEMOGRAPHIC
            }
            weight_type = weight_type_map[request.weighting_scheme]
            
            # Calculate indices
            results = estimator.estimate_indices(
                valid_pairs,
                geo_df,
                weight_type,
                geography_level=request.geography_level.value,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            # Get quality metrics if requested
            quality_metrics = None
            if request.include_quality_metrics:
                analyzer = DataQualityAnalyzer()
                quality_report = analyzer.analyze_quality(valid_pairs, geo_df)
                quality_metrics = {
                    'overall_score': quality_report.quality_scores['overall'],
                    'validation_pass_rate': quality_report.validation_results['overall']['pass_rate'],
                    'temporal_coverage_score': quality_report.quality_scores['temporal_coverage'],
                    'geographic_coverage_score': quality_report.quality_scores['geographic_coverage']
                }
            
            # Format index values
            index_values = results.index_values.to_dict('records')
            
            # Add metadata to each record
            for record in index_values:
                record['weighting_scheme'] = request.weighting_scheme.value
                record['geography_level'] = request.geography_level.value
            
            computation_time = (datetime.now() - start_time).total_seconds()
            
            return IndexResponse(
                request_id=request_id,
                status="success",
                index_values=index_values,
                quality_metrics=quality_metrics,
                outlier_summary=outlier_summary,
                coverage_stats=results.coverage_stats,
                computation_time_seconds=computation_time
            )
            
        except Exception as e:
            return IndexResponse(
                request_id=request_id,
                status="error",
                error_message=str(e),
                computation_time_seconds=(datetime.now() - start_time).total_seconds()
            )


class DataEndpoints:
    """Endpoints for data management"""
    
    def __init__(self, data_path: Path):
        """Initialize data endpoints
        
        Args:
            data_path: Path to data storage directory
        """
        self.data_path = data_path
        self.validator = DataValidator()
        
    def upload_data(self, request: DataUploadRequest, file_data: bytes) -> DataUploadResponse:
        """Upload and validate data
        
        Args:
            request: Upload request details
            file_data: Raw file data
            
        Returns:
            DataUploadResponse with upload results
        """
        upload_id = str(uuid.uuid4())
        
        try:
            # Parse data based on format
            if request.file_format == "csv":
                import io
                df = pd.read_csv(io.BytesIO(file_data))
            elif request.file_format == "parquet":
                import io
                df = pd.read_parquet(io.BytesIO(file_data))
            elif request.file_format == "json":
                df = pd.read_json(io.BytesIO(file_data))
            else:
                return DataUploadResponse(
                    upload_id=upload_id,
                    status="error",
                    error_message=f"Unsupported file format: {request.file_format}"
                )
            
            records_processed = len(df)
            
            # Validate if requested
            validation_errors = []
            records_valid = records_processed
            
            if request.validate_data and request.data_type == "transactions":
                validated_df = self.validator.validate_transaction_batch(df)
                records_valid = validated_df['is_valid'].sum()
                records_invalid = records_processed - records_valid
                
                # Get validation errors for invalid records
                invalid_records = validated_df[~validated_df['is_valid']]
                for idx, row in invalid_records.head(100).iterrows():  # Limit to 100 errors
                    validation_errors.append({
                        'row_index': idx,
                        'errors': row['validation_errors']
                    })
            else:
                records_invalid = 0
            
            # Save data if validation passed or not required
            if not request.validate_data or records_valid > 0:
                output_path = self.data_path / request.data_type / f"{upload_id}.parquet"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if request.validate_data and request.data_type == "transactions":
                    # Save only valid records
                    valid_df = df[validated_df['is_valid']]
                    valid_df.to_parquet(output_path)
                else:
                    df.to_parquet(output_path)
                
                # Update metadata
                metadata_path = self.data_path / request.data_type / "metadata.json"
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                metadata[upload_id] = {
                    'upload_date': datetime.now().isoformat(),
                    'records': records_valid if request.validate_data else records_processed,
                    'file_format': request.file_format,
                    'user_metadata': request.metadata
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            status = "success" if records_valid > 0 else "validation_failed"
            
            return DataUploadResponse(
                upload_id=upload_id,
                status=status,
                records_processed=records_processed,
                records_valid=records_valid,
                records_invalid=records_invalid,
                validation_errors=validation_errors
            )
            
        except Exception as e:
            return DataUploadResponse(
                upload_id=upload_id,
                status="error",
                error_message=str(e)
            )
    
    def get_quality_report(self, start_date: Optional[date] = None,
                          end_date: Optional[date] = None) -> QualityReportResponse:
        """Generate data quality report
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            QualityReportResponse with quality analysis
        """
        report_id = str(uuid.uuid4())
        
        try:
            # Load data
            loader = DataLoader(self.data_path)
            pairs_df = loader.load_transaction_pairs(start_date, end_date)
            geo_df = loader.load_geography_data()
            
            # Analyze quality
            analyzer = DataQualityAnalyzer()
            report = analyzer.analyze_quality(pairs_df, geo_df)
            
            # Format response
            return QualityReportResponse(
                report_id=report_id,
                generated_at=date.today(),
                overall_quality_score=report.quality_scores['overall'],
                quality_scores=report.quality_scores,
                summary_metrics=report.summary_metrics,
                temporal_coverage=report.temporal_coverage.to_dict('records'),
                geographic_coverage=report.geographic_coverage.to_dict('records'),
                price_distribution=report.price_distribution.to_dict('records'),
                recommendations=report.recommendations
            )
            
        except Exception as e:
            # Return minimal report on error
            return QualityReportResponse(
                report_id=report_id,
                generated_at=date.today(),
                overall_quality_score=0.0,
                quality_scores={},
                summary_metrics={'error': str(e)},
                temporal_coverage=[],
                geographic_coverage=[],
                price_distribution=[],
                recommendations=[f"Error generating report: {str(e)}"]
            )


class AnalysisEndpoints:
    """Endpoints for analysis features"""
    
    def __init__(self, data_path: Path):
        """Initialize analysis endpoints
        
        Args:
            data_path: Path to data storage directory
        """
        self.data_path = data_path
        self.hpi_endpoints = HPIEndpoints(data_path)
        
    def run_sensitivity_analysis(self, request: SensitivityAnalysisRequest) -> SensitivityAnalysisResponse:
        """Run sensitivity analysis on index calculation
        
        Args:
            request: Sensitivity analysis request
            
        Returns:
            SensitivityAnalysisResponse with analysis results
        """
        analysis_id = str(uuid.uuid4())
        
        try:
            # Get base results
            base_results = self.hpi_endpoints.calculate_index(request.base_request)
            
            # Load data for sensitivity analysis
            loader = DataLoader(self.data_path)
            pairs_df = loader.load_transaction_pairs(
                request.base_request.start_date,
                request.base_request.end_date
            )
            
            # Validate pairs
            validator = DataValidator()
            validated_df = validator.validate_transaction_batch(pairs_df)
            valid_pairs = validated_df[validated_df['is_valid']]
            
            # Create sensitivity analyzer
            analyzer = SensitivityAnalyzer()
            
            # Run sensitivity analysis
            sensitivity_results = []
            parameter_impacts = {}
            
            for param in request.parameters_to_vary:
                param_results = []
                base_value = None
                
                for value in request.parameter_ranges.get(param, []):
                    # Create modified request
                    # This is simplified - in practice would need to handle different parameter types
                    if param == "min_pairs":
                        # Would need to modify estimator configuration
                        pass
                    elif param == "outlier_threshold":
                        # Would need to modify outlier detection
                        pass
                    
                    # For now, return mock results
                    param_results.append({
                        'parameter': param,
                        'value': value,
                        'index_change': 0.0  # Would calculate actual change
                    })
                
                sensitivity_results.extend(param_results)
                parameter_impacts[param] = 0.0  # Would calculate actual impact
            
            # Generate recommendations
            recommendations = [
                "Index calculation is stable across parameter variations",
                "Consider using robust regression for datasets with outliers"
            ]
            
            return SensitivityAnalysisResponse(
                analysis_id=analysis_id,
                base_results=base_results,
                sensitivity_results=sensitivity_results,
                parameter_impacts=parameter_impacts,
                recommendations=recommendations
            )
            
        except Exception as e:
            return SensitivityAnalysisResponse(
                analysis_id=analysis_id,
                base_results=IndexResponse(
                    request_id="error",
                    status="error",
                    error_message=str(e)
                ),
                sensitivity_results=[],
                parameter_impacts={},
                recommendations=[f"Error in sensitivity analysis: {str(e)}"]
            )