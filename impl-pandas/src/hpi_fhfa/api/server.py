"""Flask-based REST API server for HPI system"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from pathlib import Path
from datetime import date, datetime
import json
import logging
from typing import Optional, Dict, Any
import io

from .endpoints import HPIEndpoints, DataEndpoints, AnalysisEndpoints
from .models import (
    IndexRequest, DataUploadRequest, SensitivityAnalysisRequest,
    GeographyLevel, WeightingScheme, OutputFormat
)


def create_app(data_path: Path, config: Optional[Dict[str, Any]] = None) -> Flask:
    """Create Flask application
    
    Args:
        data_path: Path to data storage directory
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Apply configuration
    if config:
        app.config.update(config)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize endpoints
    hpi_endpoints = HPIEndpoints(data_path)
    data_endpoints = DataEndpoints(data_path)
    analysis_endpoints = AnalysisEndpoints(data_path)
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    
    # Index calculation endpoint
    @app.route('/api/v1/index/calculate', methods=['POST'])
    def calculate_index():
        """Calculate house price index"""
        try:
            data = request.get_json()
            
            # Parse request
            index_request = IndexRequest(
                start_date=date.fromisoformat(data['start_date']),
                end_date=date.fromisoformat(data['end_date']),
                geography_level=GeographyLevel(data.get('geography_level', 'cbsa')),
                weighting_scheme=WeightingScheme(data.get('weighting_scheme', 'sample')),
                include_quality_metrics=data.get('include_quality_metrics', True),
                include_outlier_analysis=data.get('include_outlier_analysis', True),
                robust_regression=data.get('robust_regression', False),
                output_format=OutputFormat(data.get('output_format', 'json')),
                filters=data.get('filters')
            )
            
            # Calculate index
            response = hpi_endpoints.calculate_index(index_request)
            
            # Format based on output format
            if index_request.output_format == OutputFormat.CSV:
                # Convert to CSV
                import pandas as pd
                df = pd.DataFrame(response.index_values)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                return send_file(
                    io.BytesIO(csv_buffer.getvalue().encode()),
                    mimetype='text/csv',
                    as_attachment=True,
                    download_name=f'hpi_index_{response.request_id}.csv'
                )
            else:
                # Return JSON
                return jsonify(response.__dict__)
                
        except Exception as e:
            logger.error(f"Error calculating index: {str(e)}")
            return jsonify({
                'status': 'error',
                'error_message': str(e)
            }), 500
    
    # Data upload endpoint
    @app.route('/api/v1/data/upload', methods=['POST'])
    def upload_data():
        """Upload data to the system"""
        try:
            # Get file from request
            if 'file' not in request.files:
                return jsonify({
                    'status': 'error',
                    'error_message': 'No file provided'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'status': 'error',
                    'error_message': 'No file selected'
                }), 400
            
            # Get request parameters
            upload_request = DataUploadRequest(
                data_type=request.form.get('data_type', 'transactions'),
                file_format=request.form.get('file_format', 'csv'),
                validate_data=request.form.get('validate_data', 'true').lower() == 'true',
                replace_existing=request.form.get('replace_existing', 'false').lower() == 'true',
                metadata=json.loads(request.form.get('metadata', '{}'))
            )
            
            # Process upload
            file_data = file.read()
            response = data_endpoints.upload_data(upload_request, file_data)
            
            return jsonify(response.__dict__)
            
        except Exception as e:
            logger.error(f"Error uploading data: {str(e)}")
            return jsonify({
                'status': 'error',
                'error_message': str(e)
            }), 500
    
    # Data quality report endpoint
    @app.route('/api/v1/data/quality', methods=['GET'])
    def get_quality_report():
        """Get data quality report"""
        try:
            # Get optional date filters
            start_date = None
            end_date = None
            
            if 'start_date' in request.args:
                start_date = date.fromisoformat(request.args['start_date'])
            if 'end_date' in request.args:
                end_date = date.fromisoformat(request.args['end_date'])
            
            # Generate report
            response = data_endpoints.get_quality_report(start_date, end_date)
            
            return jsonify(response.__dict__)
            
        except Exception as e:
            logger.error(f"Error generating quality report: {str(e)}")
            return jsonify({
                'status': 'error',
                'error_message': str(e)
            }), 500
    
    # Sensitivity analysis endpoint
    @app.route('/api/v1/analysis/sensitivity', methods=['POST'])
    def run_sensitivity_analysis():
        """Run sensitivity analysis"""
        try:
            data = request.get_json()
            
            # Parse base request
            base_request = IndexRequest(
                start_date=date.fromisoformat(data['base_request']['start_date']),
                end_date=date.fromisoformat(data['base_request']['end_date']),
                geography_level=GeographyLevel(data['base_request'].get('geography_level', 'cbsa')),
                weighting_scheme=WeightingScheme(data['base_request'].get('weighting_scheme', 'sample'))
            )
            
            # Create sensitivity request
            sensitivity_request = SensitivityAnalysisRequest(
                base_request=base_request,
                parameters_to_vary=data['parameters_to_vary'],
                parameter_ranges=data['parameter_ranges'],
                include_visualizations=data.get('include_visualizations', True)
            )
            
            # Run analysis
            response = analysis_endpoints.run_sensitivity_analysis(sensitivity_request)
            
            return jsonify(response.__dict__)
            
        except Exception as e:
            logger.error(f"Error running sensitivity analysis: {str(e)}")
            return jsonify({
                'status': 'error',
                'error_message': str(e)
            }), 500
    
    # API documentation endpoint
    @app.route('/api/v1/docs', methods=['GET'])
    def get_api_docs():
        """Get API documentation"""
        docs = {
            'version': '1.0.0',
            'endpoints': {
                '/health': {
                    'method': 'GET',
                    'description': 'Health check endpoint'
                },
                '/api/v1/index/calculate': {
                    'method': 'POST',
                    'description': 'Calculate house price index',
                    'parameters': {
                        'start_date': 'ISO date string (required)',
                        'end_date': 'ISO date string (required)',
                        'geography_level': 'tract|cbsa|state|national (default: cbsa)',
                        'weighting_scheme': 'sample|value|stock|demographic (default: sample)',
                        'include_quality_metrics': 'boolean (default: true)',
                        'include_outlier_analysis': 'boolean (default: true)',
                        'robust_regression': 'boolean (default: false)',
                        'output_format': 'json|csv|parquet (default: json)',
                        'filters': 'object with column filters (optional)'
                    }
                },
                '/api/v1/data/upload': {
                    'method': 'POST',
                    'description': 'Upload data file',
                    'parameters': {
                        'file': 'File upload (required)',
                        'data_type': 'transactions|geography|demographics',
                        'file_format': 'csv|json|parquet',
                        'validate_data': 'boolean (default: true)',
                        'replace_existing': 'boolean (default: false)',
                        'metadata': 'JSON object (optional)'
                    }
                },
                '/api/v1/data/quality': {
                    'method': 'GET',
                    'description': 'Get data quality report',
                    'parameters': {
                        'start_date': 'ISO date string (optional)',
                        'end_date': 'ISO date string (optional)'
                    }
                },
                '/api/v1/analysis/sensitivity': {
                    'method': 'POST',
                    'description': 'Run sensitivity analysis',
                    'parameters': {
                        'base_request': 'Index request object',
                        'parameters_to_vary': 'List of parameter names',
                        'parameter_ranges': 'Object mapping parameters to value ranges',
                        'include_visualizations': 'boolean (default: true)'
                    }
                }
            }
        }
        
        return jsonify(docs)
    
    return app


class HPIServer:
    """HPI API Server wrapper"""
    
    def __init__(self, data_path: Path, host: str = '0.0.0.0', port: int = 5000):
        """Initialize HPI server
        
        Args:
            data_path: Path to data storage directory
            host: Host to bind to
            port: Port to bind to
        """
        self.data_path = data_path
        self.host = host
        self.port = port
        self.app = create_app(data_path)
        
    def run(self, debug: bool = False):
        """Run the server
        
        Args:
            debug: Whether to run in debug mode
        """
        print(f"Starting HPI API server on {self.host}:{self.port}")
        print(f"Data path: {self.data_path}")
        print(f"API documentation available at: http://{self.host}:{self.port}/api/v1/docs")
        
        self.app.run(host=self.host, port=self.port, debug=debug)