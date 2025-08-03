"""Main CLI entry point for HPI-FHFA"""

import click
import yaml
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Dict, Any

from ..api.server import create_app
from ..pipeline.batch import BatchProcessor, BatchJob, JobPriority
from ..models import HPICalculator
from ..data_loader import DataLoader
from ..data_generation import SyntheticDataGenerator
from ..quality import QualityAnalyzer
from ..utils.config import load_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hpi-cli')


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool):
    """HPI-FHFA Command Line Interface"""
    ctx.ensure_object(dict)
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj['config'] = {}
    
    # Set debug mode
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        ctx.obj['debug'] = True


@cli.command()
@click.option('--host', '-h', default='0.0.0.0', help='Server host')
@click.option('--port', '-p', default=8000, type=int, help='Server port')
@click.option('--workers', '-w', default=4, type=int, help='Number of workers')
@click.option('--data-path', '-d', type=click.Path(exists=True), required=True, help='Data directory path')
@click.pass_context
def serve(ctx, host: str, port: int, workers: int, data_path: str):
    """Start the HPI-FHFA API server"""
    import uvicorn
    
    logger.info(f"Starting API server on {host}:{port}")
    
    # Create Flask app
    app = create_app(Path(data_path), ctx.obj.get('config'))
    
    # Run with uvicorn for production
    uvicorn.run(
        "hpi_fhfa.api.server:create_app",
        host=host,
        port=port,
        workers=workers,
        log_level="info" if not ctx.obj.get('debug') else "debug",
        reload=ctx.obj.get('debug', False)
    )


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input transaction file')
@click.option('--output', '-o', type=click.Path(), required=True, help='Output index file')
@click.option('--start-date', '-s', type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option('--end-date', '-e', type=click.DateTime(formats=["%Y-%m-%d"]), required=True)
@click.option('--geography', '-g', type=click.Choice(['tract', 'cbsa', 'state', 'national']), default='cbsa')
@click.option('--weighting', '-w', type=click.Choice(['sample', 'value', 'unit', 'upb', 'college', 'non_white']), default='sample')
@click.option('--format', '-f', type=click.Choice(['csv', 'json', 'parquet']), default='csv', help='Output format')
@click.pass_context
def calculate(ctx, input: str, output: str, start_date: datetime, end_date: datetime, 
              geography: str, weighting: str, format: str):
    """Calculate house price index from transaction data"""
    logger.info(f"Calculating HPI from {input}")
    
    # Load data
    loader = DataLoader(input)
    transactions = loader.load_transactions(validate=True)
    logger.info(f"Loaded {len(transactions)} transactions")
    
    # Calculate index
    calculator = HPICalculator(config=ctx.obj.get('config'))
    index = calculator.calculate(
        transactions=transactions,
        start_date=start_date.date(),
        end_date=end_date.date(),
        geography_level=geography,
        weighting_scheme=weighting
    )
    
    # Save results
    output_path = Path(output)
    if format == 'csv':
        index.to_csv(output_path, index=False)
    elif format == 'json':
        index.to_json(output_path, orient='records', date_format='iso')
    elif format == 'parquet':
        index.to_parquet(output_path, engine='pyarrow')
    
    logger.info(f"Index saved to {output_path}")
    
    # Print summary
    click.echo(f"\nIndex Summary:")
    click.echo(f"Period: {start_date.date()} to {end_date.date()}")
    click.echo(f"Geography: {geography}")
    click.echo(f"Weighting: {weighting}")
    click.echo(f"Records: {len(index)}")
    click.echo(f"Average change: {index['index_value'].pct_change().mean():.2%}")


@cli.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True, help='Input transaction file')
@click.option('--output', '-o', type=click.Path(), help='Output report file')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'html']), default='text')
@click.pass_context
def quality(ctx, input: str, output: Optional[str], format: str):
    """Analyze data quality and generate report"""
    logger.info(f"Analyzing data quality for {input}")
    
    # Load data
    loader = DataLoader(input)
    transactions = loader.load_transactions(validate=False)
    
    # Analyze quality
    analyzer = QualityAnalyzer()
    report = analyzer.analyze(transactions)
    
    # Format report
    if format == 'text':
        report_text = _format_quality_report_text(report)
    elif format == 'json':
        report_text = json.dumps(report, indent=2, default=str)
    elif format == 'html':
        report_text = _format_quality_report_html(report)
    
    # Output report
    if output:
        Path(output).write_text(report_text)
        logger.info(f"Report saved to {output}")
    else:
        click.echo(report_text)


@cli.command()
@click.option('--output', '-o', type=click.Path(), required=True, help='Output directory')
@click.option('--num-cbsas', default=5, type=int, help='Number of CBSAs')
@click.option('--tracts-per-cbsa', default=10, type=int, help='Tracts per CBSA')
@click.option('--properties-per-tract', default=100, type=int, help='Properties per tract')
@click.option('--start-year', default=2020, type=int, help='Start year')
@click.option('--end-year', default=2023, type=int, help='End year')
@click.option('--transactions-per-year', default=1000, type=int, help='Transactions per year')
@click.pass_context
def generate(ctx, output: str, num_cbsas: int, tracts_per_cbsa: int, 
             properties_per_tract: int, start_year: int, end_year: int,
             transactions_per_year: int):
    """Generate synthetic test data"""
    logger.info("Generating synthetic data")
    
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create generator
    generator = SyntheticDataGenerator(
        num_cbsas=num_cbsas,
        tracts_per_cbsa=tracts_per_cbsa,
        properties_per_tract=properties_per_tract
    )
    
    # Generate data
    transactions = generator.generate_transactions(
        start_date=date(start_year, 1, 1),
        end_date=date(end_year, 12, 31),
        transactions_per_year=transactions_per_year
    )
    
    # Save data
    transactions_df = generator.transactions_to_dataframe(transactions)
    transactions_df.to_csv(output_path / 'transactions.csv', index=False)
    
    # Save metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'num_transactions': len(transactions),
        'num_cbsas': num_cbsas,
        'tracts_per_cbsa': tracts_per_cbsa,
        'properties_per_tract': properties_per_tract,
        'date_range': f"{start_year}-{end_year}"
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Generated {len(transactions)} transactions")
    logger.info(f"Data saved to {output_path}")


@cli.command()
@click.option('--workers', '-w', default=4, type=int, help='Number of worker threads')
@click.option('--timeout', '-t', default=3600, type=int, help='Job timeout in seconds')
@click.option('--result-path', '-r', type=click.Path(), help='Results directory')
@click.pass_context
def worker(ctx, workers: int, timeout: int, result_path: Optional[str]):
    """Start batch processing worker"""
    logger.info(f"Starting batch worker with {workers} threads")
    
    # Create processor
    processor = BatchProcessor(
        max_workers=workers,
        job_timeout=timeout,
        result_path=Path(result_path) if result_path else Path('batch_results')
    )
    
    # Start processing
    processor.start()
    
    try:
        # Keep running
        import signal
        signal.pause()
    except KeyboardInterrupt:
        logger.info("Shutting down worker")
        processor.stop()


@cli.command()
@click.option('--job-file', '-j', type=click.Path(exists=True), required=True, help='Job definition file (YAML/JSON)')
@click.option('--priority', '-p', type=click.Choice(['low', 'normal', 'high', 'critical']), default='normal')
@click.option('--wait/--no-wait', default=False, help='Wait for job completion')
@click.pass_context
def submit(ctx, job_file: str, priority: str, wait: bool):
    """Submit batch job for processing"""
    # Load job definition
    with open(job_file) as f:
        if job_file.endswith('.yaml') or job_file.endswith('.yml'):
            job_def = yaml.safe_load(f)
        else:
            job_def = json.load(f)
    
    # Create job
    job = BatchJob(
        job_id=job_def.get('id', f"job_{datetime.now().timestamp()}"),
        name=job_def['name'],
        pipeline=job_def['pipeline'],
        context=job_def['context'],
        priority=JobPriority[priority.upper()],
        metadata=job_def.get('metadata', {})
    )
    
    # Submit job
    # Note: In real implementation, this would connect to the batch service
    logger.info(f"Submitted job {job.job_id}")
    
    if wait:
        logger.info("Waiting for job completion...")
        # In real implementation, would poll for status


@cli.group()
def db():
    """Database management commands"""
    pass


@db.command()
@click.pass_context
def init(ctx):
    """Initialize database"""
    logger.info("Initializing database")
    # Database initialization logic here
    click.echo("Database initialized")


@db.command()
@click.pass_context
def migrate(ctx):
    """Run database migrations"""
    logger.info("Running database migrations")
    # Migration logic here
    click.echo("Migrations completed")


def _format_quality_report_text(report: Dict[str, Any]) -> str:
    """Format quality report as text"""
    lines = [
        "Data Quality Report",
        "=" * 50,
        f"Total Transactions: {report['total_transactions']}",
        f"Valid Pairs: {report['valid_pairs']}",
        f"Invalid Pairs: {report['invalid_pairs']}",
        f"Validation Rate: {report['validation_rate']:.1%}",
        "",
        "Quality Issues:",
        "-" * 30,
    ]
    
    for issue, count in report['quality_issues'].items():
        lines.append(f"  {issue}: {count}")
    
    lines.extend([
        "",
        "Geographic Coverage:",
        "-" * 30,
        f"  CBSAs: {report['geographic_coverage']['cbsas']}",
        f"  Tracts: {report['geographic_coverage']['tracts']}",
        f"  Coverage Rate: {report['geographic_coverage']['coverage_rate']:.1%}",
    ])
    
    return "\n".join(lines)


def _format_quality_report_html(report: Dict[str, Any]) -> str:
    """Format quality report as HTML"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #0066cc; }}
        </style>
    </head>
    <body>
        <h1>Data Quality Report</h1>
        
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Transactions</td><td class="metric">{report['total_transactions']:,}</td></tr>
            <tr><td>Valid Pairs</td><td class="metric">{report['valid_pairs']:,}</td></tr>
            <tr><td>Invalid Pairs</td><td class="metric">{report['invalid_pairs']:,}</td></tr>
            <tr><td>Validation Rate</td><td class="metric">{report['validation_rate']:.1%}</td></tr>
        </table>
        
        <h2>Quality Issues</h2>
        <table>
            <tr><th>Issue Type</th><th>Count</th></tr>
    """
    
    for issue, count in report['quality_issues'].items():
        html += f"<tr><td>{issue}</td><td>{count:,}</td></tr>"
    
    html += """
        </table>
        
        <h2>Geographic Coverage</h2>
        <table>
            <tr><th>Level</th><th>Count</th></tr>
    """
    
    html += f"""
            <tr><td>CBSAs</td><td>{report['geographic_coverage']['cbsas']}</td></tr>
            <tr><td>Tracts</td><td>{report['geographic_coverage']['tracts']}</td></tr>
            <tr><td>Coverage Rate</td><td>{report['geographic_coverage']['coverage_rate']:.1%}</td></tr>
        </table>
    </body>
    </html>
    """
    
    return html


def main():
    """Main entry point"""
    cli(obj={})


if __name__ == '__main__':
    main()