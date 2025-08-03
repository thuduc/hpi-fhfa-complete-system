# Quick Start Guide

This guide will help you get started with HPI-FHFA in just a few minutes.

## Basic Usage

### 1. Generate Sample Data

First, let's create some sample data to work with:

```python
from hpi_fhfa.data_generation import SyntheticDataGenerator
from datetime import date

# Create generator
generator = SyntheticDataGenerator(
    num_cbsas=5,
    tracts_per_cbsa=10,
    properties_per_tract=100
)

# Generate transactions
transactions = generator.generate_transactions(
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    transactions_per_year=1000
)

print(f"Generated {len(transactions)} transactions")
```

### 2. Calculate Basic Index

```python
from hpi_fhfa import HPICalculator

# Initialize calculator
calculator = HPICalculator()

# Calculate index
index = calculator.calculate(
    transactions=transactions,
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    geography_level='cbsa',
    weighting_scheme='sample'
)

# View results
print(index.head())
```

### 3. Visualize Results

```python
import matplotlib.pyplot as plt

# Plot index over time
index.plot(y='index_value', figsize=(10, 6))
plt.title('House Price Index Over Time')
plt.xlabel('Date')
plt.ylabel('Index Value (Base=100)')
plt.show()
```

## Advanced Examples

### Using Different Weight Schemes

```python
# Value-weighted index
value_index = calculator.calculate(
    transactions=transactions,
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    geography_level='cbsa',
    weighting_scheme='value'
)

# Demographic-weighted index
demo_index = calculator.calculate(
    transactions=transactions,
    start_date=date(2020, 1, 1),
    end_date=date(2023, 12, 31),
    geography_level='cbsa',
    weighting_scheme='college'
)
```

### Geographic Aggregation

```python
# Calculate at different geographic levels
tract_index = calculator.calculate(
    transactions=transactions,
    geography_level='tract'
)

cbsa_index = calculator.calculate(
    transactions=transactions,
    geography_level='cbsa'
)

national_index = calculator.calculate(
    transactions=transactions,
    geography_level='national'
)
```

### Quality Metrics and Outlier Detection

```python
from hpi_fhfa.quality import QualityAnalyzer

# Analyze data quality
analyzer = QualityAnalyzer()
quality_report = analyzer.analyze(transactions)

print(f"Valid pairs: {quality_report['valid_pairs']}")
print(f"Outliers detected: {quality_report['outliers']}")
print(f"Coverage: {quality_report['geographic_coverage']}%")
```

## Using the REST API

### Starting the API Server

```bash
# Start the API server
hpi-fhfa serve --port 8000 --data-path ./data
```

### Making API Requests

```python
import requests
import json

# Calculate index via API
response = requests.post(
    'http://localhost:8000/api/v1/index/calculate',
    json={
        'start_date': '2020-01-01',
        'end_date': '2023-12-31',
        'geography_level': 'cbsa',
        'weighting_scheme': 'sample'
    }
)

result = response.json()
print(f"Index calculated: {result['summary']}")
```

## Batch Processing

```python
from hpi_fhfa.pipeline import BatchProcessor, BatchJob
import tempfile

# Create batch processor
with tempfile.TemporaryDirectory() as tmpdir:
    processor = BatchProcessor(
        max_workers=4,
        result_path=Path(tmpdir)
    )
    
    # Submit multiple jobs
    jobs = []
    for year in range(2020, 2024):
        job = BatchJob(
            job_id=f"hpi_{year}",
            name=f"HPI Calculation {year}",
            pipeline="hpi_calculation",
            context={
                'start_date': f'{year}-01-01',
                'end_date': f'{year}-12-31',
                'geography_level': 'cbsa'
            }
        )
        jobs.append(job)
    
    # Process jobs
    job_ids = processor.submit_batch(jobs)
    print(f"Submitted {len(job_ids)} jobs")
```

## Common Patterns

### Loading Data from Files

```python
from hpi_fhfa.data_loader import DataLoader

# Load from CSV
loader = DataLoader('data/transactions.csv')
transactions = loader.load_transactions()

# Load with validation
transactions = loader.load_transactions(
    validate=True,
    quality_filters=True
)
```

### Saving Results

```python
# Save to CSV
index.to_csv('output/hpi_results.csv')

# Save to multiple formats
from hpi_fhfa.io import save_results

save_results(
    index,
    'output/hpi_results',
    formats=['csv', 'json', 'parquet']
)
```

### Custom Configuration

```python
# Create custom configuration
config = {
    'quality_filters': {
        'min_days_between_sales': 180,
        'max_price_ratio': 10.0,
        'min_price_ratio': 0.1
    },
    'regression': {
        'method': 'robust',
        'max_iterations': 50
    },
    'aggregation': {
        'min_pairs_per_tract': 10,
        'supertract_threshold': 30
    }
}

# Use custom config
calculator = HPICalculator(config=config)
```

## Next Steps

- Explore the [User Guide](user_guide/index.md) for detailed tutorials
- Review [API Reference](api_reference/index.md) for complete documentation
- See [Examples](https://github.com/yourusername/hpi-fhfa/tree/main/examples) for more use cases
- Check [Performance Guide](user_guide/performance.md) for optimization tips