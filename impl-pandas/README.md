# HPI-FHFA Implementation

Implementation of the FHFA House Price Index methodology using Python and Pandas.

## Overview

This project implements a flexible method of house price index construction using repeat-sales aggregates, based on FHFA Working Paper 21-01. The system constructs granular tract-level indices that can be aggregated using various weighting schemes.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from hpi_fhfa.data import SyntheticDataGenerator, GeographicDataGenerator
from hpi_fhfa.models.validators import DataValidator

# Generate synthetic data
data_gen = SyntheticDataGenerator()
transaction_data = data_gen.generate_complete_dataset(
    start_year=1975,
    end_year=2023,
    num_cbsas=20,
    num_tracts=1000,
    num_properties=50000
)

# Generate geographic data
geo_gen = GeographicDataGenerator()
geo_data = geo_gen.generate_complete_geographic_data(num_tracts=1000)

# Validate transaction pairs
validator = DataValidator()
validation_report = validator.generate_validation_report(transaction_data['pairs'])
print(f"Valid pairs: {validation_report['validation_rate']:.1%}")
```

## Features

- **Data Models**: Comprehensive data models for transactions, geography, and demographics
- **Synthetic Data Generation**: Generate realistic housing transaction data from 1975 to present
- **Geographic Data**: Create synthetic tract geometries and CBSA definitions
- **Data Validation**: Quality control filters for transaction pairs
- **Data Transformation**: Tools for preparing data for repeat-sales regression

## Project Structure

```
impl-pandas/
├── src/
│   └── hpi_fhfa/
│       ├── models/          # Data models
│       ├── data/           # Data generation and loading
│       ├── core/           # Core algorithms (Phase 3+)
│       ├── weights/        # Weight calculations (Phase 3+)
│       └── api/            # API endpoints (Phase 5+)
├── tests/                  # Unit and integration tests
├── data/                   # Data storage
├── notebooks/             # Jupyter notebooks
└── scripts/               # Utility scripts
```

## Testing

```bash
pytest tests/
pytest --cov=hpi_fhfa tests/
```

## License

MIT License