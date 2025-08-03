# Introduction

## What is HPI-FHFA?

HPI-FHFA is a comprehensive Python implementation of the Federal Housing Finance Agency's (FHFA) House Price Index (HPI) methodology. It provides tools for calculating repeat-sales price indices at various geographic levels, from census tracts to national aggregates.

## Background

The FHFA House Price Index is a broad measure of the movement of single-family house prices in the United States. The HPI is based on transactions involving conforming, conventional mortgages purchased or securitized by Fannie Mae or Freddie Mac.

### Key Methodological Components

1. **Repeat-Sales Method**: The HPI uses a modified version of the Bailey-Muth-Nourse (BMN) repeat-sales method, which tracks price changes for the same properties over time.

2. **Geographic Hierarchy**: Indices are calculated at multiple levels:
   - Census Tract (finest granularity)
   - Core-Based Statistical Area (CBSA)
   - State
   - National

3. **Weighting Schemes**: Multiple weighting options to address different analytical needs:
   - Sample weights (equal weighting)
   - Value weights (transaction value based)
   - Unit weights (housing unit based)
   - Demographic weights (college education, non-white population)

## Why Use HPI-FHFA?

### For Researchers
- Open-source implementation of FHFA methodology
- Transparent calculations with full documentation
- Extensible framework for custom analyses
- Built-in synthetic data generation for testing

### For Analysts
- Production-ready API for integration
- Batch processing capabilities
- Comprehensive monitoring and alerting
- Performance optimization for large datasets

### For Developers
- Modern Python architecture
- Extensive test coverage (95%+)
- Type hints and comprehensive docstrings
- Docker support for easy deployment

## Core Concepts

### Transaction Pairs
The foundation of repeat-sales analysis. Each pair represents two sales of the same property:

```python
from hpi_fhfa.models import TransactionPair
from datetime import date

pair = TransactionPair(
    property_id="12345",
    first_sale_date=date(2020, 1, 15),
    first_sale_price=250000.0,
    second_sale_date=date(2023, 6, 20),
    second_sale_price=325000.0,
    tract_id="06037123456"
)
```

### Index Calculation Pipeline
The system follows a multi-stage pipeline:

1. **Data Validation**: Quality control filters ensure data integrity
2. **Regression Analysis**: BMN methodology estimates log price changes
3. **Geographic Aggregation**: Tract-level indices aggregate to higher levels
4. **Weight Application**: Various weighting schemes for different use cases

### Quality Control
Built-in FHFA quality filters:
- Minimum time between sales (180 days)
- Price change constraints (0.1x to 10x)
- Geographic validation
- Data completeness checks

## System Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Input    │────▶│  Core Pipeline   │────▶│  Index Output   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
  ┌─────────────┐        ┌──────────────┐        ┌─────────────┐
  │ Validation  │        │  Regression  │        │   REST API  │
  └─────────────┘        └──────────────┘        └─────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
  ┌─────────────┐        ┌──────────────┐        ┌─────────────┐
  │   Quality   │        │ Aggregation  │        │    Cache    │
  │   Control   │        │  & Weights   │        │   System    │
  └─────────────┘        └──────────────┘        └─────────────┘
```

## Next Steps

- Continue to [Installation](installation.md) to set up the system
- See [Quick Start](quickstart.md) for a hands-on tutorial
- Explore the [User Guide](user_guide/index.md) for detailed usage instructions