HPI-FHFA Documentation
======================

Welcome to the HPI-FHFA documentation. This project implements the Federal Housing Finance Agency's
House Price Index methodology using modern Python libraries.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   user_guide/index
   api_reference/index
   deployment/index
   development/index
   faq

Overview
--------

The HPI-FHFA system provides:

* **Repeat-Sales Analysis**: Implementation of the Bailey-Muth-Nourse methodology
* **Geographic Aggregation**: Multi-level index calculation from census tracts to national level
* **Demographic Weighting**: Six different weighting schemes including demographic-based weights
* **Robust Statistics**: Outlier detection and robust regression techniques
* **Production Ready**: REST API, batch processing, monitoring, and caching
* **Comprehensive Testing**: 95%+ test coverage with extensive validation

Key Features
------------

1. **Data Processing**
   
   * Transaction pair validation and quality control
   * Geographic hierarchy management
   * Synthetic data generation for testing

2. **Index Calculation**
   
   * BMN repeat-sales regression
   * Laspeyres value-weighted aggregation
   * Supertract construction for sparse areas

3. **Advanced Analytics**
   
   * Outlier detection using Cook's distance
   * Robust regression with M-estimators
   * Sensitivity analysis tools

4. **Production Systems**
   
   * RESTful API with comprehensive endpoints
   * Batch processing with priority queues
   * Real-time monitoring and alerting
   * Multi-tier caching strategy

Quick Example
-------------

.. code-block:: python

   from hpi_fhfa import HPICalculator, DataLoader
   from datetime import date

   # Load transaction data
   loader = DataLoader('data/transactions.csv')
   transactions = loader.load_transactions()

   # Calculate HPI
   calculator = HPICalculator()
   index = calculator.calculate(
       transactions=transactions,
       start_date=date(2020, 1, 1),
       end_date=date(2023, 12, 31),
       geography_level='cbsa',
       weighting_scheme='sample'
   )

   # Display results
   print(index.head())

Getting Started
---------------

* :doc:`installation` - Install HPI-FHFA and dependencies
* :doc:`quickstart` - Get up and running quickly
* :doc:`user_guide/index` - Detailed guides for common tasks
* :doc:`api_reference/index` - Complete API documentation

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`