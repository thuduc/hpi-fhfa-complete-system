#!/usr/bin/env python3
"""Script to generate test data for HPI-FHFA"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hpi_fhfa.data import SyntheticDataGenerator, GeographicDataGenerator
from hpi_fhfa.models.validators import DataValidator
import pandas as pd


def main():
    """Generate complete test dataset"""
    print("HPI-FHFA Test Data Generator")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generators
    print("\nInitializing generators...")
    syn_gen = SyntheticDataGenerator(seed=42)
    geo_gen = GeographicDataGenerator(seed=42)
    
    # Generate transaction data
    print("\nGenerating transaction data...")
    transaction_data = syn_gen.generate_complete_dataset(
        start_year=1975,
        end_year=2023,
        num_cbsas=20,
        num_tracts=1000,
        num_properties=50000,
        target_pairs=25000
    )
    
    # Generate geographic data
    print("\nGenerating geographic data...")
    geo_data = geo_gen.generate_complete_geographic_data(
        num_tracts=1000
    )
    
    # Add demographic data
    print("\nGenerating demographic data...")
    demo_df = syn_gen.add_demographic_data(
        transaction_data['tracts'],
        years=[1990, 2000, 2010, 2017, 2020]
    )
    
    # Validate transaction pairs
    print("\nValidating transaction pairs...")
    validator = DataValidator()
    validation_report = validator.generate_validation_report(
        transaction_data['pairs']
    )
    
    print(f"\nValidation Results:")
    print(f"  Total pairs: {validation_report['total_transactions']:,}")
    print(f"  Valid pairs: {validation_report['valid_transactions']:,}")
    print(f"  Validation rate: {validation_report['validation_rate']:.1%}")
    
    # Save data
    print("\nSaving data files...")
    
    # Transaction data
    transaction_data['transactions'].to_csv(
        output_dir / 'transactions.csv', index=False
    )
    transaction_data['pairs'].to_csv(
        output_dir / 'pairs.csv', index=False
    )
    transaction_data['properties'].to_csv(
        output_dir / 'properties.csv', index=False
    )
    transaction_data['market_indices'].to_csv(
        output_dir / 'market_indices.csv', index=False
    )
    
    # Geographic data
    geo_gen.export_to_files(geo_data, str(output_dir))
    
    # Demographic data
    demo_df.to_csv(output_dir / 'demographics.csv', index=False)
    
    # Save validation report
    pd.DataFrame([validation_report]).to_csv(
        output_dir / 'validation_report.csv', index=False
    )
    
    print(f"\nData saved to: {output_dir}")
    print("\nGeneration complete!")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"  CBSAs: {len(transaction_data['tracts']['cbsa_id'].unique())}")
    print(f"  Tracts: {len(transaction_data['tracts'])}")
    print(f"  Properties: {len(transaction_data['properties'])}")
    print(f"  Transactions: {len(transaction_data['transactions'])}")
    print(f"  Repeat-sales pairs: {len(transaction_data['pairs'])}")
    print(f"  Years covered: 1975-2023 ({2023-1975+1} years)")


if __name__ == '__main__':
    main()