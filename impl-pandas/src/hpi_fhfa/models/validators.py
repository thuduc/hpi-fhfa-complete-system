"""Data validation functions"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import date

from .transaction import TransactionPair
from .geography import Tract, CBSA, Supertract
from .weights import DemographicData, WeightSet


class DataValidator:
    """Centralized data validation"""
    
    # Quality control parameters
    MIN_HALF_PAIRS = 40
    MAX_CAGR = 0.30
    MAX_APPRECIATION = 10.0
    MIN_APPRECIATION = 0.75
    MIN_MONTHS_BETWEEN_SALES = 12
    
    @classmethod
    def validate_transaction_batch(cls, 
                                 transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate batch of transactions and add validation flags
        
        Returns DataFrame with added validation columns
        """
        df = transactions_df.copy()
        
        # Ensure date columns are datetime
        for col in ['first_sale_date', 'second_sale_date']:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        # Calculate derived fields
        df['time_diff_months'] = (
            (df['second_sale_date'].dt.year - df['first_sale_date'].dt.year) * 12 +
            (df['second_sale_date'].dt.month - df['first_sale_date'].dt.month)
        )
        
        df['appreciation_ratio'] = df['second_sale_price'] / df['first_sale_price']
        
        df['time_diff_years'] = (
            (df['second_sale_date'] - df['first_sale_date']).dt.days / 365.25
        )
        
        df['cagr'] = np.where(
            df['time_diff_years'] > 0,
            np.power(df['appreciation_ratio'], 1 / df['time_diff_years']) - 1,
            0
        )
        
        # Apply filters
        df['valid_time_diff'] = df['time_diff_months'].abs() >= cls.MIN_MONTHS_BETWEEN_SALES
        df['valid_cagr'] = df['cagr'].abs() <= cls.MAX_CAGR
        df['valid_appreciation'] = (
            (df['appreciation_ratio'] <= cls.MAX_APPRECIATION) & 
            (df['appreciation_ratio'] >= cls.MIN_APPRECIATION)
        )
        df['valid_prices'] = (df['first_sale_price'] > 0) & (df['second_sale_price'] > 0)
        df['valid_dates'] = df['second_sale_date'] > df['first_sale_date']
        
        # Overall validity
        df['is_valid'] = (
            df['valid_time_diff'] & 
            df['valid_cagr'] & 
            df['valid_appreciation'] & 
            df['valid_prices'] & 
            df['valid_dates']
        )
        
        # Add rejection reason
        df['rejection_reason'] = ''
        df.loc[~df['valid_time_diff'], 'rejection_reason'] = 'Same 12-month period'
        df.loc[~df['valid_cagr'], 'rejection_reason'] = f'CAGR exceeds ±{cls.MAX_CAGR:.0%}'
        df.loc[~df['valid_appreciation'], 'rejection_reason'] = 'Appreciation out of bounds'
        df.loc[~df['valid_prices'], 'rejection_reason'] = 'Invalid prices'
        df.loc[~df['valid_dates'], 'rejection_reason'] = 'Invalid date order'
        
        return df
    
    @classmethod
    def validate_tract_data(cls, tract_gdf) -> Tuple[bool, List[str]]:
        """Validate tract geographic data"""
        errors = []
        
        # Check for required columns (tract_id can be either column or index)
        required_cols = ['cbsa_id', 'state', 'county', 'geometry']
        missing_cols = [col for col in required_cols if col not in tract_gdf.columns]
        
        # Check for tract_id either as column or index
        if 'tract_id' not in tract_gdf.columns and tract_gdf.index.name != 'tract_id':
            missing_cols.append('tract_id')
            
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        
        # Check for duplicate tract IDs
        if 'tract_id' in tract_gdf.columns:
            if tract_gdf.index.name == 'tract_id':
                duplicates = tract_gdf[tract_gdf.index.duplicated()]
                if len(duplicates) > 0:
                    errors.append(f"Duplicate tract IDs: {duplicates.index.tolist()}")
            else:
                duplicates = tract_gdf[tract_gdf['tract_id'].duplicated()]
                if len(duplicates) > 0:
                    errors.append(f"Duplicate tract IDs: {duplicates['tract_id'].tolist()}")
        
        # Check for valid geometries
        if 'geometry' in tract_gdf.columns:
            invalid_geom = tract_gdf[~tract_gdf.geometry.is_valid]
            if len(invalid_geom) > 0:
                errors.append(f"Invalid geometries for tracts: {invalid_geom.index.tolist()}")
        
        # Check for null values
        available_cols = [col for col in required_cols if col in tract_gdf.columns]
        if available_cols:
            null_counts = tract_gdf[available_cols].isnull().sum()
            null_cols = null_counts[null_counts > 0]
            if len(null_cols) > 0:
                errors.append(f"Null values found: {null_cols.to_dict()}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_supertract(cls, 
                          supertract: Supertract,
                          year_pairs_count: dict) -> Tuple[bool, Optional[str]]:
        """Validate supertract meets minimum requirements"""
        # Check minimum half-pairs
        if supertract.half_pairs_count < cls.MIN_HALF_PAIRS:
            return False, f"Insufficient half-pairs: {supertract.half_pairs_count} < {cls.MIN_HALF_PAIRS}"
        
        # Verify component tracts exist
        for tract_id in supertract.component_tract_ids:
            if tract_id not in year_pairs_count:
                return False, f"Component tract {tract_id} not found in transaction data"
        
        # Verify half-pairs calculation
        expected_count = sum(
            year_pairs_count.get(tract_id, 0) 
            for tract_id in supertract.component_tract_ids
        )
        if expected_count != supertract.half_pairs_count:
            return False, f"Half-pairs mismatch: expected {expected_count}, got {supertract.half_pairs_count}"
        
        return True, None
    
    @classmethod
    def validate_weights(cls, weight_set: WeightSet) -> Tuple[bool, Optional[str]]:
        """Validate weight set"""
        # Check for empty weights
        if not weight_set.weights:
            return False, "Empty weight set"
        
        # Check all weights are non-negative
        negative_weights = [
            (k, v) for k, v in weight_set.weights.items() if v < 0
        ]
        if negative_weights:
            return False, f"Negative weights found: {negative_weights}"
        
        # Check normalization
        if not weight_set.is_normalized:
            total = sum(weight_set.weights.values())
            return False, f"Weights not normalized: sum = {total}"
        
        return True, None
    
    @classmethod
    def generate_validation_report(cls, 
                                 transactions_df: pd.DataFrame) -> dict:
        """Generate comprehensive validation report"""
        validated_df = cls.validate_transaction_batch(transactions_df)
        
        report = {
            'total_transactions': len(validated_df),
            'valid_transactions': validated_df['is_valid'].sum(),
            'invalid_transactions': (~validated_df['is_valid']).sum(),
            'validation_rate': validated_df['is_valid'].mean(),
            'rejection_reasons': validated_df[~validated_df['is_valid']]['rejection_reason'].value_counts().to_dict(),
            'cagr_stats': {
                'mean': validated_df[validated_df['is_valid']]['cagr'].mean(),
                'std': validated_df[validated_df['is_valid']]['cagr'].std(),
                'min': validated_df[validated_df['is_valid']]['cagr'].min(),
                'max': validated_df[validated_df['is_valid']]['cagr'].max()
            },
            'time_diff_stats': {
                'mean_years': validated_df[validated_df['is_valid']]['time_diff_years'].mean(),
                'median_years': validated_df[validated_df['is_valid']]['time_diff_years'].median(),
                'min_years': validated_df[validated_df['is_valid']]['time_diff_years'].min(),
                'max_years': validated_df[validated_df['is_valid']]['time_diff_years'].max()
            }
        }
        
        return report