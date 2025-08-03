"""Data loading and transformation utilities"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json


class DataLoader:
    """Load data from various sources"""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        
    def load_transactions(self, filename: str = 'transactions.csv') -> pd.DataFrame:
        """Load transaction data"""
        filepath = self.data_dir / filename
        
        df = pd.read_csv(filepath)
        
        # Convert date columns
        date_cols = ['sale_date', 'first_sale_date', 'second_sale_date']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        return df
    
    def load_pairs(self, filename: str = 'pairs.csv') -> pd.DataFrame:
        """Load repeat-sales pairs"""
        filepath = self.data_dir / filename
        
        df = pd.read_csv(filepath)
        
        # Convert date columns
        df['first_sale_date'] = pd.to_datetime(df['first_sale_date'])
        df['second_sale_date'] = pd.to_datetime(df['second_sale_date'])
        
        return df
    
    def load_geographic_data(self, 
                           tract_file: str = 'tracts.geojson',
                           cbsa_file: str = 'cbsas.csv',
                           adjacency_file: str = 'tract_adjacency.json'
                           ) -> Dict[str, any]:
        """Load geographic data"""
        # Load tract geometries
        tract_gdf = gpd.read_file(self.data_dir / tract_file)
        
        # Load CBSA data
        cbsa_df = pd.read_csv(self.data_dir / cbsa_file)
        
        # Load adjacency
        with open(self.data_dir / adjacency_file, 'r') as f:
            adjacency = json.load(f)
            
        return {
            'tracts': tract_gdf,
            'cbsas': cbsa_df,
            'adjacency': adjacency
        }
    
    def load_demographic_data(self, filename: str = 'demographics.csv') -> pd.DataFrame:
        """Load demographic data"""
        filepath = self.data_dir / filename
        return pd.read_csv(filepath)
    
    def load_market_indices(self, filename: str = 'market_indices.csv') -> pd.DataFrame:
        """Load market index data"""
        filepath = self.data_dir / filename
        return pd.read_csv(filepath)
    
    def save_results(self, 
                    results: Dict[str, pd.DataFrame],
                    output_dir: Optional[Union[str, Path]] = None) -> None:
        """Save analysis results"""
        if output_dir is None:
            output_dir = self.data_dir / 'results'
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(exist_ok=True)
        
        for name, df in results.items():
            if isinstance(df, gpd.GeoDataFrame):
                df.to_file(output_dir / f'{name}.geojson', driver='GeoJSON')
            else:
                df.to_csv(output_dir / f'{name}.csv', index=False)
                

class DataTransformer:
    """Transform and prepare data for analysis"""
    
    @staticmethod
    def calculate_half_pairs(pairs_df: pd.DataFrame,
                           year: int,
                           window_years: int = 2) -> pd.Series:
        """
        Calculate half-pairs count by tract for a given year
        
        Half-pairs include transactions in year and year-1
        """
        # Filter to relevant years
        year_start = year - window_years + 1
        year_end = year
        
        # Transactions where either sale is in the window
        mask = (
            (pairs_df['first_sale_date'].dt.year >= year_start) & 
            (pairs_df['first_sale_date'].dt.year <= year_end)
        ) | (
            (pairs_df['second_sale_date'].dt.year >= year_start) & 
            (pairs_df['second_sale_date'].dt.year <= year_end)
        )
        
        relevant_pairs = pairs_df[mask]
        
        # Count by tract
        half_pairs = relevant_pairs.groupby('tract_id').size()
        half_pairs.name = f'half_pairs_{year}'
        
        return half_pairs
    
    @staticmethod
    def create_panel_data(pairs_df: pd.DataFrame,
                         start_year: int,
                         end_year: int) -> pd.DataFrame:
        """Create panel dataset with counts by tract and year"""
        panel_data = []
        
        for year in range(start_year, end_year + 1):
            # Get half-pairs for this year
            half_pairs = DataTransformer.calculate_half_pairs(pairs_df, year)
            
            # Create year data
            year_data = half_pairs.reset_index()
            year_data['year'] = year
            year_data.columns = ['tract_id', 'half_pairs', 'year']
            
            panel_data.append(year_data)
            
        return pd.concat(panel_data, ignore_index=True)
    
    @staticmethod
    def merge_tract_demographics(tract_df: pd.DataFrame,
                               demographic_df: pd.DataFrame,
                               year: Optional[int] = None) -> pd.DataFrame:
        """Merge tract data with demographics"""
        if year is not None:
            demo_year = demographic_df[demographic_df['year'] == year]
        else:
            # Use most recent year for each tract
            demo_year = demographic_df.sort_values('year').groupby('tract_id').last()
            
        # Merge
        merged = tract_df.merge(
            demo_year,
            on='tract_id',
            how='left',
            suffixes=('', '_demo')
        )
        
        return merged
    
    @staticmethod
    def create_design_matrix(pairs_df: pd.DataFrame,
                           periods: List[str]) -> pd.DataFrame:
        """
        Create design matrix for repeat-sales regression
        
        Returns sparse matrix representation as DataFrame
        """
        # Create period mapping
        period_map = {period: i for i, period in enumerate(periods)}
        
        # Extract period indicators
        pairs_df['first_period'] = pairs_df['first_sale_date'].dt.to_period('A').astype(str)
        pairs_df['second_period'] = pairs_df['second_sale_date'].dt.to_period('A').astype(str)
        
        # Create design matrix entries
        design_entries = []
        
        for idx, row in pairs_df.iterrows():
            # First sale: -1
            if row['first_period'] in period_map:
                design_entries.append({
                    'pair_idx': idx,
                    'period_idx': period_map[row['first_period']],
                    'value': -1
                })
            
            # Second sale: +1  
            if row['second_period'] in period_map:
                design_entries.append({
                    'pair_idx': idx,
                    'period_idx': period_map[row['second_period']],
                    'value': 1
                })
                
        return pd.DataFrame(design_entries)
    
    @staticmethod
    def filter_valid_pairs(pairs_df: pd.DataFrame,
                         min_price: float = 10000,
                         max_cagr: float = 0.30,
                         min_months: int = 12) -> pd.DataFrame:
        """Apply quality filters to transaction pairs"""
        df = pairs_df.copy()
        
        # Calculate metrics
        df['months_diff'] = (
            (df['second_sale_date'] - df['first_sale_date']).dt.days / 30.44
        )
        
        df['years_diff'] = df['months_diff'] / 12
        
        df['appreciation'] = df['second_sale_price'] / df['first_sale_price']
        
        df['cagr'] = np.where(
            df['years_diff'] > 0,
            np.power(df['appreciation'], 1 / df['years_diff']) - 1,
            0
        )
        
        # Apply filters
        valid_mask = (
            (df['first_sale_price'] >= min_price) &
            (df['second_sale_price'] >= min_price) &
            (df['months_diff'] >= min_months) &
            (df['cagr'].abs() <= max_cagr) &
            (df['appreciation'] <= 10.0) &
            (df['appreciation'] >= 0.75)
        )
        
        filtered_df = df[valid_mask].copy()
        
        print(f"Filtered {len(pairs_df)} pairs to {len(filtered_df)} valid pairs")
        print(f"Rejection rate: {1 - len(filtered_df)/len(pairs_df):.1%}")
        
        return filtered_df
    
    @staticmethod 
    def prepare_regression_data(pairs_df: pd.DataFrame,
                              tract_id: Optional[str] = None,
                              cbsa_id: Optional[str] = None) -> Dict[str, any]:
        """Prepare data for repeat-sales regression"""
        # Filter by geography if specified
        if tract_id is not None:
            data = pairs_df[pairs_df['tract_id'] == tract_id].copy()
        elif cbsa_id is not None:
            data = pairs_df[pairs_df['cbsa_id'] == cbsa_id].copy()
        else:
            data = pairs_df.copy()
            
        if len(data) == 0:
            raise ValueError("No data found for specified geography")
            
        # Calculate log price differences
        data['log_price_diff'] = np.log(data['second_sale_price']) - np.log(data['first_sale_price'])
        
        # Get unique periods
        all_periods = pd.concat([
            data['first_sale_date'].dt.to_period('A'),
            data['second_sale_date'].dt.to_period('A')
        ]).unique()
        all_periods = sorted(all_periods)
        
        # Create design matrix
        design_df = DataTransformer.create_design_matrix(
            data, 
            [str(p) for p in all_periods]
        )
        
        return {
            'pairs': data,
            'design_matrix': design_df,
            'periods': all_periods,
            'n_pairs': len(data),
            'n_periods': len(all_periods)
        }